import os
import sys
import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiLineString
from map2loop import m2l_utils
import warnings
import numpy as np
import pandas as pd
from math import sqrt
from map2loop.m2l_utils import print
from shapely.wkt import loads
import re
from osgeo import ogr
from shapely.wkt import loads


# explodes polylines and modifies objectid for exploded parts
def explode_polylines(indf, c_l, dst_crs):
    # indf = gpd.GeoDataFrame.from_file(indata)
    outdf = gpd.GeoDataFrame(columns=indf.columns, crs=dst_crs)
    for idx, row in indf.iterrows():
        if type(row.geometry) == LineString:
            outdf = outdf.append(row, ignore_index=True)
        if type(row.geometry) == MultiLineString:
            multdf = gpd.GeoDataFrame(columns=indf.columns, crs=dst_crs)
            recs = len(row.geometry)
            multdf = multdf.append([row]*recs, ignore_index=True)
            i = 0
            for geom in range(recs):
                multdf.loc[geom, 'geometry'] = row.geometry[geom]
                multdf.loc[geom, c_l['o']] = str(
                    multdf.loc[geom, c_l['o']])+'_'+str(i)
                print('map2loop warning: Fault_' +
                      multdf.loc[geom, c_l['o']], 'is one of a set of duplicates, so renumbering')
                i = i+1
            outdf = outdf.append(multdf, ignore_index=True)
    return outdf


def check_map(structure_file, geology_file, fault_file, mindep_file, fold_file, tmp_path, bbox, c_l, dst_crs, local_paths, drift_prefix):

    # print(gpd.read_file(geology_file).columns)
    # print(gpd.read_file(fault_file).columns)
    # print(gpd.read_file(fold_file).columns)
    # print(gpd.read_file(mindep_file).columns)

    y_point_list = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
    x_point_list = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
    bbox_geom = Polygon(zip(x_point_list, y_point_list))
    polygo = gpd.GeoDataFrame(index=[0], crs=dst_crs, geometry=[bbox_geom])

    m2l_errors = []
    m2l_warnings = []
    for file_name in (structure_file, geology_file, fault_file, fold_file):
        if not file_name.startswith("http") and not os.path.isfile(file_name):
            m2l_errors.append('file '+file_name+' not found')

    # Process orientation points

    if (os.path.isfile(structure_file) or structure_file.startswith("http") or not local_paths):
        orientations2 = gpd.read_file(structure_file, bbox=bbox)
        if(c_l['sf'] == c_l['ds']):
            new_code = 'NEW_'+c_l['sf']
            new_code = new_code[:10]
            orientations = orientations2.rename(
                columns={c_l['sf']: new_code}, errors="raise")
            m2l_warnings.append('To avoid conflict with geology field of same name, orientation field named "' +
                                str(c_l['sf'])+'" renamed to "'+new_code+'"')
            c_l['sf'] = new_code
        else:
            new_code = ''
            orientations = orientations2.copy()
        if(c_l['bo'] == c_l['ds'] and not new_code == ''):
            c_l['bo'] = new_code

        if(len(orientations) < 2):
            m2l_errors.append(
                'not enough orientations to complete calculations (need at least 2), projection may be inconsistent')

        orientations = orientations.replace(r'^\s+$', np.nan, regex=True)
        orientations = orientations[orientations[c_l['d']] != -999]
        for code in ('sf', 'd', 'dd', 'gi'):
            if not c_l[code] in orientations.columns:
                if(code == 'sf'):
                    orientations[c_l[code]] = 'Bed'
                    m2l_warnings.append(
                        'field named "'+str(c_l[code])+'" added with default value "Bed"')
                elif(not code == 'gi'):
                    m2l_errors.append('"'+c_l[code]+'" field needed')
                else:
                    m2l_warnings.append(
                        'field named "'+str(c_l[code])+'" added with default value')
                    orientations[c_l[code]] = np.arange(len(orientations))
            else:
                nans = orientations[c_l[code]].isnull().sum()
                if(nans > 0):
                    m2l_warnings.append(''+str(nans)+' NaN/blank found in column "'+str(
                        c_l[code])+'" of orientations file, replacing with 0')
                    orientations[c_l[code]].fillna("0", inplace=True)

        unique_o = set(orientations[c_l['gi']])

        if(not len(unique_o) == len(orientations)):
            m2l_warnings.append('duplicate orientation point unique IDs')
        show_metadata(orientations, "orientations layer")
    # Process geology polygons

    if (os.path.isfile(geology_file) or geology_file.startswith("http") or not local_paths):
        geology = gpd.read_file(geology_file, bbox=bbox)
        if(not geology.empty):
            if not c_l['o'] in geology.columns:
                # print(geology.columns)
                geology = geology.reset_index()
                geology[c_l['o']] = geology.index

            unique_g = set(geology[c_l['o']])

            if(not len(unique_g) == len(geology)):
                m2l_warnings.append('duplicate geology polygon unique IDs')

            nans = geology[c_l['c']].isnull().sum()
            if(nans > 0):
                m2l_errors.append(''+str(nans)+' NaN/blank found in column "' +
                                  str(c_l['c'])+'" of geology file, please fix')

            if(c_l['g'] == 'No_col' or not c_l['g'] in geology.columns):
                m2l_warnings.append(
                    'No secondary strat coding for geology polygons')
                c_l['g'] = 'group'
                geology[c_l['g']] = "Top"

            geology = geology.replace(r'^\s+$', np.nan, regex=True)
            geology = geology.replace(',', ' ', regex=True)
            geology[c_l['g']].fillna(geology[c_l['g2']], inplace=True)
            geology[c_l['g']].fillna(geology[c_l['c']], inplace=True)

            if(c_l['r1'] == 'No_col' or not c_l['r1'] in geology.columns):
                m2l_warnings.append('No extra litho for geology polygons')
                c_l['r1'] = 'r1'
                geology[c_l['r1']] = 'Nope'

            if(c_l['r2'] == 'No_col' or not c_l['r2'] in geology.columns):
                m2l_warnings.append('No more extra litho for geology polygons')
                c_l['r2'] = 'r2'
                geology[c_l['r2']] = 'Nope'

            if(c_l['min'] == 'No_col' or not c_l['min'] in geology.columns):
                m2l_warnings.append('No min age for geology polygons')
                c_l['min'] = 'min'
                geology[c_l['min']] = 0

            if(c_l['max'] == 'No_col' or not c_l['max'] in geology.columns):
                m2l_warnings.append('No max age for geology polygons')
                c_l['max'] = 'max'
                geology[c_l['max']] = 100

            if(c_l['c'] == 'No_col' or not c_l['c'] in geology.columns):
                m2l_errors.append(
                    'Must have primary strat coding field for geology polygons')

            for code in ('c', 'g', 'g2', 'ds', 'u', 'r1'):
                if(c_l[code] in geology.columns):

                    geology[c_l[code]].str.replace(",", " ")
                    if(code == 'c' or code == 'g' or code == 'g2'):
                        geology[c_l[code]].str.replace(" ", "_")
                        geology[c_l[code]].str.replace("-", "_")

                    nans = geology[c_l[code]].isnull().sum()
                    if(nans > 0):
                        m2l_warnings.append(''+str(nans)+' NaN/blank found in column "'+str(
                            c_l[code])+'" of geology file, replacing with 0')
                        geology[c_l[code]].fillna("0", inplace=True)
            for drift in drift_prefix:
                geology = geology[~geology[c_l['u']].str.startswith(drift)]

            show_metadata(geology, "geology layer")
        else:
            print('No geology in area, projection may be inconsistent')

    # Process fold polylines
    folds = None
    if (os.path.isfile(fold_file) or not local_paths):
        folds = gpd.read_file(fold_file, bbox=bbox)
        if(len(folds) > 0):
            if not c_l['o'] in folds.columns:
                folds = folds.reset_index()
                folds[c_l['o']] = folds.index
            unique_g = set(folds[c_l['o']])

            if(not len(unique_g) == len(folds)):
                m2l_warnings.append('duplicate fold polyline unique IDs')

            folds = folds[folds[c_l['ff']].str.contains(
                c_l['fold'], case=False)]

            folds = folds.replace(r'^\s+$', np.nan, regex=True)

            for code in ('ff', 't'):
                if(c_l['ff'] == 'No_col' or not c_l['ff'] in folds.columns):
                    m2l_warnings.append('No fold code for fold polylines')
                    c_l['ff'] = 'ff'
                    folds[c_l['ff']] = c_l['fold']

                if(c_l['t'] == 'No_col' or not c_l['t'] in folds.columns):
                    m2l_warnings.append('No fold polarity for fold polylines')
                    c_l['t'] = 't'
                    folds[c_l['t']] = 'None'

                if(c_l[code] in folds.columns):
                    folds[c_l[code]].str.replace(",", " ")

                    nans = folds[c_l[code]].isnull().sum()
                    if(nans > 0):
                        m2l_warnings.append(''+str(nans)+' NaN/blank found in column "'+str(
                            c_l[code])+'" of folds file, replacing with 0')
                        folds[c_l[code]].fillna("0", inplace=True)

            folds_clip = m2l_utils.clip_shp(folds, polygo)
            if(len(folds_clip) > 0):
                folds_explode = explode_polylines(folds_clip, c_l, dst_crs)
                if(len(folds_explode) > len(folds_clip)):
                    m2l_warnings.append(
                        'some folds are MultiPolyLines, and have been split')
                folds_explode.crs = dst_crs

            show_metadata(folds_clip, "fold layer")
        else:
            print('No folds in area, projection may be inconsistent')

    # Process fault polylines

    if (os.path.isfile(fault_file) or fault_file.startswith("http") or not local_paths):
        faults_folds = gpd.read_file(fault_file, bbox=bbox)

        faults = faults_folds[faults_folds[c_l['f']
                                           ].str.contains(c_l['fault'], case=False)]
        faults = faults.replace(r'^\s+$', np.nan, regex=True)

        if not c_l['o'] in faults.columns:
            m2l_warnings.append(
                'field named "'+str(c_l['o'])+'" added with default value')
            faults[c_l['o']] = np.arange(len(faults))

        for code in ('f', 'o', 'fdip', 'fdipdir', 'fdipest'):
            if(c_l['f'] == 'No_col' or not c_l['f'] in faults.columns):
                m2l_warnings.append('No fault type for fault polylines')
                c_l['f'] = 'ftype'
                faults[c_l['f']] = c_l['fault']

            if(c_l['fdip'] == 'No_col' or not c_l['fdip'] in faults.columns):
                m2l_warnings.append('No fault dip for fault polylines')
                c_l['fdip'] = 'fdip'
                faults[c_l['fdip']] = c_l['fdipnull']

            if(c_l['fdipdir'] == 'No_col' or not c_l['fdipdir'] in faults.columns):
                m2l_warnings.append(
                    'No fault dip direction for fault polylines')
                c_l['fdipdir'] = 'fdipdir'
                faults[c_l['fdipdir']] = 0

            if(c_l['fdipest'] == 'No_col' or not c_l['fdipest'] in faults.columns):
                m2l_warnings.append(
                    'No fault dip estimate for fault polylines')
                c_l['fdipest'] = 'fdipest'
                faults[c_l['fdipest']] = 'None'

            if(c_l['fdipest_vals'] == 'No_col' or not c_l['fdipest_vals'] in faults.columns):
                m2l_warnings.append(
                    'No fault dip estimate text for fault polylines')
                c_l['fdipest_vals'] = 'fdipest_vals'
                faults[c_l['fdipest_vals']] = 'None'

            if(c_l['n'] == 'No_col' or not c_l['n'] in faults.columns):
                m2l_warnings.append('No fault name for fault polylines')
                c_l['n'] = 'fname'
                faults[c_l['n']] = 'None'

            if not c_l[code] in faults.columns:
                m2l_errors.append(
                    'field named "'+str(c_l[code])+'" not found in fault/fold file')

            if(c_l[code] in faults.columns):
                nans = faults[c_l[code]].isnull().sum()
                if(nans > 0):
                    m2l_warnings.append(''+str(nans)+' NaN/blank found in column "'+str(
                        c_l[code])+'" of fault file, replacing with -999')
                    faults[c_l[code]].fillna("-999", inplace=True)

        unique_f = set(faults[c_l['o']])

        if(not len(unique_f) == len(faults)):
            m2l_errors.append('duplicate fault/fold polyline unique IDs')

        faults_clip = m2l_utils.clip_shp(faults, polygo)

        if(len(faults_clip) > 0):
            faults_explode = explode_polylines(faults_clip, c_l, dst_crs)
            if(len(faults_explode) > len(faults_clip)):
                m2l_warnings.append(
                    'some faults are MultiPolyLines, and have been split')
            faults_explode.crs = dst_crs

            print("original no of faults:", len(faults_explode))
            lengths = []

            for indx, flt in faults_explode.iterrows():
                if(c_l['fault'].lower() in flt[c_l['f']].lower()):
                    if(flt.geometry.type == 'LineString'):
                        flt_ls = LineString(flt.geometry)
                        dlsx = flt_ls.coords[0][0] - \
                            flt_ls.coords[len(flt_ls.coords)-1][0]
                        dlsy = flt_ls.coords[0][1] - \
                            flt_ls.coords[len(flt_ls.coords)-1][1]
                        strike = sqrt((dlsx*dlsx)+(dlsy*dlsy))
                        lengths.append(strike)
                        #print(flt)
            faults_explode['f_length'] = lengths
            faults_explode = faults_explode[faults_explode['f_length'] > 2500]

            show_metadata(faults_explode, "fault layer")
        else:

            # fault_file='None'
            print('No faults in area, projection may be inconsistent')

    # Process mindep points

    mindeps = None
    if (os.path.isfile(mindep_file) or mindep_file.startswith("http") or not local_paths):
        try:
            # No need to specify mindeps
            mindeps = gpd.read_file(mindep_file, bbox=bbox)
            if(len(mindeps) == 0):
                m2l_warnings.append('no mindeps for analysis')
            else:
                mindeps = mindeps.replace(r'^\s+$', np.nan, regex=True)

                for code in ('msc', 'msn', 'mst', 'mtc', 'mscm', 'mcom'):
                    if(c_l[code] == 'No_col'):
                        mindeps[c_l[code]] = 'No_col'
                    if not c_l[code] in mindeps.columns:
                        m2l_errors.append(
                            'field named "'+str(c_l[code])+'" not found in mineral deposits file')
                    else:
                        nans = mindeps[c_l[code]].isnull().sum()
                        if(nans > 0):
                            m2l_warnings.append(str(
                                nans)+' NaN/blank found in column '+str(c_l[code])+' of mindep file, replacing with 0')
                            mindeps[c_l[code]].fillna("0", inplace=True)
            show_metadata(mindeps, "mindeps layer")

        except Exception as e:
            m2l_warnings.append('no mindeps for analysis')
            mindeps=0

        else:
            mindeps = mindeps.replace(r'^\s+$', np.nan, regex=True)

            for code in ('msc', 'msn', 'mst', 'mtc', 'mscm', 'mcom'):
                if(c_l[code] == 'No_col'):
                    mindeps[c_l[code]] = 'No_col'
                if not c_l[code] in mindeps.columns:
                    m2l_errors.append(
                        'field named "'+str(c_l[code])+'" not found in mineral deposits file')
                else:
                    nans = mindeps[c_l[code]].isnull().sum()
                    if(nans > 0):
                        m2l_warnings.append(str(
                            nans)+' NaN/blank found in column '+str(c_l[code])+' of mindep file, replacing with 0')
                        mindeps[c_l[code]].fillna("0", inplace=True)

            show_metadata(mindeps, "mindeps layer")

        # explode fault/fold multipolylines
        # sometimes faults go off map and come back in again which after clipping creates multipolylines

    if(len(m2l_warnings) > 0):
        print("\nWarnings:")
        warnings.warn('The warnings listed above were issued')
        for w in m2l_warnings:
            print("    ", w)
    if(len(m2l_errors) > 0):
        print("\nErrors:")
        warnings.warn(
            'The errors listed above must be fixed prior to rerunning map2loop')
        for e in m2l_errors:
            print("    ", e)
        sep = '\12'
        raise NameError(sep.join(m2l_errors) +
                        '\n map2loop error: Fix errors before running again')

    if(len(m2l_errors) == 0):

        if folds is not None:
            if(len(folds) > 0):
                if(len(folds_clip) > 0):
                    fold_file = os.path.join(tmp_path, 'folds_clip.shp')
                    folds_explode = folds_explode.dropna(subset=['geometry'])
                    folds_explode.to_file(fold_file)
                else:
                    fold_file = os.path.join(tmp_path, 'folds_clip.shp')
                    print("\nFold layer metadata\n--------------------")
                    print("No folds found, projection may be inconsistent")

        if(len(faults_clip) > 0):
            fault_file = os.path.join(tmp_path, 'faults_clip.shp')
            faults_explode.crs = dst_crs
            faults_explode.to_file(fault_file)
        else:
            fault_file = os.path.join(tmp_path, 'faults_clip.shp')
            print("\nFault layer metadata\n--------------------")
            print("No faults found, projection may be inconsistent")

        geol_clip = gpd.overlay(geology, polygo, how='intersection')

        if(len(geol_clip) > 0):
            geol_gaps=check_gaps(geol_clip)
            if(len(geol_gaps)>0):
                warnings.warn('Gaps between geology polygons, consider editing geology layer or rounding vertices')
                print(geol_gaps)
                #geol_gaps.to_file(os.path.join(tmp_path,'geology_gaps.shp'))
            else:
                print("No gaps between geology polygons")
            geol_overlaps=check_overlaps(geol_clip)
            if(len(geol_overlaps)>0):
                warnings.warn('Overlaps between geology polygons, consider editing geology layer or rounding vertices')
                print(geol_overlaps)
                #geol_overlaps.to_file(os.path.join(tmp_path,'geology_overlaps.shp'))
            else:
                print("No overlaps between geology polygons")
            geol_clip.crs = dst_crs
            geol_file = os.path.join(tmp_path,'geol_clip.shp')
            geol_clip.to_file(geol_file)


        if(len(orientations) > 0):
            structure_file = os.path.join(tmp_path, 'structure_clip.shp')
            orientations.crs = dst_crs
            orientations[c_l['dd']] = pd.to_numeric(orientations[c_l['dd']])
            orientations[c_l['d']] = pd.to_numeric(orientations[c_l['d']])

            orientations.to_file(structure_file)


        try:
            if(len(mindeps) > 0):
                mindep_file = os.path.join(tmp_path, 'mindeps_clip.shp')
                mindeps.crs = dst_crs
                mindeps.to_file(mindep_file)
        except Exception as e:
            print('Not clipping mindeps, dataframe is empty')
        print('\nNo errors found, clipped and updated files saved to tmp')

        return(structure_file, geol_file, fault_file, mindep_file, fold_file, c_l)


def show_metadata(gdf, name):
    if(len(gdf) > 0):
        print("\n", name, " metadata\n--------------------")
        print("    bbox", gdf.total_bounds)
        print("    CRS", gdf.crs)
        print("    # items", len(gdf))
        types = []
        for i, g in gdf.iterrows():
            if(not g.geometry.type in types):
                types.append(g.geometry.type)

        print("    Data types", types)
    else:
        print("\n", name, " metadata\n--------------------")
        print("    empty file, check contents")


#########################################
#
# Checks for polygons overlaps and returns geopandas object of overlapping areas
#
#########################################
def check_overlaps(data_temp):
    data_temp['id'] = data_temp.index

    data_overlaps = gpd.GeoDataFrame()
    for index, row in data_temp.iterrows():
        data_temp1 = data_temp.loc[data_temp.id != row.id, ]
        # check if intersection occured
        overlaps = data_temp1[data_temp1.geometry.overlaps(
            row.geometry)]['id'].tolist()
        if len(overlaps) > 0:
            # compare the area with threshold
            for y in overlaps:
                temp_area = gpd.overlay(
                    data_temp.loc[data_temp.id == y, ], data_temp.loc[data_temp.id == row.id, ], how='intersection')
                temp_area = temp_area.loc[temp_area.geometry.area >= 9e-9]
                if temp_area.shape[0] > 0:
                    data_overlaps = gpd.GeoDataFrame(
                        pd.concat([temp_area, data_overlaps], ignore_index=True), crs=data_temp.crs)
    # get unique of list id
    if(not data_overlaps.empty):
        data_overlaps['sorted'] = data_overlaps.apply(
            lambda y: sorted([y['id_1'], y['id_2']]), axis=1)
        data_overlaps['sorted'] = data_overlaps.sorted.apply(
            lambda y: ''.join(str(y)))
        data_overlaps = data_overlaps.drop_duplicates('sorted')
        data_overlaps = data_overlaps.reset_index()[
            ['id_1', 'id_2', 'geometry']]
        data_overlaps.crs = data_temp.crs
        return(data_overlaps)
    else:
        return('')

#########################################
#
# Checks for gaps between polygons and returns geopandas object of missing sections
#
#########################################


def check_gaps(data_temp):
    data_temp['id'] = data_temp.index
    data_temp['diss_id'] = 100
    data_temp_diss = data_temp.dissolve(by='diss_id')
    interior = data_temp_diss.interiors.values.tolist()[0]
    gap_list = []
    for i in interior:
        gap_list.append(LineString(i))
    data_gaps = gpd.GeoDataFrame(geometry=gap_list, crs=data_temp.crs)
    data_gaps['feature_touches'] = data_gaps.geometry.apply(
        lambda y: data_temp.loc[data_temp.touches(y)]['id'].tolist())
    if(not data_gaps.empty):
        data_gaps.head()
        return(data_gaps)
    else:
        return('')

#########################################
#
# truncates polygon or polyline vertices to given precision and saves out as new shapefile
#
#########################################


def round_vertices(layer_file, precision, output_file):

    simpledec = re.compile(r"\d*\.\d+")

    def mround(match):
        return "{:.{}f}".format(float(match.group()), precision)

    layer_file.geometry = layer_file.geometry.apply(
        lambda x: loads(re.sub(simpledec, mround, x.wkt)))
    layer_file.to_file(output_file)
    print("rounded file written to ",output_file)
    #https://gis.stackexchange.com/questions/321518/rounding-coordinates-to-5-decimals-in-geopandas

def densify(geom):
    wkt = geom.wkt  # Get wkt
    geom = ogr.CreateGeometryFromWkt(wkt) 
    geom.Segmentize(spacing)  #Modify the geometry such it has no segment longer than the given (maximum) length.
    wkt2 = geom.ExportToWkt()
    new = loads(wkt2)
    return new

#spacing=500
#shapefile['geometry'] = shapefile['geometry'].map(densify)


import os
import time

import numpy as np
import pandas as pd
import geopandas as gpd
from map2loop.topology import Topology
from map2loop import m2l_utils
from map2loop import m2l_geometry
from map2loop import m2l_interpolation
import map2model

import networkx as nx
import matplotlib.pyplot as plt
import rasterio
import shapely


class Config(object):
    def __init__(self, geology_file, fault_file, structure_file, mindep_file, bbox_3d, polygon, step_out, dtm_crs, proj_crs, c_l={}):
        self.bbox_3d = bbox_3d
        self.bbox = tuple([bbox_3d["minx"], bbox_3d["miny"],
                           bbox_3d["maxx"], bbox_3d["maxy"]])
        self.polygon = polygon
        self.step_out = step_out
        self.c_l = c_l

        self.dtm_crs = dtm_crs
        self.proj_crs = proj_crs

        # Create file structure for model instance
        self.geology_file = geology_file
        self.structure_file = structure_file
        self.fault_file = fault_file
        self.mindep_file = mindep_file

        self.project_path = 'model-test'

        self.graph_path = self.project_path+'/graph/'
        self.tmp_path = self.project_path+'/tmp/'
        self.data_path = self.project_path+'/data/'
        self.dtm_path = self.project_path+'/dtm/'
        self.output_path = self.project_path+'/output/'
        self.vtk_path = self.project_path+'/vtk/'

        self.fault_file_csv = self.tmp_path + "faults.csv"
        self.structure_file_csv = self.tmp_path + "structure.csv"
        self.geology_file_csv = self.tmp_path + "geology.csv"
        self.mindep_file_csv = self.tmp_path + "mindep.csv"

        self.strat_graph_file = self.graph_path + "graph_strat_NONE.gml"
        self.dtm_file = self.dtm_path+'dtm.tif'
        self.dtm_reproj_file = self.dtm_path+'dtm_rp.tif'

        if(not os.path.isdir(self.project_path)):
            os.mkdir(self.project_path)
        if(not os.path.isdir(self.tmp_path)):
            os.mkdir(self.tmp_path)
        if(not os.path.isdir(self.data_path)):
            os.mkdir(self.data_path)
        if(not os.path.isdir(self.output_path)):
            os.mkdir(self.output_path)
        if(not os.path.isdir(self.dtm_path)):
            os.mkdir(self.dtm_path)
        if(not os.path.isdir(self.vtk_path)):
            os.mkdir(self.vtk_path)
        if(not os.path.isdir(self.graph_path)):
            os.mkdir(self.graph_path)

    def preprocess(self, command=""):
        geology = gpd.read_file(self.geology_file, bbox=self.bbox)
        geology[self.c_l['g']].fillna(geology[self.c_l['g2']], inplace=True)
        geology[self.c_l['g']].fillna(geology[self.c_l['c']], inplace=True)
        faults = gpd.read_file(self.fault_file, bbox=self.bbox)
        structures = gpd.read_file(self.structure_file, bbox=self.bbox)
        mindeps = gpd.read_file(self.mindep_file, bbox=self.bbox)

        self.geology = geology
        self.faults = faults
        self.structures = structures
        self.mindeps = mindeps

        if command == "plot":
            try:
                base = geology.plot(column=self.c_l['c'], figsize=(
                    10, 10), edgecolor='#000000', linewidth=0.2, legend=True)
                leg = base.get_legend()
                leg.set_bbox_to_anchor((1.04, 1))
                self.geology_figure = base.get_figure()

                structures.plot(ax=base, color='none', edgecolor='black')

                faults.plot(ax=base, cmap='rainbow',
                            column=self.c_l['f'], figsize=(10, 10), linewidth=0.4)
                structures[['geometry', self.c_l['gi'],
                            self.c_l['d'], self.c_l['dd']]].plot(ax=base)

                fig = self.polygon.plot(ax=base, color='none', edgecolor='black').set_title(
                    "Input {}".format(self.bbox)).get_figure()
                fig.savefig(self.tmp_path+"/input-data.png")

                print("Input graphic saved to: " +
                      self.tmp_path + "input-fig.png")

                self.export_png()
                plt.show()

                return
            except Exception as e:
                print(e)

    def export_png(self):
        self.geology_figure.savefig(self.tmp_path+"geology.png")
        print("Geology graphic exported to: " + self.tmp_path+"geology.png")

    def export_csv(self):
        # Save geology polygons
        hint_flag = False  # use GSWA strat database to provide topology hints
        sub_geol = self.geology[['geometry', self.c_l['o'], self.c_l['c'], self.c_l['g'],
                                 self.c_l['u'], self.c_l['min'], self.c_l['max'], self.c_l['ds'], self.c_l['r1'], self.c_l['r2']]]
        Topology.save_geol_wkt(
            sub_geol, self.geology_file_csv, self.c_l, hint_flag)

        # Save mineral deposits
        sub_mindep = self.mindeps[['geometry', self.c_l['msc'], self.c_l['msn'],
                                   self.c_l['mst'], self.c_l['mtc'], self.c_l['mscm'], self.c_l['mcom']]]
        Topology.save_mindep_wkt(
            sub_mindep, self.mindep_file_csv, self.c_l)

        # Save orientation data
        sub_pts = self.structures[[
            'geometry', self.c_l['gi'], self.c_l['d'], self.c_l['dd']]]
        Topology.save_structure_wkt(
            sub_pts, self.structure_file_csv, self.c_l)

        # Save faults
        sub_faults = self.faults[['geometry', self.c_l['o'], self.c_l['f']]]
        Topology.save_faults_wkt(sub_faults, self.fault_file_csv, self.c_l)

    def update_parfile(self):
        Topology.save_parfile(self, self.c_l, self.output_path, self.geology_file_csv, self.fault_file_csv, self.structure_file_csv,
                              self.mindep_file_csv, self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], 500.0, 'Fe,Cu,Au,NONE')

    def runMap2Model(self):
        print(map2model.run(self.graph_path, self.geology_file_csv,
                            self.fault_file_csv, self.mindep_file_csv,
                            self.bbox_3d,
                            self.c_l,
                            "Fe,Cu,Au,NONE"))

        print("Resolving ambiguities using ASUD...", end='\toutput_dir:')
        aus = True
        if aus:
            Topology.use_asud(self.strat_graph_file,  self.graph_path)
            self.strat_graph_file = self.graph_path+'ASUD_strat.gml'
        print("Done.")

        print("Generating topology graph display and unit groups...")
        self.G = nx.read_gml(self.strat_graph_file, label='id')
        selected_nodes = [n for n, v in self.G.nodes(data=True) if n >= 0]

        nx.draw_networkx(self.G, pos=nx.kamada_kawai_layout(
            self.G), arrows=True, nodelist=selected_nodes)
        nlist = list(self.G.nodes.data('LabelGraphics'))
        nlist.sort()
        for node in nlist:
            if node[0] >= 0:
                elem = str(node[1]).replace(
                    "{'text':", "").replace(", 'fontSize': 14}", "")
                # second=elem.split(":").replace("'","")
                print(node[0], " ", elem)

        # plt.savefig(self.tmp_path+"topology-fig.png")
        print("Topology figure saved to", self.tmp_path+"topology-fig.png")

        # Save groups of stratigraphic units
        groups, self.glabels, G = Topology.get_series(
            self.strat_graph_file, 'id')
        Topology.save_units(G, self.tmp_path, self.glabels,
                            Australia=True, asud_strat_file="https://gist.githubusercontent.com/yohanderose/3b257dc768fafe5aaf70e64ae55e4c42/raw/8598c7563c1eea5c0cd1080f2c418dc975cc5433/ASUD.csv")

        print("Done")

    def load_dtm(self):
        polygon_ll = self.polygon.to_crs(self.dtm_crs)

        minlong = polygon_ll.total_bounds[0]-self.step_out
        maxlong = polygon_ll.total_bounds[2]+self.step_out
        minlat = polygon_ll.total_bounds[1]-self.step_out
        maxlat = polygon_ll.total_bounds[3]+self.step_out

        print("Fetching DTM... ", end=" bbox:")
        print(minlong, maxlong, minlat, maxlat)
        downloaded = False
        i = 0
        print('Attempt: 0 ', end='')
        while downloaded == False:
            try:
                m2l_utils.get_dtm(self.dtm_file, minlong,
                                  maxlong, minlat, maxlat)
                downloaded = True
            except:
                time.sleep(10)
                i = i+1
                print(' ', i, end='')
        if(i == 100):
            raise NameError(
                'map2loop error: Could not access DTM server after 100 attempts')
        print('Done.')

        geom_rp = m2l_utils.reproject_dtm(
            self.dtm_file, self.dtm_reproj_file, self.dtm_crs, self.proj_crs)
        self.dtm = rasterio.open(self.dtm_reproj_file)
        plt.imshow(self.dtm.read(1), cmap='terrain',
                   vmin=0, vmax=1000)

        plt.title('DTM')
        plt.show()

    def join_features(self):
        # Geology
        self.geol_clip = m2l_utils.explode(self.geology)
        self.geol_clip.crs = self.proj_crs
        self.geol_clip_file = self.tmp_path + "self.geol_clip.shp"
        self.geol_clip.to_file(self.geol_clip_file)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        # Structures
        list1 = ['geometry', self.c_l['d'],
                 self.c_l['dd'], self.c_l['sf'], self.c_l['bo']]
        list2 = list(set(list1))
        sub_pts = self.structures[list2]
        structure_code = gpd.sjoin(
            sub_pts, self.geol_clip, how="left", op="within")

        minx, miny, maxx, maxy = self.bbox
        y_point_list = [miny, miny, maxy, maxy, miny]
        x_point_list = [minx, maxx, maxx, minx, minx]

        bbox_geom = shapely.geometry.Polygon(zip(x_point_list, y_point_list))

        # TODO: 'polygo' is never used
        polygo = gpd.GeoDataFrame(
            index=[0], crs=self.proj_crs, geometry=[bbox_geom])
        is_bed = structure_code[self.c_l['sf']].str.contains(
            self.c_l['bedding'], regex=False)

        structure_clip = structure_code[is_bed]
        structure_clip.crs = self.proj_crs

        if(self.c_l['otype'] == 'strike'):
            structure_clip['azimuth2'] = structure_clip.apply(
                lambda row: row[self.c_l['dd']]+90.0, axis=1)
            self.c_l['dd'] = 'azimuth2'
            self.c_l['otype'] = 'dip direction'

        self.structure_clip = structure_clip[~structure_clip[self.c_l['o']].isnull(
        )]
        self.structure_clip_file = self.tmp_path+'structure_clip.shp'
        self.structure_clip.to_file(self.structure_clip_file)

        # Save geology clips
        Topology.save_group(Topology, self.G, self.tmp_path,
                            self.glabels, self.geol_clip, self.c_l)

    def calc_depth_grid(self):
        dtm = self.dtm

        self.dtb = 0
        self.dtb_null = 0

        print("dtb and dtb_null set to 0")
        return

        # TODO: Paths need to be defined, every function call bellow here that has
        #       a False boolean is referening to the workflow['cover_map'] flag
        # dtb_grid=data_path+'young_cover_grid.tif' #obviously hard-wired for the moment
        # dtb_null='-2147483648' #obviously hard-wired for the moment
        # cover_map_path=data_path+'Young_Cover_FDS_MGA_clean.shp' #obviously hard-wired for the moment
        # dtb_clip=output_path+'young_cover_grid_clip.tif' #obviously hard-wired for the moment
        # cover_dip=10 # dip of cover away from contact
        # cover_spacing=5000 # of contact grid in metres

        dtb_raw = rasterio.open(dtb_grid)

        cover = gpd.read_file(cover_map_path)

        with fiona.open(cover_map_path, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]

        with rasterio.open(dtb_grid) as src:
            out_image, out_transform = rasterio.mask.mask(
                src, shapes, crop=True)
            out_meta = src.meta.copy()

        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        with rasterio.open(dtb_clip, "w", **out_meta) as dest:
            dest.write(out_image)

        dtb = rasterio.open(dtb_clip)

        m2l_geometry.process_cover(output_path, dtm, dtb, dtb_null, cover,
                                   workflow['cover_map'], cover_dip, bbox, proj_crs, cover_spacing, contact_decimate=3, use_vector=True, use_grid=True)

    def export_orientations(self):
        # store every nth orientation point (in object order)
        orientation_decimate = 0
        m2l_geometry.save_orientations(
            self.structure_clip, self.output_path, self.c_l, orientation_decimate, self.dtm, self.dtb, self.dtb_null, False)
        m2l_utils.plot_points(self.output_path+'orientations.csv',
                              self.geol_clip, 'formation', 'X', 'Y', False, 'alpha')

        # Create arbitrary points for series without orientation data
        m2l_geometry.create_orientations(
            self.tmp_path, self.output_path, self.dtm, self.dtb, self.dtb_null, False, self.geol_clip, self.structure_clip, self.c_l)

    def export_contacts(self):
        contact_decimate = 5  # store every nth contact point (in object order)
        intrusion_mode = 0      # 1 all intrusions exluded from basal contacts, 0 only sills

        ls_dict, ls_dict_decimate = m2l_geometry.save_basal_contacts(
            self.tmp_path, self.dtm, self.dtb, self.dtb_null, False, self.geol_clip, contact_decimate, self.c_l, intrusion_mode)

        # Remove basal contacts defined by faults, no decimation
        m2l_geometry.save_basal_no_faults(
            self.tmp_path+'basal_contacts.shp', self.tmp_path+'faults_clip.shp', ls_dict, 10, self.c_l, self.proj_crs)

        # Remove faults from decimated basal contacts then save
        contacts = gpd.read_file(self.tmp_path+'basal_contacts.shp')
        m2l_geometry.save_basal_contacts_csv(
            contacts, self.output_path, self.dtm, self.dtb, self.dtb_null, False, contact_decimate, self.c_l)
        # False in this call was already false and isn't the cover flag
        m2l_utils.plot_points(self.output_path+'contacts4.csv',
                              self.geol_clip, 'formation', 'X', 'Y', False, 'alpha')

    # Interpolates a regular grid of orientations from an shapefile of
    # arbitrarily-located points and saves out four csv files of l,m & n
    # direction cosines and dip dip direction data
    def test_interpolation(self):

        geology_file = self.geol_clip_file
        structure_file = self.structure_clip_file
        basal_contacts = self.tmp_path+'basal_contacts.shp'
        spacing = 500  # grid spacing in meters
        misorientation = 30
        scheme = 'scipy_rbf'
        orientations = self.structures
        group_girdle = m2l_utils.plot_bedding_stereonets(
            orientations, self.geology, self.c_l)
        super_groups, use_gcode3 = Topology.super_groups_and_groups(
            group_girdle, self.tmp_path, misorientation)
        # print(super_groups)
        # print(self.geology['GROUP_'].unique())
        bbox = self.bbox

        orientation_interp, contact_interp, combo_interp = m2l_interpolation.interpolation_grids(
            geology_file, structure_file, basal_contacts, bbox, spacing, self.proj_crs, scheme, super_groups, self.c_l)

        with open(self.tmp_path+'interpolated_orientations.csv', 'w') as f:
            f.write('X,Y,l,m,n,dip,dip_dir\n')
            for row in orientation_interp:
                ostr = '{},{},{},{},{},{},{}\n'.format(
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6])
                f.write(ostr)
        with open(self.tmp_path+'interpolated_contacts.csv', 'w') as f:
            f.write('X,Y,l,m,angle\n')
            for row in contact_interp:
                ostr = '{},{},{},{},{}\n'.format(
                    row[0], row[1], row[2], row[3], row[4])
                f.write(ostr)
        with open(self.tmp_path+'interpolated_combined.csv', 'w') as f:
            f.write('X,Y,l,m,n,dip,dip_dir\n')
            for row in combo_interp:
                ostr = '{},{},{},{},{},{},{}\n'.format(
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6])
                f.write(ostr)

        if(spacing < 0):
            spacing = -(bbox[2]-bbox[0])/spacing
        x = int((bbox[2]-bbox[0])/spacing)+1
        y = int((bbox[3]-bbox[1])/spacing)+1
        print(x, y)
        dip_grid = np.ones((y, x))
        dip_grid = dip_grid*-999
        dip_dir_grid = np.ones((y, x))
        dip_dir_grid = dip_dir_grid*-999
        contact_grid = np.ones((y, x))
        contact_grid = dip_dir_grid*-999
        for row in combo_interp:
            r = int((row[1]-bbox[1])/spacing)
            c = int((row[0]-bbox[0])/spacing)
            dip_grid[r, c] = float(row[5])
            dip_dir_grid[r, c] = float(row[6])

        for row in contact_interp:
            r = int((row[1]-bbox[1])/spacing)
            c = int((row[0]-bbox[0])/spacing)
            contact_grid[r, c] = float(row[4])

        print('interpolated dips')
        plt.imshow(dip_grid, cmap="hsv", origin='lower', vmin=-90, vmax=90)
        plt.show()

        print('interpolated dip directions')
        plt.imshow(dip_dir_grid, cmap="hsv", origin='lower', vmin=0, vmax=360)
        plt.show()

        print('interpolated contacts')
        plt.imshow(contact_grid, cmap="hsv",
                   origin='lower', vmin=-360, vmax=360)
        plt.show()

    def export_faults(self):
        pass

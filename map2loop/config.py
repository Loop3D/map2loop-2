import os
import sys
import time
import shutil
import random

import numpy as np
import pandas as pd
import geopandas as gpd
from map2loop.topology import Topology
from map2loop import m2l_utils
from map2loop import m2l_geometry
from map2loop import m2l_interpolation
from map2loop import m2l_map_checker
from map2loop.m2l_utils import display, enable_quiet_mode, disable_quiet_mode, print
from map2loop.m2l_export import export_to_projectfile
import map2model

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import rasterio
import shapely


class Config(object):
    """Object that represents a sub-project. It is defined by some source data, 
    a region of interest (bounding box or polygon) and some execution flags.
    """

    def __init__(self,
                 project_path,
                 overwrite,
                 geology_file,
                 fault_file,
                 fold_file,
                 structure_file,
                 mindep_file,
                 bbox_3d,
                 polygon,
                 step_out,
                 dtm_crs,
                 proj_crs,
                 local,
                 quiet,
                 loopFilename,
                 c_l={},
                 **kwargs):

        self.project_path = project_path

        if overwrite is False:
            print(
                "WARNING: Overwrite should be a string value {true, in-place} ...")
            self.check_overwrite()
        if overwrite is True:
            print(
                "WARNING: Overwrite should be a string value {true, in-place} ... converting to true.")
            overwrite = 'true'

        if (not os.path.exists(project_path)):
            # Create proj root dir if doesn't exist
            os.mkdir(project_path)
        elif overwrite == "in-place":
            # Pass if proj root exists and complete overwrite not wanted
            pass
        else:
            # Remove if exists and accept user's direction
            if overwrite == "true":
                shutil.rmtree(project_path)
                while os.path.exists(project_path):
                    pass
                os.mkdir(project_path)
            else:
                self.check_overwrite()

        self.graph_path = os.path.join(self.project_path, 'graph')
        self.tmp_path = os.path.join(self.project_path, 'tmp')
        self.data_path = os.path.join(self.project_path, 'data')
        self.dtm_path = os.path.join(self.project_path, 'dtm')
        self.output_path = os.path.join(self.project_path, 'output')
        self.vtk_path = os.path.join(self.project_path, 'vtk')

        self.fault_file_csv = os.path.join(self.tmp_path, "faults.csv")
        self.fault_output_file_csv = os.path.join(self.output_path,
                                                  "faults.csv")
        self.structure_file_csv = os.path.join(self.tmp_path, "structure.csv")
        self.geology_file_csv = os.path.join(self.tmp_path, "geology.csv")
        self.mindep_file_csv = os.path.join(self.tmp_path, "mindep.csv")

        self.strat_graph_file = os.path.join(self.graph_path,
                                             "graph_strat_NONE.gml")
        self.dtm_file = os.path.join(self.dtm_path, 'dtm.tif')
        self.dtm_reproj_file = os.path.join(self.dtm_path, 'dtm_rp.tif')

        if (not os.path.isdir(self.tmp_path)):
            os.mkdir(self.tmp_path)
        if (not os.path.isdir(self.data_path)):
            os.mkdir(self.data_path)
        if (not os.path.isdir(self.output_path)):
            os.mkdir(self.output_path)
        if (not os.path.isdir(self.dtm_path)):
            os.mkdir(self.dtm_path)
        if (not os.path.isdir(self.vtk_path)):
            os.mkdir(self.vtk_path)
        if (not os.path.isdir(self.graph_path)):
            os.mkdir(self.graph_path)

        self.quiet = quiet
        if self.quiet == 'all':
            enable_quiet_mode()

        self.clut_path = kwargs['clut_path']
        self.run_flags = kwargs['run_flags']

        self.bbox_3d = bbox_3d
        self.bbox = tuple([
            bbox_3d["minx"], bbox_3d["miny"], bbox_3d["maxx"], bbox_3d["maxy"]
        ])
        self.polygon = polygon
        self.step_out = step_out

        self.quiet = quiet
        self.c_l = c_l

        self.dtm_crs = dtm_crs
        self.proj_crs = proj_crs

        self.loop_projectfile = loopFilename

        # Check input maps for missing values
        drift_prefix = kwargs.get('drift_prefix', ['None'])
        self.local = local
        #       - Check if fold file is always the same as fault or needs to be seperated
        # TODO: Allow for input as a polygon, not just a bounding box.
        structure_file, geology_file, fault_file, mindep_file, fold_file, c_l = m2l_map_checker.check_map(
            structure_file, geology_file, fault_file, mindep_file, fold_file,
            self.tmp_path, self.bbox, c_l, proj_crs, self.local, drift_prefix)

        # Process and store workflow params
        self.geology_file = geology_file
        self.structure_file = structure_file
        self.fault_file = fault_file
        self.fold_file = fold_file
        self.mindep_file = mindep_file

        disable_quiet_mode()

    def check_overwrite(self):
        allow = input(
            "Directory \"{}\" exists, overwrite? (y/[n])".format(
                self.project_path))
        if allow == "y":
            shutil.rmtree(self.project_path)
            while os.path.exists(self.project_path):
                pass
            os.mkdir(self.project_path)
        else:
            sys.exit(
                'Either set overwrite to true or specify a different output_path.')

    def preprocess(self):
        """[summary]

        :param command: [description], defaults to ""
        :type command: str, optional
        """

        if self.quiet == 'all':
            enable_quiet_mode()

        geology = gpd.read_file(self.geology_file, bbox=self.bbox)
        geology[self.c_l['g']].fillna(geology[self.c_l['g2']], inplace=True)
        geology[self.c_l['g']].fillna(geology[self.c_l['c']], inplace=True)
        faults = gpd.read_file(self.fault_file, bbox=self.bbox)
        folds = gpd.read_file(self.fold_file, bbox=self.bbox)
        structures = gpd.read_file(self.structure_file, bbox=self.bbox)
        mindeps = None
        try:
            mindeps = gpd.read_file(self.mindep_file, bbox=self.bbox)
            mindeps.crs = self.proj_crs
        except Exception as e:
            print("Warning: Valid mineral deposit file missing")

        # Fix crs to project default and overwrite source
        geology.crs = self.proj_crs
        faults.crs = self.proj_crs
        folds.crs = self.proj_crs
        structures.crs = self.proj_crs
        self.mindeps = mindeps

        self.geology = geology
        self.faults = faults
        self.structures = structures

        # Faults
        self.faults_clip = faults.copy()
        self.faults_clip.crs = self.proj_crs
        self.faults_clip_file = os.path.join(self.tmp_path, "faults_clip.shp")
        self.faults_clip.to_file(self.faults_clip_file)

        # Geology
        self.geol_clip = m2l_utils.explode(self.geology)
        self.geol_clip.crs = self.proj_crs
        self.geol_clip_file = os.path.join(self.tmp_path, "geol_clip.shp")
        self.geol_clip.to_file(self.geol_clip_file)

        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_rows', None)

        # Check if bedding data uses the strike convention instead of dip direction
        if (self.c_l['otype'] == 'strike'):
            structures['azimuth2'] = structures.apply(
                lambda row: row[self.c_l['dd']] + 90.0, axis=1)
            self.c_l['dd'] = 'azimuth2'
            self.c_l['otype'] = 'dip direction'
            structures.to_file(self.structure_file)

        # Structures
        list1 = [
            'geometry', self.c_l['d'], self.c_l['dd'], self.c_l['sf'],
            self.c_l['bo']
        ]
        list2 = list(set(list1))
        sub_pts = self.structures[list2]
        structure_code = gpd.sjoin(sub_pts,
                                   self.geol_clip,
                                   how="left",
                                   op="within")

        minx, miny, maxx, maxy = self.bbox
        y_point_list = [miny, miny, maxy, maxy, miny]
        x_point_list = [minx, maxx, maxx, minx, minx]

        bbox_geom = shapely.geometry.Polygon(zip(x_point_list, y_point_list))

        polygo = gpd.GeoDataFrame(index=[0],
                                  crs=self.proj_crs,
                                  geometry=[bbox_geom])
        is_bed = structure_code[self.c_l['sf']].str.contains(
            self.c_l['bedding'], regex=False)

        structure_clip = structure_code[is_bed]
        structure_clip.crs = self.proj_crs

        if (self.c_l['otype'] == 'strike'):
            structure_clip['azimuth2'] = structure_clip.apply(
                lambda row: row[self.c_l['dd']] + 90.0, axis=1)
            self.c_l['dd'] = 'azimuth2'
            self.c_l['otype'] = 'dip direction'

        self.structure_clip = structure_clip[~structure_clip[self.c_l['o']].
                                             isnull()]
        self.structure_clip_file = os.path.join(self.tmp_path,
                                                'structure_clip.shp')
        self.structure_clip.to_file(self.structure_clip_file)

        self.create_cmap()

        try:
            fig, ax = plt.subplots()
            plt.tight_layout()
            ax.ticklabel_format(axis='both', useOffset=False, style='plain')
            ax.margins(0.0)
            fig.set_facecolor("#ffffff00")

            self.geology_figure = geology.copy().plot(
                column=self.c_l['c'],
                ax=ax,
                figsize=(10, 10),
                edgecolor='#000000',
                linewidth=0.2,
                cmap=self.cmap).get_figure()

            # self.export_png()
            fig, ax = plt.subplots()

            base = geology.plot(column=self.c_l['c'],
                                figsize=(10, 10),
                                ax=ax,
                                edgecolor='#000000',
                                linewidth=0.2,
                                legend=True,
                                cmap=self.cmap)
            leg = base.get_legend()
            leg.set_bbox_to_anchor((1.04, 1))

            structures.plot(ax=base, color='none', edgecolor='black')

            faults.plot(ax=base,
                        cmap='rainbow',
                        column=self.c_l['f'],
                        figsize=(10, 10),
                        linewidth=0.4)
            structures[[
                'geometry', self.c_l['gi'], self.c_l['d'], self.c_l['dd']
            ]].plot(ax=base)

            fig = self.polygon.plot(ax=base, color='none',
                                    edgecolor='black').get_figure()
            fig.savefig(os.path.join(self.tmp_path, "input-data.png"))

            if self.quiet == 'None':
                plt.show()

            return
        except Exception as e:
            print(e)

        disable_quiet_mode()

    def create_cmap(self):
        # Make colours consistent from map to model
        formations = sorted([
            formation.replace(" ", "_").replace('-', '_') for formation in
            list(set(self.geol_clip[self.c_l['c']].to_numpy()))
        ])
        temp_colours = [""] * len(formations)
        self.colour_dict = dict(zip(formations, temp_colours))
        try:
            # Try to retrieve the clut reference
            colour_ref = pd.read_csv(self.clut_path)
            for formation in formations:
                key = formation
                colour = None
                try:
                    colour = colour_ref[colour_ref['UNITNAME'] ==
                                        key]['colour'].to_numpy()[0]
                except Exception as e:
                    colour = ('#%02X%02X%02X' %
                              (random.randint(0, 255), random.randint(
                                  0, 255), random.randint(0, 255)))

                self.colour_dict[key] = colour
                print(key, colour)

        except Exception as e:
            # Otherwise, just append a random set
            self.clut_path = ""
            random_colours = [
                '#%02X%02X%02X' % (random.randint(
                    0, 255), random.randint(0, 255), random.randint(0, 255))
                for i in range(len(formations))
            ]
            i = 0
            for key in self.colour_dict.keys():
                self.colour_dict[key] = random_colours[i]

        self.cmap = colors.ListedColormap(self.colour_dict.values(),
                                          name='geol_key')

    def export_csv(self):
        # TODO: - Move away from tab seperators entirely (topology and map2model)

        # Save geology polygons
        hint_flag = False  # use GSWA strat database to provide topology hints
        sub_geol = self.geology[[
            'geometry', self.c_l['o'], self.c_l['c'], self.c_l['g'],
            self.c_l['u'], self.c_l['min'], self.c_l['max'], self.c_l['ds'],
            self.c_l['r1'], self.c_l['r2']
        ]]
        Topology.save_geol_wkt(sub_geol, self.geology_file_csv, self.c_l,
                               hint_flag)

        # Save mineral deposits
        if self.mindeps is not None:
            sub_mindep = self.mindeps[[
                'geometry', self.c_l['msc'], self.c_l['msn'], self.c_l['mst'],
                self.c_l['mtc'], self.c_l['mscm'], self.c_l['mcom']
            ]]
            Topology.save_mindep_wkt(sub_mindep, self.mindep_file_csv,
                                     self.c_l)

        # Save orientation data
        sub_pts = self.structures[[
            'geometry', self.c_l['gi'], self.c_l['d'], self.c_l['dd']
        ]]
        Topology.save_structure_wkt(sub_pts, self.structure_file_csv, self.c_l)

        # Save faults
        sub_lines = self.faults[['geometry', self.c_l['o'], self.c_l['f']]]
        Topology.save_faults_wkt(sub_lines, self.fault_file_csv, self.c_l)

    def update_parfile(self):
        Topology.save_parfile(self, self.c_l, self.output_path,
                              self.geology_file_csv, self.fault_file_csv,
                              self.structure_file_csv, self.mindep_file_csv,
                              self.bbox[0], self.bbox[1], self.bbox[2],
                              self.bbox[3], 500.0, 'Fe,Cu,Au,NONE')

    def run_map2model(self, deposits, aus):
        quiet_m2m = False
        if self.quiet == 'all':
            quiet_m2m = True
        if self.mindeps is not None:
            run_log = map2model.run(self.graph_path, self.geology_file_csv,
                                    self.fault_file_csv, self.mindep_file_csv,
                                    self.bbox_3d, self.c_l, quiet_m2m,
                                    deposits)
        else:
            run_log = map2model.run(self.graph_path, self.geology_file_csv,
                                    self.fault_file_csv, "", self.bbox_3d,
                                    self.c_l, quiet_m2m, deposits)

        print(run_log)

        print("Resolving ambiguities using ASUD...", end='\toutput_dir:')
        if aus:
            Topology.use_asud(self.strat_graph_file, self.graph_path)
            self.strat_graph_file = os.path.join(self.graph_path,
                                                 'ASUD_strat.gml')
        print("Done.")

        print("Generating topology graph display and unit groups...")
        self.G = nx.read_gml(self.strat_graph_file, label='id')
        selected_nodes = [n for n, v in self.G.nodes(data=True) if n >= 0]

        if self.quiet == 'None':
            nx.draw_networkx(self.G,
                             pos=nx.kamada_kawai_layout(self.G),
                             arrows=True,
                             nodelist=selected_nodes)

        nlist = list(self.G.nodes.data('LabelGraphics'))
        nlist.sort()
        for node in nlist:
            if node[0] >= 0:
                elem = str(node[1]).replace("{'text':", "").replace(
                    ", 'fontSize': 14}", "")
                # second = elem.split(":").replace("'", "")
                print(node[0], " ", elem)

        # plt.savefig(os.path.join(self.tmp_path,"topology-fig.png"))
        print("Topology figure saved to",
              os.path.join(self.tmp_path, "topology-fig.png"))

        # Save groups of stratigraphic units
        groups, self.glabels, G = Topology.get_series(self.strat_graph_file,
                                                      'id')

        quiet_topology = True
        if self.quiet == 'None':
            quiet_topology = False
        Topology.save_units(
            G,
            self.tmp_path,
            self.glabels,
            Australia=True,
            asud_strat_file="https://gist.githubusercontent.com/yohanderose/3b257dc768fafe5aaf70e64ae55e4c42/raw/8598c7563c1eea5c0cd1080f2c418dc975cc5433/ASUD.csv",
            quiet=quiet_topology)

        print("Done")

    def load_dtm(self, source="AU"):
        # group all Australian states codes under the global country code (ISO 3166 ALPHA-2)
        polygon_ll = self.polygon.to_crs(self.dtm_crs)
        minlong = polygon_ll.total_bounds[0] - self.step_out
        maxlong = polygon_ll.total_bounds[2] + self.step_out
        minlat = polygon_ll.total_bounds[1] - self.step_out
        maxlat = polygon_ll.total_bounds[3] + self.step_out
        print("Fetching DTM... ", end=" bbox:")
        print(minlong, maxlong, minlat, maxlat)
        #source='QLD'
        if source in ("WA", "NSW", "VIC", "SA", "QLD", "ACT", "TAS"):
            source = 'AU'
            i, done = 0, False
            while not done:
                if i >= 10:
                    raise NameError(
                        f'map2loop error: Could not access DTM server after {i} attempts'
                    )
                try:
                    print(f'Attempt: {i} ...', end='')
                    if source.upper() in ("AU", "AUSTRALIA"):
                        m2l_utils.get_dtm(self.dtm_file, minlong, maxlong, minlat,
                                          maxlat)
                    elif source.upper() in ("T.H", "HAWAII"):  # beware, TH is ISO 3166 code for Thailand
                        m2l_utils.get_dtm_hawaii(self.dtm_file, minlong, maxlong,
                                                 minlat, maxlat)
                    else:  # try from opentopography
                        m2l_utils.get_dtm_topography_org(
                            self.dtm_file, minlong, maxlong, minlat, maxlat)
                    print("Succeeded !")
                    done = True
                except:
                    time.sleep(1)
                    i += 1
                    print(f' Failed !')
        elif source.startswith('http'):
            i, done = 0, False
            while not done:
                if i >= 10:
                    raise NameError(
                        f'map2loop error: Could not access DTM server after {i} attempts'
                    )
                try:
                    print(f'Attempt: {i} ...', end='')
                    if 'au' in source:
                        m2l_utils.get_dtm(self.dtm_file, minlong, maxlong, minlat,
                                          maxlat, url=source)
                    elif 'hawaii' in source:  # beware, TH is ISO 3166 code for Thailand
                        m2l_utils.get_dtm_hawaii(self.dtm_file, minlong, maxlong,
                                                 minlat, maxlat, url=source)
                    else:  # try from opentopography
                        m2l_utils.get_dtm_topography_org(
                            self.dtm_file, minlong, maxlong, minlat, maxlat)
                    print("Succeeded !")
                    done = True
                except:
                    time.sleep(1)
                    i += 1
                    print(f' Failed !')
        else:
            bbox = [
                self.bbox_3d["minx"], self.bbox_3d["miny"],
                self.bbox_3d["maxx"], self.bbox_3d["maxy"]
            ]
            m2l_utils.get_local_dtm(self.dtm_file, source, self.dtm_crs, bbox)

        m2l_utils.reproject_dtm(self.dtm_file,
                                self.dtm_reproj_file,
                                self.dtm_crs, self.proj_crs)

        self.dtm = rasterio.open(self.dtm_reproj_file)

        if self.quiet == 'None':
            plt.imshow(self.dtm.read(1), cmap='terrain', vmin=0, vmax=1000)

            plt.title('DTM')
            plt.show()

    def join_features(self):
        # Save geology clips
        quiet_topology = True
        if self.quiet == "None":
            quiet_topology = False
        Topology.save_group(Topology, self.G, self.tmp_path, self.glabels,
                            self.geol_clip, self.c_l, quiet_topology)

    def calc_depth_grid(self, dtb):
        dtm = self.dtm

        if dtb == "":
            self.dtb = 0
            self.dtb_null = 0

            print("dtb and dtb_null set to 0")
            return

        # TODO: DTB need to be defined, every function call bellow here that has a False boolean is referencing to the workflow['cover_map'] flag
        # dtb_grid = os.path.join(data_path,'young_cover_grid.tif') #obviously hard-wired for the moment
        # dtb_null = '-2147483648' #obviously hard-wired for the moment
        # cover_map_path = os.path.join(data_path,'Young_Cover_FDS_MGA_clean.shp') #obviously hard-wired for the moment
        # dtb_clip = os.path.join(output_path,'young_cover_grid_clip.tif') #obviously hard-wired for the moment
        # cover_dip = 10 # dip of cover away from contact
        # cover_spacing = 5000 # of contact grid in metres

        dtb_raw = rasterio.open(dtb_grid)

        cover = gpd.read_file(cover_map_path)

        with fiona.open(cover_map_path, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]

        with rasterio.open(dtb_grid) as src:
            out_image, out_transform = rasterio.mask.mask(src,
                                                          shapes,
                                                          crop=True)
            out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        with rasterio.open(dtb_clip, "w", **out_meta) as dest:
            dest.write(out_image)

        dtb = rasterio.open(dtb_clip)

        m2l_geometry.process_cover(output_path,
                                   dtm,
                                   dtb,
                                   dtb_null,
                                   cover,
                                   workflow['cover_map'],
                                   cover_dip,
                                   bbox,
                                   proj_crs,
                                   cover_spacing,
                                   contact_decimate=3,
                                   use_vector=True,
                                   use_grid=True)

    def export_orientations(self, orientation_decimate):
        m2l_geometry.save_orientations(self.structure_clip, self.output_path,
                                       self.c_l, orientation_decimate,
                                       self.dtm, self.dtb, self.dtb_null,
                                       False)

        if self.quiet == 'None':
            m2l_utils.plot_points(
                os.path.join(self.output_path, 'orientations.csv'),
                self.geol_clip, 'formation', 'X', 'Y', False, 'alpha')

        # Create arbitrary points for series without orientation data
        m2l_geometry.create_orientations(self.tmp_path, self.output_path,
                                         self.dtm, self.dtb, self.dtb_null,
                                         False, self.geol_clip,
                                         self.structure_clip, self.c_l)

    def export_contacts(self, contact_decimate, intrusion_mode):

        ls_dict, ls_dict_decimate = m2l_geometry.save_basal_contacts(
            self.tmp_path, self.dtm, self.dtb, self.dtb_null, False,
            self.geol_clip, contact_decimate, self.c_l, intrusion_mode)

        # Remove basal contacts defined by faults, no decimation
        m2l_geometry.save_basal_no_faults(
            os.path.join(self.tmp_path, 'basal_contacts.shp'),
            os.path.join(self.tmp_path, 'faults_clip.shp'), ls_dict, 10,
            self.c_l, self.proj_crs)

        # Remove faults from decimated basal contacts then save
        contacts = gpd.read_file(
            os.path.join(self.tmp_path, 'basal_contacts.shp'))
        m2l_geometry.save_basal_contacts_csv(contacts, self.output_path,
                                             self.dtm, self.dtb, self.dtb_null,
                                             False, contact_decimate, self.c_l)
        # False in this call was already false and isn't the cover flag
        if self.quiet == "None":
            m2l_utils.plot_points(
                os.path.join(self.output_path, 'contacts4.csv'),
                self.geol_clip, 'formation', 'X', 'Y', False, 'alpha')

    # Interpolates a regular grid of orientations from an shapefile of
    # arbitrarily-located points and saves out four csv files of l, m & n
    # direction cosines and dip dip direction data
    def test_interpolation(self, interpolation_spacing, misorientation,
                           interpolation_scheme):

        geology_file = self.geol_clip_file
        structure_file = self.structure_clip_file
        basal_contacts = os.path.join(self.tmp_path, 'basal_contacts.shp')
        self.spacing = interpolation_spacing  # grid spacing in meters
        # misorientation = misorientation
        self.scheme = interpolation_scheme
        orientations = self.structures

        quiet_interp = True
        if self.quiet == "None":
            quiet_interp = False

        group_girdle = m2l_utils.plot_bedding_stereonets(
            orientations, self.geology, self.c_l, quiet_interp)
        super_groups, self.use_gcode3 = Topology.super_groups_and_groups(
            group_girdle, self.tmp_path, misorientation)
        # print(super_groups)
        # print(self.geology['GROUP_'].unique())
        bbox = self.bbox

        orientation_interp, contact_interp, combo_interp = m2l_interpolation.interpolation_grids(
            geology_file, structure_file, basal_contacts, bbox, self.spacing,
            self.proj_crs, self.scheme, super_groups, self.c_l)

        with open(os.path.join(self.tmp_path, 'interpolated_orientations.csv'),
                  'w') as f:
            f.write('X, Y, l, m, n, dip, dip_dir\n')
            for row in orientation_interp:
                ostr = '{}, {}, {}, {}, {}, {}, {}\n'.format(
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6])
                f.write(ostr)
        with open(os.path.join(self.tmp_path, 'interpolated_contacts.csv'),
                  'w') as f:
            f.write('X, Y, l, m, angle\n')
            for row in contact_interp:
                ostr = '{}, {}, {}, {}, {}\n'.format(row[0], row[1], row[2],
                                                     row[3], row[4])
                f.write(ostr)
        with open(os.path.join(self.tmp_path, 'interpolated_combined.csv'),
                  'w') as f:
            f.write('X, Y, l, m, n, dip, dip_dir\n')
            for row in combo_interp:
                ostr = '{}, {}, {}, {}, {}, {}, {}\n'.format(
                    row[0], row[1], row[2], row[3], row[4], row[5], row[6])
                f.write(ostr)

        if (self.spacing < 0):
            self.spacing = -(bbox[2] - bbox[0]) / spacing
        self.x = int((bbox[2] - bbox[0]) / self.spacing) + 1
        self.y = int((bbox[3] - bbox[1]) / self.spacing) + 1
        x = self.x
        y = self.y
        print(x, y)
        dip_grid = np.ones((y, x))
        dip_grid = dip_grid * -999
        dip_dir_grid = np.ones((y, x))
        dip_dir_grid = dip_dir_grid * -999
        contact_grid = np.ones((y, x))
        contact_grid = dip_dir_grid * -999
        for row in combo_interp:
            r = int((row[1] - bbox[1]) / self.spacing)
            c = int((row[0] - bbox[0]) / self.spacing)
            dip_grid[r, c] = float(row[5])
            dip_dir_grid[r, c] = float(row[6])

        for row in contact_interp:
            r = int((row[1] - bbox[1]) / self.spacing)
            c = int((row[0] - bbox[0]) / self.spacing)
            contact_grid[r, c] = float(row[4])

        self.dip_grid = dip_grid
        self.dip_dir_grid = dip_dir_grid

        if self.quiet == 'None':
            print('interpolated dips')
            plt.imshow(self.dip_grid,
                       cmap="hsv",
                       origin='lower',
                       vmin=-90,
                       vmax=90)
            plt.show()

            print('interpolated dip directions')
            plt.imshow(self.dip_dir_grid,
                       cmap="hsv",
                       origin='lower',
                       vmin=0,
                       vmax=360)
            plt.show()

            print('interpolated contacts')
            plt.imshow(contact_grid,
                       cmap="hsv",
                       origin='lower',
                       vmin=-360,
                       vmax=360)
            plt.show()

    def save_cmap(self):
        """Create a colourmap for the model using the colour code
        """
        all_sorts = pd.read_csv(
            os.path.join(self.tmp_path, 'all_sorts_clean.csv'))

        colours = []
        for code in all_sorts['code']:
            colours.append([self.colour_dict[code]])

        data = colours
        expected_extra_cols = pd.DataFrame(columns=['colour'], data=data)
        all_sorts = pd.concat([all_sorts, expected_extra_cols], axis=1)
        all_sorts.to_csv(os.path.join(self.tmp_path, 'all_sorts_clean.csv'),
                         ",",
                         index=None)

    def export_faults(self, fault_decimate, min_fault_length, fault_dip):
        # fault_decimate = 5
        # min_fault_length = 5000
        # fault_dip = 90

        m2l_geometry.save_faults(
            os.path.join(self.tmp_path, 'faults_clip.shp'), self.output_path,
            self.dtm, self.dtb, self.dtb_null, False, self.c_l, fault_decimate,
            min_fault_length, fault_dip)

        faults = pd.read_csv(self.fault_output_file_csv, sep=",")
        faults_len = len(faults)
        if (faults_len > 0):
            m2l_interpolation.process_fault_throw_and_near_faults_from_grid(
                self.tmp_path, self.output_path, self.dtm_reproj_file,
                self.dtb, self.dtb_null, False, self.c_l, self.proj_crs,
                self.bbox, self.scheme, self.dip_grid, self.dip_dir_grid,
                self.x, self.y, self.spacing)

    def process_plutons(self, pluton_dip, pluton_form, dist_buffer,
                        contact_decimate):
        # pluton_dip = 45
        pluton_dip = str(pluton_dip)
        self.pluton_form = pluton_form  # 'domes'

        # dist_buffer = 10
        # contact_decimate = 5  # store every nth contact point (in object order)

        m2l_geometry.process_plutons(self.tmp_path, self.output_path,
                                     self.geol_clip, self.local, self.dtm,
                                     self.dtb, self.dtb_null, False,
                                     self.pluton_form, pluton_dip,
                                     contact_decimate, self.c_l)

    def extract_section_features(self, seismic_line_file, seismic_bbox_file,
                                 seismic_interp_file):
        # Extract faults and basal contacts of groups from seismic section
        # input geology file (if local)

        seismic_line_file = seismic_line_file
        seismic_line = gpd.read_file(seismic_line_file)  # import map
        seismic_line.plot(figsize=(10, 10), edgecolor='#000000',
                          linewidth=0.2)  # display map
        display(seismic_line)

        # input geology file (if local)
        seismic_bbox_file = seismic_bbox_file
        seismic_bbox = gpd.read_file(seismic_bbox_file)  # import map
        seismic_bbox.set_index('POSITION', inplace=True)

        # input geology file (if local)
        seismic_interp_file = seismic_interp_file
        seismic_interp = gpd.read_file(seismic_interp_file)  # import map
        seismic_interp.plot(column='FEATURE',
                            figsize=(10, 10),
                            edgecolor='#000000',
                            linewidth=0.5)  # display map
        display(seismic_interp)

        surface_cut = 2000

        m2l_geometry.extract_section(self.tmp_path, self.output_path,
                                     seismic_line, seismic_bbox,
                                     seismic_interp, self.dtm, self.dtb,
                                     self.dtb_null, False, surface_cut)

        contacts = pd.read_csv(os.path.join(self.output_path, 'contacts4.csv'),
                               ", ")
        seismic_contacts = pd.read_csv(
            os.path.join(self.output_path, 'seismic_base.csv'), ", ")
        all_contacts = pd.concat([contacts, seismic_contacts], sort=False)
        all_contacts.to_csv(os.path.join(self.output_path, 'contacts4.csv'),
                            index=None,
                            header=True)

        faults = pd.read_csv(os.path.join(self.output_path, 'faults.csv'),
                             ", ")
        seismic_faults = pd.read_csv(
            os.path.join(self.output_path, 'seismic_faults.csv'), ", ")
        all_faults = pd.concat([faults, seismic_faults], sort=False)
        all_faults.to_csv(os.path.join(self.output_path, 'faults.csv'),
                          index=None,
                          header=True)

    def propagate_contact_dips(self, contact_dip,
                               contact_orientation_decimate):
        print("Propagating dips along contacts...")
        orientations = pd.read_csv(
            os.path.join(self.output_path, 'orientations.csv'), ", ")
        # This is supposed to be a csv but my csv doesn't have a geometry part
        contacts = gpd.read_file(
            os.path.join(self.tmp_path, 'basal_contacts.shp'))
        # contact_dip = -999
        # contact_orientation_decimate = 5
        m2l_geometry.save_basal_contacts_orientations_csv(
            contacts, orientations, self.geol_clip, self.tmp_path,
            self.output_path, self.dtm, self.dtb, self.dtb_null, False,
            contact_orientation_decimate, self.c_l, contact_dip, self.dip_grid,
            self.spacing, self.bbox)

    def calc_thickness(self, contact_decimate, null_scheme, thickness_buffer,
                       max_thickness_allowed, cl):
        # Estimate formation thickness and normalised formation thickness
        geology_file = os.path.join(self.tmp_path, 'basal_contacts.shp')
        # contact_decimate = 5
        # null_scheme = 'null'
        m2l_interpolation.save_contact_vectors(geology_file, self.tmp_path,
                                               self.dtm, self.dtb,
                                               self.dtb_null, False, self.bbox,
                                               self.c_l, null_scheme,
                                               contact_decimate)

        # buffer = 5000
        # max_thickness_allowed = 10000

        # TODO: multi thread / numba jit
        m2l_geometry.calc_thickness_with_grid(self.tmp_path, self.output_path,
                                              thickness_buffer,
                                              max_thickness_allowed, self.c_l,
                                              self.bbox, self.dip_grid,
                                              self.dip_dir_grid, self.x,
                                              self.y, self.spacing, self.dtm)

        m2l_geometry.calc_min_thickness_with_grid(
            self.tmp_path, self.output_path, thickness_buffer,
            max_thickness_allowed, self.c_l, self.bbox, self.dip_grid,
            self.dip_dir_grid, self.x, self.y, self.spacing, self.dtm)

        m2l_geometry.normalise_thickness(self.output_path)

        if self.quiet == "None":
            m2l_utils.plot_points(
                os.path.join(self.output_path,
                             'formation_thicknesses_norm.csv'), self.geol_clip,
                'norm_th', 'x', 'y', True, 'numeric')

    def create_fold_axial_trace_points(self, fold_decimate, fat_step,
                                       close_dip):
        # fold_decimate = 5
        folds_clip = gpd.read_file(self.fold_file)
        if (len(folds_clip) > 0):

            m2l_geometry.save_fold_axial_traces(self.fold_file,
                                                self.output_path, self.dtm,
                                                self.dtb, self.dtb_null, False,
                                                self.c_l, fold_decimate)

            # TODO : better approximation / multithread / numba
            m2l_geometry.save_fold_axial_traces_orientations(
                self.fold_file, self.output_path, self.tmp_path, self.dtm,
                self.dtb, self.dtb_null, False, self.c_l, self.proj_crs,
                fold_decimate, fat_step, close_dip, self.scheme, self.bbox,
                self.spacing, self.dip_grid, self.dip_dir_grid)

    def postprocess(self, inputs, workflow, use_interpolations, use_fat):
        # use_interpolations = True
        # use_fat = True

        m2l_geometry.tidy_data(self.output_path, self.tmp_path, self.clut_path,
                               self.use_gcode3, use_interpolations, use_fat,
                               self.pluton_form, inputs, workflow, self.c_l)
        model_top = round(np.amax(self.dtm.read(1)), -2)

        # self.dtm.close()
        # if(workflow['cover_map']):
        #     dtb.close()

        # Calculate polarity of original bedding orientation data
        if (workflow['polarity']):
            m2l_geometry.save_orientations_with_polarity(
                os.path.join(self.output_path, 'orientations.csv'),
                self.output_path,
                self.c_l,
                os.path.join(self.tmp_path, 'basal_contacts.shp'),
                os.path.join(self.tmp_path, 'all_sorts.csv'),
            )

            if self.quiet == "None":
                m2l_utils.plot_points(
                    os.path.join(self.output_path,
                                 'orientations_polarity.csv'), self.geol_clip,
                    'polarity', 'X', 'Y', True, 'alpha')

        # Calculate minimum fault offset from stratigraphy and stratigraphic fault offset
        if (workflow['strat_offset']):
            fault_test = pd.read_csv(
                os.path.join(self.output_path, 'fault_dimensions.csv'), ', ')
            if (len(fault_test) > 0):

                m2l_geometry.fault_strat_offset(
                    self.output_path, self.c_l, self.proj_crs,
                    os.path.join(self.output_path,
                                 'formation_summary_thicknesses.csv'),
                    os.path.join(self.tmp_path, 'all_sorts.csv'),
                    os.path.join(self.tmp_path, 'faults_clip.shp'),
                    os.path.join(self.tmp_path, 'geol_clip.shp'),
                    os.path.join(self.output_path, 'fault_dimensions.csv'))

                if self.quiet == "None":
                    m2l_utils.plot_points(
                        os.path.join(self.output_path,
                                     'fault_strat_offset3.csv'),
                        self.geol_clip, 'min_offset', 'X', 'Y', True,
                        'numeric')
                    m2l_utils.plot_points(
                        os.path.join(self.output_path,
                                     'fault_strat_offset3.csv'),
                        self.geol_clip, 'strat_offset', 'X', 'Y', True,
                        'numeric')

        # Analyse fault topologies
        fault_parse_figs = True
        if self.quiet == "None":
            fault_parse = False
        Topology.parse_fault_relationships(self.graph_path, self.tmp_path,
                                           self.output_path, fault_parse_figs)

        # TODO: Figures sometimes look a bit squashed in notebooks

    def update_projectfile(self):
        self.loop_projectfile = export_to_projectfile(self.loop_projectfile,
                                                      self.tmp_path,
                                                      self.output_path,
                                                      self.bbox_3d,
                                                      self.proj_crs)

        print("PROJECTFILE FOUND AT", self.loop_projectfile)

    def export_png(self):
        filename = self.loop_projectfile
        if self.loop_projectfile is None:
            # TODO: Make sure these user provided paths end with a slash or are joined properly
            filename = os.path.join(
                self.project_path, '{}'.format(self.project_path))
        print("Exporting graphical map...")
        try:

            self.geology_figure.savefig("{}.png".format(filename))
            print("Geology graphic exported to: ", filename)
        except Exception as e:
            print(e)
            print("WARNING: Could not save geology graphic")

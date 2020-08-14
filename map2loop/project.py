import hjson
import sys

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon

from map2loop.config import Config


class Project(object):
    def __init__(self,
                 # region of interest coordinates in metre-based system (or non-degree system)
                 # (minx, miny, maxx, maxy, bottom, top)
                 # TODO: Remove harcoding if local files ship with examples
                 geology_file,
                 fault_file,
                 fold_file,
                 structure_file,
                 mindep_file,
                 metadata=None,
                 workflow={'model_engine': 'loopstructural'},
                 ):

        # TODO: Create ways to get and set local files
        self.geology_file = geology_file
        self.fault_file = fault_file
        self.fold_file = fold_file
        self.structure_file = structure_file
        self.mindep_file = mindep_file

        # Load in dictionary that describes the column names
        if metadata is None:
            print(
                "Please pass the path to a valid metadata file (.json or .hjson) to update_config(...) that identifies your input table column names.")
            print("You can find an example config here https://gist.github.com/yohanderose/a127c29cb88529f049a5bafc881bb1a0")
            sys.exit(1)
        else:
            try:
                with open(metadata) as raw_data:
                    self.c_l = hjson.load(raw_data)
            except Exception as e:
                print(e)
                sys.exit(1)

        self.set_proj_defaults()
        self.update_workflow(workflow)

    def set_proj_defaults(self):
        geology = gpd.read_file(self.geology_file)
        self.proj_bounds = geology.total_bounds
        self.proj_crs = geology.crs
        self.step_out = 0.1

    def update_workflow(self, workflow):
        if(workflow['model_engine'] == 'geomodeller'):
            workflow.update({'seismic_section': False,
                             'cover_map': False,
                             'near_fault_interpolations': True,
                             'fold_axial_traces': False,
                             'stereonets': False,
                             'formation_thickness': True,
                             'polarity': False,
                             'strat_offset': False,
                             'contact_dips': True})
        elif(workflow['model_engine'] == 'loopstructural'):
            workflow.update({'seismic_section': False,
                             'cover_map': False,
                             'near_fault_interpolations': False,
                             'fold_axial_traces': True,
                             'stereonets': False,
                             'formation_thickness': True,
                             'polarity': False,
                             'strat_offset': True,
                             'contact_dips': True})
        elif(workflow['model_engine'] == 'gempy'):
            workflow.update({'seismic_section': False,
                             'cover_map': False,
                             'near_fault_interpolations': False,
                             'fold_axial_traces': True,
                             'stereonets': False,
                             'formation_thickness': False,
                             'polarity': False,
                             'strat_offset': False,
                             'contact_dips': False})
        elif(workflow['model_engine'] == 'noddy'):
            workflow.update({'seismic_section': False,
                             'cover_map': False,
                             'near_fault_interpolations': False,
                             'fold_axial_traces': False,
                             'stereonets': False,
                             'formation_thickness': False,
                             'polarity': False,
                             'strat_offset': False,
                             'contact_dips': False})
        else:
            workflow.update({'seismic_section': False,
                             'cover_map': False,
                             'near_fault_interpolations': False,
                             'fold_axial_traces': False,
                             'stereonets': True,
                             'formation_thickness': True,
                             'polarity': False,
                             'strat_offset': True,
                             'contact_dips': False})
        self.workflow = workflow

    def update_config(self,
                      bbox_3d={
                          "minx": 0,
                          "maxx": 0,
                          "maxx": 0,
                          "maxy": 0,
                          "base": -10000,
                          "top": 1200,
                      },
                      dtm_crs={'init': 'EPSG:4326'},
                      proj_crs=None,
                      step_out=None,
                      ):

        if bbox_3d["minx"] == 0 and bbox_3d["maxx"] == 0:
            bbox_3d.update({
                "minx": self.proj_bounds[0],
                "minx": self.proj_bounds[1],
                "minx": self.proj_bounds[2],
                "minx": self.proj_bounds[3],
            })

        if proj_crs is None:
            proj_crs = self.proj_crs

        if step_out is None:
            step_out = self.step_out

        # TODO: Make crs defaults and specifiable not from config
        minx, miny, maxx, maxy = tuple([bbox_3d["minx"], bbox_3d["miny"],
                                        bbox_3d["maxx"], bbox_3d["maxy"]])
        lat_point_list = [miny, miny, maxy, maxy, maxy]
        lon_point_list = [minx, maxx, maxx, minx, minx]
        bbox_geom = Polygon(zip(lon_point_list, lat_point_list))
        polygon = gpd.GeoDataFrame(
            index=[0], crs=proj_crs, geometry=[bbox_geom])
        self.config = Config(
            self.geology_file, self.fault_file, self.fold_file,
            self.structure_file, self.mindep_file,
            bbox_3d, polygon, step_out, dtm_crs, proj_crs, self.c_l
        )

        # Store important data frames and display
        self.config.preprocess("plot")

    def run(self):
        print("Generating topology analyser input...")
        self.config.export_csv()
        self.config.runMap2Model()
        self.config.load_dtm()
        self.config.join_features()

        if(self.workflow['cover_map']):
            # Depth to basement calculation
            self.config.calc_depth_grid()
        else:
            # FOr now this function just zeros dtb and exits prematurely
            self.config.calc_depth_grid()

        self.config.export_orientations()
        self.config.export_contacts()

        self.config.test_interpolation()

        self.config.export_faults()
        self.config.process_plutons()

        # Seismic section is in the hamersely model area
        # TODO: Implement this option better and test with Turner
        if (self.workflow['seismic_section']):
            self.config.extract_section_features(seismic_line_file="",
                                                 seismic_bbox_file="",
                                                 seismic_interp_file="")

        if(self.workflow['contact_dips']):
            self.config.propagate_contacts_dips()

        if(self.workflow['formation_thickness']):
            self.config.calc_thickness()

        if(self.workflow['fold_axial_traces']):
            self.config.create_fold_axial_trace_points()

        # Prepocess model inputs
        inputs = ('')
        if(self.workflow['model_engine'] == 'geomodeller'):
            inputs = ('invented_orientations', 'intrusive_orientations',
                      'fat_orientations', 'fault_tip_contacts', 'contact_orientations')
        elif(self.workflow['model_engine'] == 'loopstructural'):
            inputs = ('invented_orientations',
                      'fat_orientations', 'contact_orientations')
        elif(self.workflow['model_engine'] == 'gempy'):
            inputs = ('invented_orientations', 'interpolated_orientations',
                      'fat_orientations', 'contact_orientations')
        elif(self.workflow['model_engine'] == 'noddy'):
            inputs = ('')

        self.config.postprocess(inputs, self.workflow)

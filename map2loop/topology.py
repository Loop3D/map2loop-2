import os
from tabnanny import verbose
from map2loop.m2l_enums import Datatype, VerboseLevel
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import functools
import operator
import shapely.affinity
from shapely.ops import split
from shapely.geometry import (
    Point,
    LineString,
    MultiLineString,
    GeometryCollection,
    Polygon,
)
from math import degrees, atan2, acos, degrees
import warnings
import rasterio
import beartype

from . import m2l_utils
from .m2l_utils import display
from .config import Config


class Topology(object):
    def __init__(self):
        pass

    def save_parfile(
        c_l,
        output_path,
        geology_file_csv,
        fault_file_csv,
        structure_file_csv,
        mindep_file_csv,
        minx,
        maxx,
        miny,
        maxy,
        deposit_dist,
        commodities,
    ):
        with open("Parfile", "w") as f:
            f.write(
                "--- COLUMN NAMES IN CSV DATA FILES: -------------------------------------------------------------\n"
            )
            f.write("OBJECT COORDINATES              =WKT\n")
            f.write("FAULT: ID                       =" + "GEOMETRY_OBJECT_ID" + "\n")
            f.write("FAULT: FEATURE                  =" + "FEATURE" + "\n")
            f.write("POLYGON: ID                     =" + "GEOMETRY_OBJECT_ID" + "\n")
            f.write("POLYGON: LEVEL1 NAME            =" + "CODE" + "\n")
            f.write("POLYGON: LEVEL2 NAME            =" + "GROUP" + "\n")
            f.write("POLYGON: MIN AGE                =" + "MIN_AGE" + "\n")
            f.write("POLYGON: MAX AGE                =" + "MAX_AGE" + "\n")
            f.write("POLYGON: CODE                   =" + "UNIT_NAME" + "\n")
            f.write("POLYGON: DESCRIPTION            =" + "DESCRIPTION" + "\n")
            f.write("POLYGON: ROCKTYPE1              =" + "ROCKTYPE1" + "\n")
            f.write("POLYGON: ROCKTYPE2              =" + "ROCKTYPE2" + "\n")
            f.write("DEPOSIT: SITE CODE              =" + c_l["msc"] + "\n")
            f.write("DEPOSIT: SITE TYPE              =" + c_l["mst"] + "\n")
            f.write("DEPOSIT: SITE COMMODITY         =" + c_l["mscm"] + "\n")

            f.write(
                "--- SOME CONSTANTS: ----------------------------------------------------------------------------\n"
            )
            f.write("FAULT AXIAL FEATURE NAME        =" + c_l["fold"] + "\n")
            f.write("SILL UNIT DESCRIPTION CONTAINS  =" + c_l["sill"] + "\n")
            f.write(
                "IGNEOUS ROCKTYPE CONTAINS                           ="
                + c_l["intrusive"]
                + "\n"
            )
            f.write(
                "VOLCANIC ROCKTYPE CONTAINS                          ="
                + c_l["volcanic"]
                + "\n"
            )
            f.write(
                "IGNORE DEPOSITS WITH SITE TYPE                      =Infrastructure\n"
            )
            f.write("Intersect Contact With Fault: angle epsilon (deg)   =1.0\n")
            f.write("Intersect Contact With Fault: distance epsilon (m)  =15.0\n")
            f.write("Distance buffer (fault stops on another fault) (m)  =20.0\n")
            f.write(
                "Distance buffer (point on contact) (m)              ="
                + str(deposit_dist)
                + "\n"
            )
            f.write("Intersect polygons distance buffer (for bad maps)   =3.\n")
            f.write(
                "------------------------------------------------------------------------------------------------\n"
            )
            f.write(
                "Path to the output data folder                      ="
                + output_path
                + "\n"
            )
            f.write(
                "Path to geology data file                           ="
                + geology_file_csv
                + "\n"
            )
            f.write(
                "Path to faults data file                            ="
                + fault_file_csv
                + "\n"
            )
            f.write(
                "Path to mineral deposits data file                  ="
                + mindep_file_csv
                + "\n"
            )
            f.write(
                "------------------------------------------------------------------------------------------------\n"
            )
            f.write(
                "Clipping window X1 Y1 X2 Y2 (zeros for infinite)    ="
                + str(minx)
                + " "
                + str(miny)
                + " "
                + str(maxx)
                + " "
                + str(maxy)
                + "\n"
            )
            f.write("Min length fraction for strat/fault graphs          =0.0\n")
            f.write(
                "Graph edge width categories (three doubles)         =2000. 20000. 200000.\n"
            )
            f.write(
                "Graph edge direction (0-min age, 1-max age, 2-avg)  =" + str(2) + "\n"
            )
            f.write(
                "Deposit names for adding info on the graph          ="
                + commodities
                + "\n"
            )
            f.write("Partial graph polygon ID                            =32\n")
            f.write("Partial graph depth                                 =4\n")
            f.write("Map subregion size dx, dy [m] (zeros for full map)  =0. 0.\n")
            f.write(
                "------------------------------------------------------------------------------------------------\n"
            )

    ####################################
    # parse stratigraphy GML file to get number of series and series names
    #
    # get_series(path_in,id_label)
    # Args:
    # path_in path to gml file
    # id_label code for eacah node that defines node type (group or formation)
    ####################################
    def get_series(path_in, id_label):

        # load a stratigraphy with groups (no longer true?: needs to feed via yed!!)
        G = nx.read_gml(path_in, label=id_label)

        glabels = {}
        for n in list(
            G.nodes
        ):  # Find out total number of groups and their names groups
            if "isGroup" in G.nodes[n]:
                glabels[n] = (
                    G.nodes[n]["LabelGraphics"]["text"]
                    .replace(" ", "_")
                    .replace("-", "_")
                )
        return (glabels, G)

    ####################################
    # parse stratigraphy GML file to save units for each series
    #
    # Save out csv of maximum and minimum ages of formations within each group so that groups can be better sorted.
    # save_units(G,tmp_path,glabels)
    # Args:
    # G networkx format stratigraphy graph
    # tmp_path directory of temporary outputs glabels list of group names
    #
    # The choice of what constitutes basic unit and what a group of units is defined in the c_l codes. Not even sure we need two levels but it seemed like a good idea at the time. Text outputs list alternate topologies for series and surfaces, which if confirmed by comapring max-min ages will be a nice source of uncertainty.
    ####################################
    @beartype.beartype
    def save_units(graph, graph_labels, config: Config):
        if config.run_flags["aus"]:
            default_asud_strat_file = "https://gist.githubusercontent.com/yohanderose/3b257dc768fafe5aaf70e64ae55e4c42/raw/8598c7563c1eea5c0cd1080f2c418dc975cc5433/ASUD.csv"
            ASUD = pd.read_csv(default_asud_strat_file, sep=",")
        for (
            p
        ) in (
            graph_labels
        ):  # process each group, removing nodes that are not part of that group, and other groups
            GD = graph.copy()  # temporary copy of full graph
            # print()
            # print(p,graph_labels[p].replace(" ","_").replace("-","_"),"----------------------")
            nlist = list(graph.nodes)

            for n in nlist:  # Calculate total number of groups and their names groups
                if "gid" in GD.nodes[n]:  # normal node
                    if (
                        GD.nodes[n]["gid"] != p
                    ):  # normal node but not part of current group
                        GD.remove_node(n)
                else:  # group node
                    GD.remove_node(n)

            labels = {}
            for node in GD.nodes():  # local store of node labels
                labels[node] = (
                    graph.nodes[node]["LabelGraphics"]["text"]
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace("?", "_")
                )

            cycles = nx.simple_cycles(GD)
            for cy in cycles:
                found = False
                if config.run_flags["aus"]:
                    len_cy = len(cy)
                    for i in range(len_cy - 1):
                        glabel_0 = GD.nodes[cy[i]]["LabelGraphics"]["text"]
                        glabel_1 = GD.nodes[cy[i + 1]]["LabelGraphics"]["text"]
                        edge = ASUD.loc[
                            (ASUD["over"] == glabel_0) & (ASUD["under"] == glabel_1)
                        ]
                        if len(edge) == 0 and (
                            not ("isGroup" in GD.nodes[cy[i]])
                            and not ("isGroup" in GD.nodes[cy[i + 1]])
                        ):
                            if not GD.has_edge(cy[i], cy[i + 1]):
                                continue
                            else:
                                warning_msg = (
                                    "map2loop warning 1: Stratigraphic relationship: "
                                    + str(GD.nodes[cy[i]]["LabelGraphics"]["text"])
                                    + " overlies "
                                    + str(GD.nodes[cy[i + 1]]["LabelGraphics"]["text"])
                                    + " removed to prevent cycle"
                                )
                                warnings.warn(warning_msg)
                                GD.remove_edge(cy[i], cy[i + 1])
                                found = True

                    if not found:
                        glabel_0 = GD.nodes[cy[len_cy - 1]]["LabelGraphics"]["text"]
                        glabel_1 = GD.nodes[cy[0]]["LabelGraphics"]["text"]
                        edge = ASUD.loc[
                            (ASUD["over"] == glabel_0) & (ASUD["under"] == glabel_1)
                        ]
                        if len(edge) == 0 and (
                            not ("isGroup" in GD.nodes[cy[len_cy - 1]])
                            and not ("isGroup" in GD.nodes[cy[0]])
                        ):
                            if GD.has_edge(cy[len_cy - 1], cy[0]):
                                warning_msg = (
                                    "map2loop warning 1: Stratigraphic relationship: "
                                    + str(
                                        GD.nodes[cy[len_cy - 1]]["LabelGraphics"][
                                            "text"
                                        ]
                                    )
                                    + " overlies "
                                    + str(GD.nodes[cy[0]]["LabelGraphics"]["text"])
                                    + " removed to prevent cycle"
                                )
                                warnings.warn(warning_msg)
                                GD.remove_edge(cy[len_cy - 1], cy[0])
                                found = True
                        if not found:
                            warning_msg = (
                                "map2loop warning 2: Stratigraphic relationship: "
                                + str(GD.nodes[cy[0]]["LabelGraphics"]["text"])
                                + " overlies "
                                + str(GD.nodes[cy[1]]["LabelGraphics"]["text"])
                                + " removed to prevent cycle"
                            )
                            warnings.warn(warning_msg)
                            if GD.has_edge(cy[len_cy - 1], cy[0]):
                                if GD.has_edge(cy[0], cy[1]):
                                    GD.remove_edge(cy[0], cy[1])

                else:
                    warning_msg = (
                        "map2loop warning 3: Stratigraphic relationship: "
                        + str(GD.nodes[cy[0]]["LabelGraphics"]["text"])
                        + " overlies "
                        + str(GD.nodes[cy[1]]["LabelGraphics"]["text"])
                        + " removed to prevent cycle"
                    )
                    warnings.warn(warning_msg)
                    if GD.has_edge(cy[0], cy[1]):
                        GD.remove_edge(cy[0], cy[1])

            if config.verbose_level == VerboseLevel.ALL:
                plt.figure(p + 1)  # display strat graph for one group
                plt.title(graph_labels[p])
                plt.tight_layout()
                nx.draw_networkx(
                    GD, pos=nx.kamada_kawai_layout(GD), arrows=True, with_labels=False
                )
                nx.draw_networkx_labels(
                    GD,
                    pos=nx.kamada_kawai_layout(GD),
                    labels=labels,
                    font_size=12,
                    font_family="sans-serif",
                )

            one = True
            if one:
                # one possible sorted directional graphs
                nlist = list(nx.topological_sort(GD))
            else:
                # all possible sorted directional graphs
                nlist = list(nx.all_topological_sorts(GD))

            f = open(
                os.path.join(
                    config.tmp_path,
                    graph_labels[p]
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace("?", "_")
                    + ".csv",
                ),
                "w",
            )

            if one:
                f.write("Choice " + str(0))
                for n in range(0, len(GD)):  # display nodes for one sorted graph

                    f.write(
                        ","
                        + graph.nodes[nlist[n]]["LabelGraphics"]["text"]
                        .replace(" ", "_")
                        .replace("-", "_")
                        .replace("?", "_")
                    )

                f.write("\n")
            else:
                for m in range(10):  # process first ten sorted graphs
                    f.write("Choice " + str(m))
                    for n in range(0, len(GD)):  # display nodes for one sorted graph
                        # print(nlist[m][n],graph.nodes[nlist[m][n]]['LabelGraphics']['text'].replace(" ","_").replace("-","_"))
                        f.write(
                            ","
                            + graph.nodes[nlist[m][n]]["LabelGraphics"]["text"]
                            .replace(" ", "_")
                            .replace("-", "_")
                            .replace("?", "_")
                        )
                    # if(m<len(nlist)-1):
                    # print("....")
                    f.write("\n")
            f.close()

    ####################################
    # save out a list of max/min/ave ages of all formations in a group
    #
    # abs_age_groups(geol,tmp_path,c_l)
    # Args:
    # geol geopandas GeoDataBase of geology polygons
    # c_l dictionary of codes and labels specific to input geo information layers
    #
    # Save out csv of maximum and minimum ages of formations within each group so that groups can be better sorted.
    ####################################
    def abs_age_groups(geol, tmp_path, c_l):
        groups = []
        info = []
        ages = []
        for indx, a_poly in geol.iterrows():  # loop through all polygons
            if str(a_poly["GROUP"]) == "None":
                grp = (
                    a_poly["UNIT_NAME"]
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace("?", "_")
                )
            else:
                grp = (
                    a_poly["GROUP"]
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace("?", "_")
                )
            # print(grp)
            if not grp in groups:
                groups += [(grp)]

            info += [(grp, a_poly["MIN_AGE"], a_poly["MAX_AGE"])]

        # display(info)
        # display(groups)
        for j in range(0, len(groups)):
            # print(groups[j],'------------------')
            min_age = 1e10
            max_age = 0
            for i in range(0, len(info)):
                if info[i][0] == groups[j]:
                    if float(info[i][1]) < min_age:
                        min_age = float(info[i][1])
                        min_ind = i
                    if float(info[i][2]) > max_age:
                        max_age = float(info[i][2])
                        max_ind = i
            #        print(groups[j],min_age,max_age,(max_age+min_age)/2)
            ages += [(groups[j], min_age, max_age, (max_age + min_age) / 2)]
        #    print()
        #    for j in range(0,len(ages)):
        #        print(ages[j][0],ages[j][1],ages[j][2],ages[j][3],sep='\t')
        #    print()

        slist = sorted(ages, key=lambda l: l[3])
        f = open(os.path.join(tmp_path, "age_sorted_groups.csv"), "w")
        f.write("index,group_,min,max,ave\n")
        for j in range(0, len(slist)):
            f.write(
                str(j)
                + ","
                + slist[j][0]
                + ","
                + str(slist[j][1])
                + ","
                + str(slist[j][2])
                + ","
                + str(slist[j][3])
                + "\n"
            )
        f.close()

    ####################################
    # save out tables of groups and sorted formation data
    #
    # save_group(G,tmp_path,glabels,geol_clip,c_l)
    # G networkx format stratigraphy graph
    # tmp_path directory of temporary outputs glabels list of group names
    # geol_clip path to clipped geology layer c_l dictionary of codes and labels specific to input geo information layers
    #
    # Takes stratigraphy graph created by map2model c++ code to generate list of groups found in the region of interest
    # Uses first of each possible set of toplogies per unit and per group, which is arbitrary. On the other hand we are not checking relative ages again to see if this helps reduce ambiguity, which I think it would.
    ####################################

    @beartype.beartype
    def save_group(G, glabels, config: Config, map_data):
        Gp = nx.Graph().to_directed()  # New Group graph

        # geology_file = gpd.read_file(os.path.join(config.tmp_path, 'geol_clip.shp'))

        geol = map_data.get_map_data(Datatype.GEOLOGY)
        Topology.abs_age_groups(geol, config.tmp_path, config.c_l)
        local_geol = geol.copy()
        local_geol.drop_duplicates(subset="UNIT_NAME", inplace=True)

        local_geol.set_index("UNIT_NAME", inplace=True)
        # display(local_geol)

        gp_ages = pd.read_csv(os.path.join(config.tmp_path, "age_sorted_groups.csv"))
        # display(gp_ages)
        gp_ages.set_index("group_", inplace=True)

        if config.verbose_level == VerboseLevel.ALL:
            display(gp_ages)
        gp_ids = []
        nlist = list(G.nodes)
        for n in nlist:  # Find out total number of groups and their names groups
            if "isGroup" in G.nodes[n]:
                Gp.add_nodes_from([n])

        if config.verbose_level == VerboseLevel.ALL:
            display(gp_ids)
        for e in G.edges:
            if G.nodes[e[0]]["gid"] != G.nodes[e[1]]["gid"]:
                glabel_0 = G.nodes[e[0]]["LabelGraphics"]["text"]
                glabel_1 = G.nodes[e[1]]["LabelGraphics"]["text"]
                # print(glabel_0, glabel_1)
                # print(df[df.CODE == "A-FOh-xs-f"])
                # exit()
                if str(local_geol.loc[glabel_0]["GROUP"]) == "None":
                    grp0 = (
                        glabel_0.replace(" ", "_").replace("-", "_").replace("?", "_")
                    )
                else:
                    grp0 = (
                        local_geol.loc[glabel_0]["GROUP"]
                        .replace(" ", "_")
                        .replace("-", "_")
                    )
                if str(local_geol.loc[glabel_1]["GROUP"]) == "None":
                    grp1 = (
                        glabel_1.replace(" ", "_").replace("-", "_").replace("?", "_")
                    )
                else:
                    grp1 = (
                        local_geol.loc[glabel_1]["GROUP"]
                        .replace(" ", "_")
                        .replace("-", "_")
                        .replace("?", "_")
                    )

                # print(glabel_0,glabel_1,gp_ages.loc[grp0],gp_ages.loc[grp1])
                if grp0 in glabels.values() and grp1 in glabels.values():
                    if gp_ages.loc[grp0]["ave"] < gp_ages.loc[grp1]["ave"]:
                        Gp.add_edge(G.nodes[e[0]]["gid"], G.nodes[e[1]]["gid"])

        GpD = Gp.copy()  # temporary copy of full graph
        GpD2 = Gp.copy()  # temporary copy of full graph

        for e in GpD2.edges:  # remove duplicate edges with opposite directions
            for f in GpD.edges:
                # arbitrary choice to ensure edge is not completely removed
                if e[0] == f[1] and e[1] == f[0] and e[0] < f[0]:
                    Gp.remove_edge(e[0], e[1])
        if config.verbose_level == VerboseLevel.ALL:
            display(glabels)
            plt.figure(1)  # display strat graph for one group
            plt.title("groups")
            if len(glabels) > 1:
                nx.draw_networkx(
                    Gp, pos=nx.kamada_kawai_layout(Gp), arrows=True, with_labels=False
                )
                nx.draw_networkx_labels(
                    Gp,
                    pos=nx.kamada_kawai_layout(Gp),
                    font_size=12,
                    font_family="sans-serif",
                )

        # when lots of groups very large number of possibilities so short cut to first choice.
        if len(gp_ages) > 10:
            # all possible sorted directional graphs
            glist = list(nx.topological_sort(Gp))
            if config.verbose_level != VerboseLevel.NONE:
                print("group choices: 1 (more than 10 groups)")

            f = open(os.path.join(config.tmp_path, "groups.csv"), "w")
            glen = len(glist)
            f.write("Choice 0")
            for n in range(0, glen):
                f.write("," + str(glabels[glist[n]]))  # check underscore
            f.write("\n")
            f.close()
        else:
            # all possible sorted directional graphs
            glist = list(nx.all_topological_sorts(Gp))
            if config.verbose_level != VerboseLevel.NONE:
                print("group choices:", len(glist))

            f = open(os.path.join(config.tmp_path, "groups.csv"), "w")
            glen = len(glist)
            if glen > 100:
                glen = 100
            for n in range(0, glen):
                f.write("Choice " + str(n))
                for m in range(0, len(glist[0])):
                    f.write("," + str(glabels[glist[n][m]]))  # check underscore
                f.write("\n")
            f.close()

        # display(glist)
        nx.write_gml(Gp, os.path.join(config.tmp_path, "groups.gml"))
        # plt.show()

        # g=open(os.path.join(config.tmp_path,'groups.csv'),"r")
        # contents =g.readlines()
        # g.close
        # hdr=contents[0].split(" ")
        contents = np.genfromtxt(
            os.path.join(config.tmp_path, "groups.csv"), delimiter=",", dtype="U100"
        )

        # display('lencon',len(contents[0]))
        k = 0
        ag = open(os.path.join(config.tmp_path, "all_sorts.csv"), "w")
        ag.write("index,group number,index in group,number in group,code,group\n")
        if len(contents.shape) == 1:
            for i in range(1, len(contents)):
                ucontents = np.genfromtxt(
                    os.path.join(
                        config.tmp_path,
                        contents[i].replace("\n", "").replace(" ", "_") + ".csv",
                    ),
                    delimiter=",",
                    dtype="U100",
                )
                # f=open(os.path.join(config.tmp_path, contents[i].replace("\n","").replace(" ","_")+".csv"),"r")#check underscore
                # ucontents =f.readlines()
                # f.close
                # print(len(ucontents.shape),ucontents)
                if len(ucontents.shape) == 1:
                    for j in range(1, len(ucontents)):
                        ag.write(
                            str(k)
                            + ","
                            + str(i)
                            + ","
                            + str(j)
                            + ","
                            + str(len(ucontents) - 1)
                            + ","
                            + ucontents[j].replace("\n", "")
                            + ","
                            + contents[i]
                            .replace("\n", "")
                            .replace(" ", "_")
                            .replace("-", "_")
                            + "\n"
                        )
                        k = k + 1
                else:
                    for j in range(1, len(ucontents[0])):
                        ag.write(
                            str(k)
                            + ","
                            + str(i)
                            + ","
                            + str(j)
                            + ","
                            + str(len(ucontents[0]) - 1)
                            + ","
                            + ucontents[0][j].replace("\n", "")
                            + ","
                            + contents[i]
                            .replace("\n", "")
                            .replace(" ", "_")
                            .replace("-", "_")
                            + "\n"
                        )
                        k = k + 1
        else:
            for i in range(1, len(contents[0])):
                ucontents = np.genfromtxt(
                    os.path.join(
                        config.tmp_path,
                        contents[0][i].replace("\n", "").replace(" ", "_") + ".csv",
                    ),
                    delimiter=",",
                    dtype="U100",
                )
                # f=open(os.path.join(config.tmp_path,contents[i].replace("\n","").replace(" ","_")+".csv"),"r")#check underscore
                # ucontents =f.readlines()
                # f.close
                # print(len(ucontents.shape),ucontents)
                if len(ucontents.shape) == 1:
                    for j in range(1, len(ucontents)):
                        ag.write(
                            str(k)
                            + ","
                            + str(i)
                            + ","
                            + str(j)
                            + ","
                            + str(len(ucontents) - 1)
                            + ","
                            + ucontents[j].replace("\n", "")
                            + ","
                            + contents[0][i]
                            .replace("\n", "")
                            .replace(" ", "_")
                            .replace("-", "_")
                            + "\n"
                        )
                        k = k + 1
                else:
                    for j in range(1, len(ucontents[0])):
                        ag.write(
                            str(k)
                            + ","
                            + str(i)
                            + ","
                            + str(j)
                            + ","
                            + str(len(ucontents[0]) - 1)
                            + ","
                            + ucontents[0][j].replace("\n", "")
                            + ","
                            + contents[0][i]
                            .replace("\n", "")
                            .replace(" ", "_")
                            .replace("-", "_")
                            + "\n"
                        )
                        k = k + 1
        ag.close()

    ####################################
    # save out fault fault relationship information as array
    #
    # parse_fault_relationships(graph_path,tmp_path,output_path)
    # graph_path path to graphs
    # tmp_path path to tmp directory
    # output_path path to output directory
    #
    # Saves fault vs unit, group and fault relationship tables using outputs from map2model c++ code
    ####################################
    @beartype.beartype
    def parse_fault_relationships(config: Config):
        uf = open(os.path.join(config.graph_path, "unit-fault-intersection.txt"), "r")
        contents = uf.readlines()
        uf.close()

        all_long_faults = np.genfromtxt(
            os.path.join(config.output_path, "fault_dimensions.csv"),
            delimiter=",",
            dtype="U100",
        )
        n_faults = len(all_long_faults)
        # print(n_faults)
        all_faults = {}
        unique_list = []
        # display(unique_list)
        for i in range(1, n_faults):
            f = all_long_faults[i][0]
            # print(all_long_faults[i][0])
            if f not in unique_list:
                unique_list.append(f.replace("Fault_", ""))

        # display('Long Faults',unique_list)

        uf = open(os.path.join(config.output_path, "unit-fault-relationships.csv"), "w")
        uf.write(
            "code,"
            + str(unique_list)
            .replace("[", "Fault_")
            .replace(",", ",Fault_")
            .replace("'", "")
            .replace("]", "")
            .replace(" ", "")
            + "\n"
        )
        for row in contents:
            row = row.replace("\n", "").split("{")
            unit = row[0].split(",")
            faults = row[1].replace("}", "").replace(" ", "").split(",")
            ostr = str(unit[1]).strip().replace(" ", "_").replace("-", "_")
            for ul in unique_list:
                out = [item for item in faults if ul == item]
                if len(out) > 0:
                    ostr = ostr + ",1"
                else:
                    ostr = ostr + ",0"
            uf.write(ostr + "\n")
        uf.close()

        summary = pd.read_csv(os.path.join(config.tmp_path, "all_sorts_clean.csv"))
        summary.set_index("code", inplace=True)
        # display('summary',summary)
        uf_rel = pd.read_csv(
            os.path.join(config.output_path, "unit-fault-relationships.csv")
        )

        groups = summary.group.unique()
        ngroups = len(summary.group.unique())
        supergroups = summary.supergroup.unique()
        nsupergroups = len(summary.supergroup.unique())
        # print(ngroups,'groups',groups,groups[0])
        uf_array = uf_rel.to_numpy()
        gf_array = np.zeros((ngroups, uf_array.shape[1]), dtype="U100")

        for i in range(0, ngroups):
            for j in range(0, len(uf_rel)):
                if uf_rel.iloc[j][0] in summary.index.values:
                    gsummary = summary.loc[uf_rel.iloc[j][0]]
                    if groups[i].replace("\n", "") == gsummary["group"]:
                        for k in range(1, len(uf_rel.iloc[j])):
                            if uf_rel.iloc[j][k] == 1:
                                gf_array[i - 1, k] = "1"
                            else:
                                continue
                    else:
                        continue
                else:
                    continue

        ug = open(
            os.path.join(config.output_path, "group-fault-relationships.csv"), "w"
        )
        ug.write("group")
        for k in range(1, len(uf_rel.columns)):
            ug.write("," + uf_rel.columns[k])
        ug.write("\n")
        for i in range(0, ngroups):
            ug.write(groups[i].replace("\n", ""))
            for k in range(1, len(uf_rel.columns)):
                if gf_array[i - 1, k] == "1":
                    ug.write(",1")
                else:
                    ug.write(",0")
            ug.write("\n")

        ug.close()

        summary = pd.read_csv(os.path.join(config.tmp_path, "all_sorts_clean.csv"))
        summary.sort_values(by="supergroup", inplace=True)
        summary.drop_duplicates(subset="group", inplace=True)

        groups = summary.group.unique()
        ngroups = len(summary.group.unique())
        supergroups = summary.supergroup.unique()
        supergroups = pd.DataFrame(supergroups)
        supergroups.sort_values(by=0, inplace=True)

        index = np.arange(0, len(supergroups))
        supergroups["index"] = index
        summary.set_index("group", inplace=True)

        uf_rel = pd.read_csv(
            os.path.join(config.output_path, "unit-fault-relationships.csv")
        )

        uf_array = uf_rel.to_numpy()
        gf_array = np.zeros((ngroups, uf_array.shape[1]), dtype="U100")

        group_faults = pd.read_csv(
            os.path.join(config.output_path, "group-fault-relationships.csv")
        )
        group_faults.sort_values(by="group", inplace=True)

        group_faults.set_index("group", inplace=True)
        uf_array = group_faults.to_numpy()
        sg_array = np.zeros((len(supergroups), len(uf_rel.columns) - 1))
        supergroups.set_index(0, inplace=True)

        i = 0
        for ind, gp in group_faults.iterrows():
            for ind2, sgroup in summary.iterrows():
                if sgroup["supergroup"] == summary.loc[ind]["supergroup"]:
                    for k in range(0, len(uf_rel.columns) - 1):
                        if uf_array[i, k] == 1:
                            sg_array[
                                supergroups.loc[summary.loc[ind]["supergroup"]][
                                    "index"
                                ],
                                k,
                            ] = 1

            i = i + 1

        sg = open(
            os.path.join(config.output_path, "supergroup-fault-relationships.csv"), "w"
        )
        sg.write("supergroup")
        for k in range(1, len(uf_rel.columns)):
            sg.write("," + uf_rel.columns[k])
        sg.write("\n")
        for k, sgroups in supergroups.iterrows():
            sg.write(k)

            for i in range(0, sg_array.shape[1]):

                sg.write("," + str(sg_array[sgroups["index"], i]))
            sg.write("\n")

        sg.close()

        uf = open(os.path.join(config.graph_path, "fault-fault-intersection.txt"), "r")
        contents = uf.readlines()
        uf.close()
        # display(unique_list)
        unique_list_ff = []

        for row in contents:
            row = row.replace("\n", "").split("{")
            fault_1o = row[0].split(",")
            fault_1o = fault_1o[1]

            faults_2o = (
                row[1].replace("(", "").replace(")", "").replace("}", "").split(",")
            )

            if (fault_1o.replace(" ", "") not in unique_list_ff) and (
                fault_1o.replace(" ", "") in unique_list
            ):
                unique_list_ff.append(fault_1o.replace(" ", ""))
            for i in range(0, len(faults_2o), 3):
                if (faults_2o[i].replace(" ", "") not in unique_list_ff) and (
                    faults_2o[i].replace(" ", "") in unique_list
                ):
                    unique_list_ff.append(faults_2o[i].replace(" ", ""))
        # display(unique_list)

        G = nx.DiGraph()
        ff = open(
            os.path.join(config.output_path, "fault-fault-relationships.csv"), "w"
        )
        ff.write("fault_id")
        for i in range(0, len(unique_list_ff)):
            ff.write("," + "Fault_" + unique_list_ff[i])
            G.add_node("Fault_" + unique_list_ff[i])
        ff.write("\n")

        for i in range(0, len(unique_list_ff)):  # loop thorugh rows
            ff.write("Fault_" + unique_list_ff[i])
            found = False
            # for j in range(0,len(unique_list)):
            for row in contents:  # loop thorugh known intersections
                row = row.replace("\n", "").split("{")
                fault_1o = row[0].split(",")
                fault_1o = fault_1o[1]
                faults_2o = (
                    row[1].replace("(", "").replace(")", "").replace("}", "").split(",")
                )

                # correct first order fault for this row
                if unique_list_ff[i].replace(" ", "") == fault_1o.replace(" ", ""):
                    found = True
                    for k in range(0, len(unique_list_ff)):  # loop through columns
                        found2 = False
                        if k == i:  # no self intersections
                            ff.write(",0")
                        else:
                            # loop through second order faults for this row
                            for f2o in range(0, len(faults_2o), 3):
                                if faults_2o[f2o].replace(" ", "") == unique_list_ff[
                                    k
                                ].replace(" ", ""):
                                    ff.write(",1")
                                    G.add_edge(
                                        "Fault_" + unique_list_ff[i],
                                        "Fault_" + faults_2o[f2o].replace(" ", ""),
                                    )
                                    found2 = True
                                    break

                        if not found2 and k != i:
                            # this is not a second order fault for this row
                            ff.write(",0")
                if found:
                    break
            if (
                not found
            ):  # this fault is not a first order fault relative to another fault
                for i in range(0, len(unique_list_ff)):
                    ff.write(",0")

            ff.write("\n")

        ff.close()

        if config.verbose_level == VerboseLevel.ALL and len(unique_list_ff) > 0:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            nx.draw(G, ax=ax, with_labels=True, font_weight="bold")
            plt.title("Fault Network")
            plt.show()

        GD = G.copy()
        edges = list(G.edges)
        cycles = list(nx.simple_cycles(G))

        for i in range(0, len(edges)):  # remove first edge from fault cycle
            for j in range(0, len(cycles)):
                if edges[i][0] == cycles[j][0] and edges[i][1] == cycles[j][1]:
                    if GD.has_edge(edges[i][0], edges[i][1]):
                        GD.remove_edge(edges[i][0], edges[i][1])
                        print("fault cycle removed:", edges[i][0], edges[i][1])

        g = open(os.path.join(config.graph_path, "fault-fault-intersection.txt"), "r")
        contents = g.readlines()
        for line in contents:
            parts = line.split(",")
            f1 = "Fault_" + parts[1]
            f1 = f1.replace(" ", "")
            for fn in range(int((len(parts) - 2) / 3)):
                f2 = "Fault_" + parts[2 + (fn * 3)].replace("{", "").replace("(", "")
                f2 = f2.replace(" ", "")
                topol = parts[3 + (fn * 3)].replace(" ", "")
                angle = parts[4 + (fn * 3)].replace("}", "").replace(")", "")
                angle = (
                    angle.replace(" ", "").replace("}", "").replace(")", "").rstrip()
                )
                if GD.has_edge(f1, f2):
                    # TODO: replace with .loc as pandas warning for setting on a slice
                    GD[f1][f2]["angle"] = angle
                    GD[f1][f2]["topol"] = topol

        for f in unique_list:
            if not GD.has_node("Fault_" + f):
                GD.add_node("Fault_" + f)

        nx.write_gml(GD, os.path.join(config.tmp_path, "fault_network.gml"))

        f = open(os.path.join(config.tmp_path, "fault_network_edges.csv"), "w")
        f.write("fault_1,fault_2,angle,topol\n")
        for e in GD.edges():
            f.write(
                "{},{},{},{}\n".format(
                    e[0], e[1], GD[e[0]][e[1]]["angle"], GD[e[0]][e[1]]["topol"]
                )
            )
        f.close()

        if config.verbose_level != VerboseLevel.NONE:
            print(
                "edges saved as :",
                os.path.join(config.tmp_path, "fault_network_edges.csv"),
            )

        try:
            if config.verbose_level != VerboseLevel.NONE:
                print("cycles", list(nx.simple_cycles(GD)))
        except:
            if config.verbose_level != VerboseLevel.NONE:
                print("no cycles")

    ####################################
    # save out geology polygons in WKT format
    #
    # save_geol_wkt(sub_geol,geology_file_csv, c_l)
    # sub_geol geopandas format geology layer geology_file_csv path to output WKT format file
    # c_l dictionary of codes and labels specific to input geo information layers
    #
    # Saves geology layer as WKT format for use by map2model c++ code
    ####################################
    def save_geol_wkt(
        sub_geol, geology_file_csv, c_l, hint_flag, verbose_level=VerboseLevel.ALL
    ):
        # hint_flag=False
        if hint_flag == True:
            if verbose_level != VerboseLevel.NONE:
                print("Using ENS age hints")
            code_hint_file = (
                "../test_data3/data/code_age_hint.csv"  # input code_hint file
            )
            code_hints = pd.read_csv(code_hint_file, sep=",")
            code_hints.drop_duplicates(inplace=True)
            code_hints.set_index("code", inplace=True)
            hint_list = []
            for indx, row in code_hints.iterrows():
                if not indx in hint_list:
                    hint_list.append(indx)
        f = open(geology_file_csv, "w+")
        f.write(
            "WKT\t"
            + "GEOMETRY_OBJECT_ID"
            + "\t"
            + "CODE".replace("\n", "")
            + "\t"
            + "GROUP"
            + "\t"
            + "MIN_AGE".replace("\n", "")
            + "\t"
            + "MAX_AGE".replace("\n", "")
            + "\t"
            + "UNIT_NAME".replace("\n", "")
            + "\t"
            + "ROCKTYPE1".replace("\n", "")
            + "\t"
            + "ROCKTYPE2".replace("\n", "")
            + "\t"
            + "DESCRIPTION".replace("\n", "")
            + "\n"
        )
        # display(sub_geol)
        if verbose_level != VerboseLevel.NONE:
            print(len(sub_geol), " polygons")
        # print(sub_geol)
        for i in range(0, len(sub_geol)):
            # print('**',sub_geol.loc[i][["GEOMETRY_OBJECT_ID"]],'++')
            f.write('"' + str(sub_geol.loc[i].geometry) + '"\t')
            f.write('"' + str(sub_geol.loc[i]["GEOMETRY_OBJECT_ID"]) + '"\t')
            f.write('"' + str(sub_geol.loc[i]["UNIT_NAME"]) + '"\t')
            # since map2model is looking for "" not "None"
            f.write('"' + str(sub_geol.loc[i]["GROUP"]).replace("None", "") + '"\t')
            if hint_flag == True and sub_geol.loc[i]["UNIT_NAME"] in hint_list:
                hint = code_hints.loc[sub_geol.loc[i]["UNIT_NAME"]]["hint"]
            else:
                hint = 0.0
            if str(sub_geol.loc[i]["MIN_AGE"]) == "None":
                min = 0.0
            else:
                min = sub_geol.loc[i]["MIN_AGE"].astype("float64") + float(hint)
            # print(str(sub_geol.loc[i]["MAX_AGE"]))
            if str(sub_geol.loc[i]["MAX_AGE"]) == "None":
                max = 4500000000
            else:
                max = sub_geol.loc[i]["MAX_AGE"].astype("float64") + float(hint)
            # f.write("\""+str(sub_geol.loc[i]["MIN_AGE"])+"\"\t")
            # f.write("\""+str(sub_geol.loc[i]["MAX_AGE"])+"\"\t")
            f.write('"' + str(min) + '"\t')
            f.write('"' + str(max) + '"\t')
            f.write('"' + str(sub_geol.loc[i]["CODE"]) + '"\t')
            f.write('"' + str(sub_geol.loc[i]["ROCKTYPE1"]) + '"\t')
            f.write('"' + str(sub_geol.loc[i]["ROCKTYPE2"]) + '"\t')
            f.write('"' + str(sub_geol.loc[i]["DESCRIPTION"]) + '"\n')
        f.close()

    ####################################
    # save out orientation points in WKT format
    #
    # save_structure_wkt(sub_pts,structure_file_csv,c_l)
    # sub_pts geopandas format orientation layer
    # structure_file_csv path to output WKT format file
    # c_l dictionary of codes and labels specific to input geo information layers
    #
    # Saves orientation layer as WKT format for use by map2model c++ code
    ####################################
    def save_structure_wkt(sub_pts, structure_file_csv, c_l):

        sub_pts.columns = ["WKT"] + list(sub_pts.columns[1:])
        sub_pts.to_csv(structure_file_csv, sep="\t", index=False)

    ####################################
    # save out mineral deposit points in WKT format
    #
    # save_mindep_wkt(sub_mindep,mindep_file_csv,c_l)
    # sub_mindep geopandas format mineral deposit layer
    # mindep_file_csv path to output WKT format file
    # c_l dictionary of codes and labels specific to input geo information layers
    #
    # Saves mineral deposit layer as WKT format for use by map2model c++ code
    ####################################
    def save_mindep_wkt(sub_mindep, mindep_file_csv, c_l):

        sub_mindep.columns = ["WKT"] + list(sub_mindep.columns[1:])
        sub_mindep.to_csv(mindep_file_csv, sep="\t", index=False)

        # f = open(mindep_file_csv, "w+")
        # f.write('WKT\t'+c_l['msc']+'\t'+c_l['msn']+'\t'+c_l['mst'] +
        #         '\t'+c_l['mtc']+'\t'+c_l['mscm']+'\t'+c_l['mcom']+'\n')

        # print(len(sub_mindep), " points")

        # for i in range(0, len(sub_mindep)):
        #     line = "\""+str(sub_mindep.loc[i].geometry)+"\"\t\""+str(sub_mindep.loc[i][c_l['msc']])+"\"\t\"" +\
        #         str(sub_mindep.loc[i][c_l['msn']])+"\"\t\""+str(sub_mindep.loc[i][c_l['mst']])+"\"\t\"" +\
        #         str(sub_mindep.loc[i][c_l['mtc']])+"\"\t\""+str(sub_mindep.loc[i][c_l['mscm']])+"\"\t\"" +\
        #         str(sub_mindep.loc[i][c_l['mcom']])+"\"\n"
        #     f.write(functools.reduce(operator.add, (line)))

        # f.close()

    ####################################
    # save out fault polylines in WKT format
    #
    # save_faults_wkt(sub_lines,fault_file_csv,c_l)
    # sub_geol geopandas format fault and fold axial trace layer geology_file_csv path to output WKT format file
    # c_l dictionary of codes and labels specific to input geo information layers
    #
    # Saves fault/fold axial trace layer as WKT format for use by map2model c++ code
    ####################################
    def save_faults_wkt(sub_lines, fault_file_csv, c_l):

        # Change geometry column name to WKT
        sub_lines.columns = ["WKT"] + list(sub_lines.columns[1:])
        # Filter cells with a feature description containing the word 'fault'
        mask = sub_lines["FEATURE"].str.contains(
            "fault", na=False, case=False, regex=True
        )
        sub_faults = sub_lines[mask]
        sub_faults.to_csv(fault_file_csv, sep="\t", index=False)

    def check_near_fault_contacts(
        path_faults,
        all_sorts_path,
        fault_dimensions_path,
        gp_fault_rel_path,
        contacts_path,
        c_l,
        proj_crs,
    ):
        faults_clip = gpd.read_file(path_faults)
        gp_fault_rel = pd.read_csv(gp_fault_rel_path)
        gp_fault_rel.set_index("group", inplace=True)
        contacts = pd.read_csv(contacts_path)
        all_sorts = pd.read_csv(all_sorts_path)
        all_sorts.set_index("code", inplace=True)
        fault_dimensions = pd.read_csv(fault_dimensions_path)
        fault_dimensions2 = fault_dimensions.set_index("Fault")
        groups = all_sorts["group"].unique()

        for indx, flt in faults_clip.iterrows():
            # print('Fault_'+str(flt["GEOMETRY_OBJECT_ID"]) )
            if (
                "Fault_" + str(flt["GEOMETRY_OBJECT_ID"])
                in fault_dimensions["Fault"].values
            ):
                # print('Fault_'+str(flt["GEOMETRY_OBJECT_ID"]),flt.geometry.type,flt.geometry.centroid )
                if flt.geometry.type == "LineString":
                    flt_ls = LineString(flt.geometry)
                    midx = flt_ls.coords[0][0] + (
                        (flt_ls.coords[0][0] - flt_ls.coords[len(flt_ls.coords) - 1][0])
                        / 2
                    )
                    midy = flt_ls.coords[0][1] + (
                        (flt_ls.coords[0][1] - flt_ls.coords[len(flt_ls.coords) - 1][1])
                        / 2
                    )
                    l, m = m2l_utils.pts2dircos(
                        flt_ls.coords[0][0],
                        flt_ls.coords[0][1],
                        flt_ls.coords[len(flt_ls.coords) - 1][0],
                        flt_ls.coords[len(flt_ls.coords) - 1][1],
                    )
                    angle = (360 + degrees(atan2(l, m))) % 360
                    xax = (
                        fault_dimensions2.loc[
                            "Fault_" + str(flt["GEOMETRY_OBJECT_ID"])
                        ]["HorizontalRadius"]
                        * 0.99
                        * 0.81
                    )
                    yax = (
                        fault_dimensions2.loc[
                            "Fault_" + str(flt["GEOMETRY_OBJECT_ID"])
                        ]["InfluenceDistance"]
                        * 0.99
                        * 0.81
                    )
                    circle = Point(
                        flt.geometry.centroid.x, flt.geometry.centroid.y
                    ).buffer(
                        1
                    )  # type(circle)=polygon
                    ellipse = shapely.affinity.scale(
                        circle, xax, yax
                    )  # type(ellipse)=polygon
                    ellipse = shapely.affinity.rotate(
                        ellipse, 90 - angle, origin="center", use_radians=False
                    )
                    splits = split(ellipse, flt.geometry)
                    # display(flt.geometry)
                    i = 0
                    for gp in groups:
                        if not gp == "cover":
                            all_sorts2 = all_sorts[all_sorts["group"] == gp]
                            first = True
                            for half in splits:
                                half_poly = Polygon(half)
                                half_ellipse = gpd.GeoDataFrame(
                                    index=[0], crs=proj_crs, geometry=[half_poly]
                                )
                                has_contacts = True
                                for indx, as2 in all_sorts2.iterrows():
                                    contacts2 = contacts[contacts["formation"] == indx]
                                    if first:
                                        first = False
                                        all_contacts = contacts2.copy()
                                    else:
                                        all_contacts = pd.concat(
                                            [all_contacts, contacts2], sort=False
                                        )
                                contacts_gdf = gpd.GeoDataFrame(
                                    all_contacts,
                                    geometry=[
                                        Point(x, y)
                                        for x, y in zip(all_contacts.X, all_contacts.Y)
                                    ],
                                )
                                found = gpd.sjoin(
                                    contacts_gdf,
                                    half_ellipse,
                                    how="inner",
                                    predicate="within",
                                )

                                if len(found) > 0 and has_contacts:
                                    has_contacts = True
                                else:
                                    has_contacts = False
                                i = i + 1
                            # print('Fault_'+str(flt["GEOMETRY_OBJECT_ID"]),gp,has_contacts)
                            if not has_contacts:
                                if (
                                    gp_fault_rel.loc[
                                        gp, "Fault_" + str(flt["GEOMETRY_OBJECT_ID"])
                                    ]
                                    == 1
                                ):
                                    print(
                                        gp,
                                        "Fault_" + str(flt["GEOMETRY_OBJECT_ID"]),
                                        "combination switched OFF",
                                    )
                                    gp_fault_rel.loc[
                                        gp, "Fault_" + str(flt["GEOMETRY_OBJECT_ID"])
                                    ] = 0

                elif (
                    flt.geometry.type == "MultiLineString"
                    or flt.geometry.type == "GeometryCollection"
                ):
                    raise NameError(
                        "map2loop error: Fault_"
                        + str(flt["GEOMETRY_OBJECT_ID"])
                        + "cannot be analysed as it is a multilinestring,\n  it may be a fault that is clipped into two parts by the bounding box\nfix in original shapefile??"
                    )
                    continue

        gp_fault_rel.to_csv(gp_fault_rel_path)

    @beartype.beartype
    def super_groups_and_groups(group_girdle, config: Config, map_data, workflow: dict):
        group_girdle = pd.DataFrame.from_dict(group_girdle, orient="index")
        group_girdle.columns = ["plunge", "bearing", "num orientations"]
        group_girdle.sort_values(by="num orientations", ascending=False, inplace=True)
        # print(group_girdle)

        l, m, n = m2l_utils.ddd2dircos(
            group_girdle.iloc[0]["plunge"], group_girdle.iloc[0]["bearing"]
        )
        super_group = pd.DataFrame(
            [[group_girdle[0:1].index[0], "Super_Group_0", l, m, n]],
            columns=["Group", "Super_Group", "l", "m", "n"],
        )
        super_group.set_index("Group", inplace=True)

        # geol = gpd.read_file(os.path.join(config.tmp_path, 'geol_clip.shp'))
        local_geol = map_data.get_map_data(Datatype.GEOLOGY).copy()
        local_geol.drop_duplicates(subset="UNIT_NAME", keep="first", inplace=True)
        local_geol = local_geol.set_index("GROUP")

        sg_index = 0
        for i in range(0, len(group_girdle)):
            # if(config.c_l['intrusive'] in geol.loc[group_girdle.iloc[i].name.replace("_"," ")]["ROCKTYPE1"]
            # and config.c_l['sill'] not in geol.loc[group_girdle.iloc[i].name.replace("_"," ")]["DESCRIPTION"]):
            if group_girdle.iloc[i].name in local_geol["UNIT_NAME"]:
                if (
                    config.c_l["intrusive"]
                    in local_geol.loc[group_girdle.iloc[i].name]["ROCKTYPE1"]
                    and config.c_l["sill"]
                    not in local_geol.loc[group_girdle.iloc[i].name]["DESCRIPTION"]
                ):
                    sg_index = sg_index + 1
                    sgname = "Super_Group_" + str(sg_index)
                    super_group_new = pd.DataFrame(
                        [[group_girdle[i : i + 1].index[0], sgname, l, m, n]],
                        columns=["Group", "Super_Group", "l", "m", "n"],
                    )
                    super_group_new.set_index("Group", inplace=True)
                    super_group = pd.concat([super_group, super_group_new])

                elif group_girdle.iloc[i]["num orientations"] > 5:
                    # print('>5',group_girdle.iloc[i].name)
                    l, m, n = m2l_utils.ddd2dircos(
                        group_girdle.iloc[i]["plunge"], group_girdle.iloc[i]["bearing"]
                    )

                    found = False
                    sg_i = 0
                    for ind, sg in super_group.iterrows():

                        c = sg["l"] * l + sg["m"] * m + sg["n"] * n
                        if c > 1:
                            c = 1
                        c = degrees(acos(c))
                        if c < config.run_flags["misorientation"] and not found:
                            found = True
                            sgname = "Super_Group_" + str(sg_i)
                            super_group_old = pd.DataFrame(
                                [[group_girdle[i : i + 1].index[0], sgname, l, m, n]],
                                columns=["Group", "Super_Group", "l", "m", "n"],
                            )
                            super_group_old.set_index("Group", inplace=True)
                            super_group = pd.concat([super_group, super_group_old])
                        sg_i = sg_i + 1
                    if not found:

                        sg_index = sg_index + 1
                        # print('not found',sg_index)
                        sgname = "Super_Group_" + str(sg_index)
                        super_group_new = pd.DataFrame(
                            [[group_girdle[i : i + 1].index[0], sgname, l, m, n]],
                            columns=["Group", "Super_Group", "l", "m", "n"],
                        )
                        super_group_new.set_index("Group", inplace=True)
                        super_group = pd.concat([super_group, super_group_new])
                else:  # not enough orientations to test, so lumped with group with most orientations
                    # print('<5',group_girdle.iloc[i].name)
                    sgname = "Super_Group_" + str(0)
                    super_group_old = pd.DataFrame(
                        [[group_girdle[i : i + 1].index[0], sgname, l, m, n]],
                        columns=["Group", "Super_Group", "l", "m", "n"],
                    )
                    super_group_old.set_index("Group", inplace=True)
                    super_group = pd.concat([super_group, super_group_old])
        # print('super group from stereonets',super_group)
        use_gcode3 = []
        for ind, sg in super_group.iterrows():
            clean = ind.replace(" ", "_").replace("-", "_")
            use_gcode3.append(clean)
        if workflow["cover_map"]:
            use_gcode3.append("cover")

        sg2 = set(super_group["Super_Group"])
        super_groups = []
        if workflow["cover_map"]:
            super_groups.append(["cover"])
        for s in sg2:
            temp = []
            for ind, sg in super_group.iterrows():
                if s == sg["Super_Group"]:
                    temp.append(ind)
            super_groups.append(temp)

        f = open(os.path.join(config.tmp_path, "super_groups.csv"), "w")
        for sg in super_groups:
            for s in sg:
                f.write(str(s) + ",")
            f.write("\n")
        f.close()
        return (super_groups, use_gcode3)

    def use_asud(strat_graph_file, graph_path):
        asud_strat_file = "https://gist.githubusercontent.com/yohanderose/3b257dc768fafe5aaf70e64ae55e4c42/raw/8598c7563c1eea5c0cd1080f2c418dc975cc5433/ASUD.csv"

        G = nx.read_gml(strat_graph_file, label="id")
        Gp = G.copy().to_directed()

        try:
            ASUD = pd.read_csv(asud_strat_file, sep=",")
        except Exception as e:
            print(e)
            return

        for e in G.edges:
            glabel_0 = G.nodes[e[0]]["LabelGraphics"]["text"]
            glabel_1 = G.nodes[e[1]]["LabelGraphics"]["text"]

            edge = ASUD.loc[(ASUD["over"] == glabel_0) & (ASUD["under"] == glabel_1)]
            if len(edge) > 0 and (
                not ("isGroup" in Gp.nodes[e[0]]) and not ("isGroup" in Gp.nodes[e[1]])
            ):
                if Gp.has_edge(e[1], e[0]):
                    Gp.remove_edge(e[1], e[0])
                if Gp.has_edge(e[0], e[1]):
                    Gp.remove_edge(e[0], e[1])
                Gp.add_edge(e[0], e[1])

            edge = ASUD.loc[(ASUD["under"] == glabel_0) & (ASUD["over"] == glabel_1)]
            if len(edge) > 0 and (
                not ("isGroup" in Gp.nodes[e[0]]) and not ("isGroup" in Gp.nodes[e[1]])
            ):
                if Gp.has_edge(e[0], e[1]):
                    Gp.remove_edge(e[0], e[1])
                if Gp.has_edge(e[1], e[0]):
                    Gp.remove_edge(e[1], e[0])
                Gp.add_edge(e[1], e[0])

        recode = {}
        i = 0
        for n in Gp.nodes:
            if "isGroup" in Gp.nodes[n]:
                recode[n] = i
            i = i + 1

        for n in Gp.nodes:
            if not "isGroup" in Gp.nodes[n]:
                Gp.nodes[n]["gid"] = recode[Gp.nodes[n]["gid"]]

        # check for cycle and arbitrarily remove first of the edges
        # nx.write_gml(Gp,os.path.join(graph_path, 'ASUD_strat_test.gml'))
        try:
            cycles = list(nx.simple_cycles(G))
            for c in cycles:
                G.remove_edge(c[0], c[1])
                warning_msg = (
                    'map2loop warning: The stratigraphic relationship: "'
                    + c[0]
                    + " overlies "
                    + c[1]
                    + '" was removed as it conflicts with another relationship'
                )
                warnings.warn(warning_msg)
        except:
            print("no cycles")

        nx.write_gml(Gp, os.path.join(graph_path, "ASUD_strat.gml"))

    ####################################
    # combine multiple outputs into single graph that contains all infor needed by LoopStructural
    #
    # make_Loop_graph(tmp_path,output_path)
    # tmp_path path to tmp directory
    # output_path path to output directory
    #
    # Returns networkx graph
    ####################################

    @beartype.beartype
    def make_Loop_graph(config: Config, map_data, point_data):
        Gloop = nx.DiGraph()
        strats = map_data.get_map_data(Datatype.GEOLOGY).copy()
        strats.drop_duplicates(subset=["UNIT_NAME"], inplace=True)
        strats["UNIT_NAME"] = strats["UNIT_NAME"].replace(" ", "_").replace("-", "_")
        strats.set_index(["UNIT_NAME"], inplace=True)

        # Load faults and stratigraphy
        Gf = nx.read_gml(os.path.join(config.tmp_path, "fault_network.gml"))
        Astrat = pd.read_csv(
            os.path.join(config.tmp_path, "all_sorts_clean.csv"), sep=","
        )

        # add formation stratigraphy to graph as nodes
        Astrat = Astrat.set_index("code")
        for ind, s in Astrat.iterrows():
            if s.name != "cover" and s.name != "cover_up":
                Gloop.add_node(
                    s.name,
                    s_colour=s["colour"],
                    ntype="formation",
                    group=s["group"],
                    StratType=s["strat_type"],
                    uctype=s["uctype"],
                    GroupNumber=s["group number"],
                    IndexInGroup=s["index in group"],
                    NumberInGroup=s["number in group"],
                    MinAge=strats.loc[s.name]["MIN_AGE"],
                    MaxAge=strats.loc[s.name]["MAX_AGE"],
                )
            else:
                Gloop.add_node(
                    s.name,
                    s_colour=s["colour"],
                    ntype="formation",
                    group=s["group"],
                    StratType=s["strat_type"],
                    uctype=s["uctype"],
                    GroupNumber=s["group number"],
                    IndexInGroup=s["index in group"],
                    NumberInGroup=s["number in group"],
                    MinAge=0,
                    MaxAge=1,
                )

        # add formation-formation stratigraphy to graph as edges
        i = 0
        for ind, s in Astrat.iterrows():
            if ind != Astrat.index[-1]:
                Gloop.add_edge(Astrat.iloc[i].name, Astrat.iloc[i + 1].name)
                Gloop[Astrat.iloc[i].name][Astrat.iloc[i + 1].name][
                    "etype"
                ] = "formation_formation"
            i = i + 1

        # add faults to graph as nodes
        Af_d = pd.read_csv(
            os.path.join(config.output_path, "fault_dimensions.csv"), sep=","
        )
        for ind, f in Af_d.iterrows():
            Gloop.add_node(f["Fault"], ntype="fault")

        # add fault centroid to node
        Afgeom = pd.read_csv(os.path.join(config.output_path, "faults.csv"), sep=",")

        pd.to_numeric(Afgeom["X"], downcast="float")
        pd.to_numeric(Afgeom["Y"], downcast="float")
        pd.to_numeric(Afgeom["Z"], downcast="float")

        for n in Gloop.nodes:
            if "Fault" in n:
                subset = Afgeom[Afgeom["formation"] == n]
                Gloop.nodes[n]["Xmean"] = subset["X"].mean()
                Gloop.nodes[n]["Ymean"] = subset["Y"].mean()
                Gloop.nodes[n]["Zmean"] = subset["Z"].mean()

        for e in Gf.edges:
            Gloop.add_edge(e[0], e[1])
            Gloop[e[0]][e[1]]["Angle"] = Gf.edges[e]["angle"]
            Gloop[e[0]][e[1]]["Topol"] = Gf.edges[e]["topol"]
            Gloop[e[0]][e[1]]["etype"] = "fault_fault"

        # add fault dimension info to fault nodes
        Af_d = pd.read_csv(
            os.path.join(config.output_path, "fault_dimensions.csv"), sep=","
        )
        fault_l = pd.DataFrame(Af_d["Fault"])
        Af_d = Af_d.set_index("Fault")
        fault_l = fault_l.set_index("Fault")
        fault_l["incLength"] = Af_d["incLength"]

        for n in Gloop.nodes:
            if "Fault" in n:
                if Af_d.loc[n].name in Gloop.nodes:
                    Gloop.nodes[n]["HorizontalRadius"] = Af_d.loc[n]["HorizontalRadius"]
                    Gloop.nodes[n]["VerticalRadius"] = Af_d.loc[n]["VerticalRadius"]
                    Gloop.nodes[n]["InfluenceDistance"] = Af_d.loc[n][
                        "InfluenceDistance"
                    ]
                    Gloop.nodes[n]["IncLength"] = Af_d.loc[n]["incLength"]
                    Gloop.nodes[n]["f_colour"] = Af_d.loc[n]["colour"]

        # add fault orientation info and clustering on orientation and length to fault nodes
        Af_o = pd.read_csv(
            os.path.join(config.output_path, "fault_orientations.csv"), sep=","
        )
        Af_o = Af_o.drop_duplicates(subset="formation")
        Af_o = Af_o.set_index("formation")

        pd.to_numeric(Af_o["dip"], downcast="float")
        pd.to_numeric(Af_o["DipDirection"], downcast="float")
        from sklearn import cluster

        def convert_dd_lmn(dip, dipdir):
            r_dd = np.radians(dipdir)
            r_90_dip = np.radians(90 - dip)
            sin_dd = np.sin(r_dd)
            cos_dd = np.cos(r_dd)

            cos_90_dip = np.cos(r_90_dip)
            sin_90_dip = np.sin(r_90_dip)

            l = sin_dd * cos_90_dip
            m = cos_dd * cos_90_dip
            n = sin_90_dip
            return (l, m, n)

        l, m, n = convert_dd_lmn(Af_o["dip"], Af_o["DipDirection"])
        faults = pd.DataFrame(l, columns=["l"])
        faults["m"] = m
        faults["n"] = n

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(Af_d) >= config.run_flags["fault_orientation_clusters"]:
                k_means_o = cluster.KMeans(
                    n_clusters=config.run_flags["fault_orientation_clusters"],
                    max_iter=50,
                    random_state=1,
                )
                k_means_o.fit(faults)
                labels_o = k_means_o.labels_
                clusters_o = pd.DataFrame(labels_o, columns=["cluster_o"])

            if len(Af_d) >= config.run_flags["fault_length_clusters"]:
                k_means_l = cluster.KMeans(
                    n_clusters=config.run_flags["fault_length_clusters"],
                    max_iter=50,
                    random_state=1,
                )
                k_means_l.fit(fault_l)
                labels_l = k_means_l.labels_
                clusters_l = pd.DataFrame(labels_l, columns=["cluster_l"])

        Af_o = Af_o.reset_index()
        if len(Af_d) >= config.run_flags["fault_orientation_clusters"]:
            Af_o["cluster_o"] = clusters_o["cluster_o"]
        if len(Af_d) >= config.run_flags["fault_length_clusters"]:
            Af_o["cluster_l"] = clusters_l["cluster_l"]
        Af_o = Af_o.set_index(keys="formation")
        Af_o.to_csv(os.path.join(config.output_path, "fault_clusters.csv"))
        for n in Gloop.nodes:
            if "Fault" in n:
                if Af_o.loc[n].name in Gloop.nodes:
                    Gloop.nodes[n]["Dip"] = Af_o.loc[n]["dip"]
                    Gloop.nodes[n]["DipDirection"] = Af_o.loc[n]["DipDirection"]
                    Gloop.nodes[n]["DipPolarity"] = Af_o.loc[n]["DipPolarity"]
                    if len(Af_d) >= config.run_flags["fault_orientation_clusters"]:
                        Gloop.nodes[n]["OrientationCluster"] = Af_o.loc[n]["cluster_o"]
                    else:
                        Gloop.nodes[n]["OrientationCluster"] = -1
                    if len(Af_d) >= config.run_flags["fault_length_clusters"]:
                        Gloop.nodes[n]["LengthCluster"] = Af_o.loc[n]["cluster_l"]
                    else:
                        Gloop.nodes[n]["LengthCluster"] = -1

        # add centrality measures to fault nodes

        Gfcc = nx.closeness_centrality(Gf)
        Gfbc = nx.betweenness_centrality(Gf)

        for n in Gloop.nodes:
            if "Fault" in n:
                if n in Gfcc.keys():
                    Gloop.nodes[n]["ClosenessCentrality"] = Gfcc[n]
                    Gloop.nodes[n]["BetweennessCentrality"] = Gfbc[n]
                else:
                    Gloop.nodes[n]["ClosenessCentrality"] = -1
                    Gloop.nodes[n]["BetweennessCentrality"] = -1

        # add formation thickness info to formation nodes
        As_t = pd.read_csv(
            os.path.join(config.output_path, "formation_summary_thicknesses.csv"),
            sep=",",
        )
        As_t = As_t.set_index("formation")

        for n in Gloop.nodes:
            if not "Fault" in n and not "Point_data" in n:
                if As_t.loc[n].name in Gloop.nodes:
                    if np.isnan(As_t.loc[n]["thickness std"]):
                        Gloop.nodes[n]["ThicknessStd"] = -1
                    else:
                        Gloop.nodes[n]["ThicknessStd"] = As_t.loc[n]["thickness std"]
                    if np.isnan(As_t.loc[n]["thickness median"]):
                        Gloop.nodes[n]["ThicknessMedian"] = -1
                    else:
                        Gloop.nodes[n]["ThicknessMedian"] = As_t.loc[n][
                            "thickness median"
                        ]

                    Gloop.nodes[n]["ThicknessMethod"] = As_t.loc[n]["method"]

        # add group-formation stratigraphy to graph as edges
        for ind, s in Astrat.iterrows():
            if not s["group"] + "_gp" in Gloop.nodes():
                Gloop.add_node(s["group"] + "_gp", ntype="group")
            Gloop.add_edge(s["group"] + "_gp", s.name)
            Gloop[s["group"] + "_gp"][s.name]["etype"] = "group_formation"

        # add group-fault edges to graph
        Af_s = pd.read_csv(
            os.path.join(config.output_path, "group-fault-relationships.csv"), sep=","
        )
        for ind, s in Af_s.iterrows():
            if s["group"] + "_gp" in Gloop.nodes:
                for col in Af_s.columns[1:]:
                    if col in Gloop.nodes:
                        if Af_s.loc[ind, col] == 1:
                            Gloop.add_edge(col, s["group"] + "_gp")
                            Gloop[col][s["group"] + "_gp"]["etype"] = "fault_group"

        # add group-group edges to graph
        Ag_g = pd.read_csv(
            os.path.join(config.tmp_path, "groups_clean.csv"),
            header=None,
            index_col=None,
        )
        i = 0
        for ind, g in Ag_g.iterrows():
            if ind != Ag_g.index[-1]:
                if (
                    Ag_g.iloc[i][0] + "_gp" in Gloop.nodes
                    and Ag_g.iloc[i + 1][0] + "_gp" in Gloop.nodes
                ):
                    Gloop.add_edge(Ag_g.iloc[i][0] + "_gp", Ag_g.iloc[i + 1][0] + "_gp")
                    Gloop[Ag_g.iloc[i][0] + "_gp"][Ag_g.iloc[i + 1][0] + "_gp"][
                        "etype"
                    ] = "group_group"
            i = i + 1

        # add supergroups as nodes and supergroup-group relationships as edges
        sgi = 0
        supergroups = {}
        with open(os.path.join(config.tmp_path, "super_groups.csv")) as sgf:
            lines = sgf.readlines()
            for l in lines:
                Gloop.add_node("supergroup_{}".format(sgi), ntype="supergroup")
                for g in l.split(","):
                    g = g.replace("-", "_").replace(" ", "_").rstrip()
                    if g:
                        supergroups[g] = "supergroup_{}".format(sgi)
                        # print(g+'_gp')
                        if g + "_gp" in Gloop.nodes():
                            Gloop.add_edge(supergroups[g], g + "_gp")
                            Gloop[supergroups[g]][g + "_gp"][
                                "etype"
                            ] = "supergroup_group"
                sgi += 1

        # add all geolocated data to a single unconnected node
        Gloop.add_node("Point_data", ntype="points", data=point_data)

        with map_data.get_map_data(Datatype.DTM).open() as dtm:
            dtm_data = dtm.read(1)
            bounds = dtm.bounds
            minx = bounds.left
            miny = bounds.bottom
            maxx = bounds.right
            maxy = bounds.top
            xscale = (maxx - minx) / dtm_data.shape[1]
            yscale = (maxy - miny) / dtm_data.shape[0]
            Gloop.add_node(
                "DTM_data",
                ntype="dtm",
                data=str(dtm_data.tolist()),
                shape=str(dtm_data.shape),
                minx=minx,
                miny=miny,
                maxx=maxx,
                maxy=maxy,
                xscale=xscale,
                yscale=yscale,
            )

        Gloop.add_node("bbox", ntype="bbox", data=str(config.bbox))

        Gloop.add_node(
            "dst_crs", ntype="dst_crs", data=str(map_data.working_projection)
        )

        config_metadata = {
            "project_path": config.project_path,
            "geology_file": map_data.get_filename(Datatype.GEOLOGY),
            "fault_file": map_data.get_filename(Datatype.FAULT),
            "fold_file": map_data.get_filename(Datatype.FOLD),
            "structure_file": map_data.get_filename(Datatype.STRUCTURE),
            "mindep_file": map_data.get_filename(Datatype.MINERAL_DEPOSIT),
            "section_files": map_data.get_filename(Datatype.SECTION),
            "drillhole_file": map_data.get_filename(Datatype.DRILLHOLE),
            "dtb_grid_file": map_data.get_filename(Datatype.DTB_GRID),
            "cover_map_file": map_data.get_filename(Datatype.COVER_MAP),
            "bbox_3d": config.bbox_3d,
            "step_out": config.step_out,
            "dtm_crs": config.dtm_crs,
            "project_crs": map_data.working_projection,
        }
        Gloop.add_node(
            "metadata",
            ntype="metadata",
            run_flags=str(config.run_flags),
            c_l=str(config.c_l),
            config=str(config_metadata),
        )
        return Gloop

    def colour_Loop_graph(output_path, prefix):
        with open(os.path.join(output_path, prefix + ".gml")) as graph:
            new_graph = open(os.path.join(output_path, prefix + "_colour.gml"), "w")
            lines = graph.readlines()
            for l in lines:
                if "s_colour" in l:
                    new_graph.write(
                        "    "
                        + l.strip().replace("s_colour", "graphics [ fill ")
                        + " ]\n"
                    )
                elif "f_colour" in l:
                    new_graph.write(
                        "    "
                        + l.strip().replace(
                            "f_colour", 'graphics [ type "ellipse" fill '
                        )
                        + " ]\n"
                    )
                elif 'ntype "supergroup"' in l:
                    new_graph.write('    graphics [ type "triangle" fill "#FF0000" ]\n')
                    new_graph.write(l)
                elif 'ntype "group"' in l:
                    new_graph.write('    graphics [ type "triangle" fill "#FF9900" ]\n')
                    new_graph.write(l)
                elif (
                    'ntype "points"' in l
                    or 'ntype "metadata"' in l
                    or 'ntype "dtm"' in l
                    or 'ntype "bbox"' in l
                    or 'ntype "dst_crs"' in l
                ):
                    new_graph.write('    graphics [ type "octagon" fill "#00FF00" ]\n')
                    new_graph.write(l)
                elif 'etype "formation_formation"' in l:
                    new_graph.write(
                        '    graphics [ style "line" arrow "last" fill "#6666FF" ]\n'
                    )
                    new_graph.write(l)
                elif 'etype "fault_group"' in l:
                    new_graph.write(
                        '    graphics [ style "line" arrow "last" fill "#666600" ]\n'
                    )
                    new_graph.write(l)
                elif 'etype "fault_formation"' in l:
                    new_graph.write(
                        '    graphics [ style "line" arrow "last" fill "#0066FF" ]\n'
                    )
                    new_graph.write(l)
                elif 'etype "fault_fault"' in l:
                    new_graph.write(
                        '    graphics [ style "line" arrow "last" fill "#000000" ]\n'
                    )
                    new_graph.write(l)
                elif 'etype "group_formation"' in l:
                    new_graph.write(
                        '    graphics [ style "line" arrow "last" fill "#FF6600" ]\n'
                    )
                    new_graph.write(l)
                else:
                    new_graph.write(l)
            new_graph.close()

    ####################################
    # combine multiple outputs into single graph that describes map and its relationship to deposits
    #
    # make_complete_Loop_graph(tmp_path,output_path)
    # tmp_path path to tmp directory
    # output_path path to output directory
    #
    # Returns networkx graph
    ####################################

    def make_complete_Loop_graph(
        Gloop,
        tmp_path,
        output_path,
        fault_orientation_clusters,
        fault_length_clusters,
        point_data,
        dtm_file,
        dst_crs,
        c_l,
        run_flags,
        config,
        bbox,
    ):

        # add group-fault edges to graph
        Af_s = pd.read_csv(
            os.path.join(output_path, "unit-fault-relationships.csv"), sep=","
        )
        for ind, s in Af_s.iterrows():
            if s["code"] in Gloop.nodes:
                for col in Af_s.columns[1:]:
                    if col in Gloop.nodes:
                        if Af_s.loc[ind, col] == 1:
                            Gloop.add_edge(col, s["code"])
                            Gloop[col][s["code"]]["etype"] = "fault_formation"

        return Gloop

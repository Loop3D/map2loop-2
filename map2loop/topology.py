import os
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import functools
import operator
import shapely.affinity
from shapely.ops import split
from shapely.geometry import Point, LineString, MultiLineString, GeometryCollection, Polygon
from math import degrees, atan2, acos, degrees
import warnings

from map2loop import m2l_utils
from map2loop.m2l_utils import display, print


class Topology(object):

    def __init__(self):
        pass

    def save_parfile(self, c_l, output_path, geology_file_csv, fault_file_csv, structure_file_csv, mindep_file_csv, minx, maxx, miny, maxy, deposit_dist, commodities):
        with open('Parfile', 'w') as f:
            f.write(
                '--- COLUMN NAMES IN CSV DATA FILES: -------------------------------------------------------------\n')
            f.write('OBJECT COORDINATES              =WKT\n')
            f.write('FAULT: ID                       ='+c_l['o']+'\n')
            f.write('FAULT: FEATURE                  ='+c_l['f']+'\n')
            f.write('POLYGON: ID                     ='+c_l['o']+'\n')
            f.write('POLYGON: LEVEL1 NAME            ='+c_l['u']+'\n')
            f.write('POLYGON: LEVEL2 NAME            ='+c_l['g']+'\n')
            f.write('POLYGON: MIN AGE                ='+c_l['min']+'\n')
            f.write('POLYGON: MAX AGE                ='+c_l['max']+'\n')
            f.write('POLYGON: CODE                   ='+c_l['c']+'\n')
            f.write('POLYGON: DESCRIPTION            ='+c_l['ds']+'\n')
            f.write('POLYGON: ROCKTYPE1              ='+c_l['r1']+'\n')
            f.write('POLYGON: ROCKTYPE2              ='+c_l['r2']+'\n')
            f.write('DEPOSIT: SITE CODE              ='+c_l['msc']+'\n')
            f.write('DEPOSIT: SITE TYPE              ='+c_l['mst']+'\n')
            f.write('DEPOSIT: SITE COMMODITY         ='+c_l['mscm']+'\n')

            f.write(
                '--- SOME CONSTANTS: ----------------------------------------------------------------------------\n')
            f.write('FAULT AXIAL FEATURE NAME        ='+c_l['fold']+'\n')
            f.write('SILL UNIT DESCRIPTION CONTAINS  ='+c_l['sill']+'\n')
            f.write(
                'IGNEOUS ROCKTYPE CONTAINS                           ='+c_l['intrusive']+'\n')
            f.write(
                'VOLCANIC ROCKTYPE CONTAINS                          ='+c_l['volcanic']+'\n')
            f.write(
                'IGNORE DEPOSITS WITH SITE TYPE                      =Infrastructure\n')
            f.write('Intersect Contact With Fault: angle epsilon (deg)   =1.0\n')
            f.write('Intersect Contact With Fault: distance epsilon (m)  =15.0\n')
            f.write('Distance buffer (fault stops on another fault) (m)  =20.0\n')
            f.write(
                'Distance buffer (point on contact) (m)              ='+str(deposit_dist)+'\n')
            f.write('Intersect polygons distance buffer (for bad maps)   =3.\n')
            f.write(
                '------------------------------------------------------------------------------------------------\n')
            f.write(
                'Path to the output data folder                      ='+output_path+'\n')
            f.write(
                'Path to geology data file                           ='+geology_file_csv+'\n')
            f.write(
                'Path to faults data file                            ='+fault_file_csv+'\n')
            f.write(
                'Path to mineral deposits data file                  ='+mindep_file_csv+'\n')
            f.write(
                '------------------------------------------------------------------------------------------------\n')
            f.write('Clipping window X1 Y1 X2 Y2 (zeros for infinite)    =' +
                    str(minx)+' '+str(miny)+' '+str(maxx)+' '+str(maxy)+'\n')
            f.write('Min length fraction for strat/fault graphs          =0.0\n')
            f.write(
                'Graph edge width categories (three doubles)         =2000. 20000. 200000.\n')
            f.write(
                'Graph edge direction (0-min age, 1-max age, 2-avg)  ='+str(2)+'\n')
            f.write(
                'Deposit names for adding info on the graph          ='+commodities+'\n')
            f.write('Partial graph polygon ID                            =32\n')
            f.write('Partial graph depth                                 =4\n')
            f.write(
                'Map subregion size dx, dy [m] (zeros for full map)  =0. 0.\n')
            f.write(
                '------------------------------------------------------------------------------------------------\n')

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
        groups = 0
        nlist = list(G.nodes)
        for n in nlist:  # Find out total number of groups and their names groups
            if('isGroup' in G.nodes[n]):
                groups += 1
                glabels[n] = G.nodes[n]['LabelGraphics']['text'].replace(
                    " ", "_").replace("-", "_")
        return(groups, glabels, G)

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
    def save_units(G, path_out, glabels, Australia, asud_strat_file, quiet):
        if Australia:
            ASUD = pd.read_csv(asud_strat_file, ',')
        for p in glabels:  # process each group, removing nodes that are not part of that group, and other groups
            GD = G.copy()  # temporary copy of full graph
            # print()
            #print(p,glabels[p].replace(" ","_").replace("-","_"),"----------------------")
            nlist = list(G.nodes)

            for n in nlist:  # Calculate total number of groups and their names groups
                if('gid' in GD.nodes[n]):  # normal node
                    if(GD.nodes[n]['gid'] != p):  # normal node but not part of current group
                        GD.remove_node(n)
                else:  # group node
                    GD.remove_node(n)

            labels = {}
            for node in GD.nodes():  # local store of node labels
                labels[node] = G.nodes[node]['LabelGraphics']['text'].replace(
                    " ", "_").replace("-", "_")

            cycles = nx.simple_cycles(GD)
            for cy in cycles:
                found = False
                if Australia:
                    len_cy = len(cy)
                    for i in range(len_cy-1):
                        glabel_0 = GD.nodes[cy[i]]['LabelGraphics']['text']
                        glabel_1 = GD.nodes[cy[i+1]]['LabelGraphics']['text']
                        edge = ASUD.loc[(ASUD['over'] == glabel_0) & (
                            ASUD['under'] == glabel_1)]
                        if(len(edge) == 0 and (not('isGroup' in GD.nodes[cy[i]]) and not('isGroup' in GD.nodes[cy[i+1]]))):
                            if(not GD.has_edge(cy[i], cy[i+1])):
                                continue
                            else:
                                warning_msg = 'map2loop warning 1: Stratigraphic relationship: ' + \
                                    str(GD.nodes[cy[i]]['LabelGraphics']['text'])+' overlies '+str(
                                        GD.nodes[cy[i+1]]['LabelGraphics']['text'])+' removed to prevent cycle'
                                warnings.warn(warning_msg)
                                GD.remove_edge(cy[i], cy[i+1])
                                found = True

                    if(not found):
                        glabel_0 = GD.nodes[cy[len_cy-1]
                                            ]['LabelGraphics']['text']
                        glabel_1 = GD.nodes[cy[0]]['LabelGraphics']['text']
                        edge = ASUD.loc[(ASUD['over'] == glabel_0) & (
                            ASUD['under'] == glabel_1)]
                        if(len(edge) == 0 and (not('isGroup' in GD.nodes[cy[len_cy-1]]) and not('isGroup' in GD.nodes[cy[0]]))):
                            if(GD.has_edge(cy[len_cy-1], cy[0])):
                                warning_msg = 'map2loop warning 1: Stratigraphic relationship: ' + \
                                    str(GD.nodes[cy[len_cy-1]]['LabelGraphics']['text'])+' overlies '+str(
                                        GD.nodes[cy[0]]['LabelGraphics']['text'])+' removed to prevent cycle'
                                warnings.warn(warning_msg)
                                GD.remove_edge(cy[len_cy-1], cy[0])
                                found = True
                        if(not found):
                            warning_msg = 'map2loop warning 2: Stratigraphic relationship: ' + \
                                str(GD.nodes[cy[0]]['LabelGraphics']['text'])+' overlies '+str(
                                    GD.nodes[cy[1]]['LabelGraphics']['text'])+' removed to prevent cycle'
                            warnings.warn(warning_msg)
                            if( GD.has_edge(cy[len_cy-1],cy[0])):
                                if GD.has_edge(cy[0],cy[1]):
                                    GD.remove_edge(cy[0], cy[1])

                else:
                    warning_msg = 'map2loop warning 3: Stratigraphic relationship: ' + \
                        str(GD.nodes[cy[0]]['LabelGraphics']['text'])+' overlies '+str(
                            GD.nodes[cy[1]]['LabelGraphics']['text'])+' removed to prevent cycle'
                    warnings.warn(warning_msg)
                    if GD.has_edge(cy[0],cy[1]):
                        GD.remove_edge(cy[0], cy[1])

            if not quiet:
                plt.figure(p+1)  # display strat graph for one group
                plt.title(glabels[p])
                plt.tight_layout()
                nx.draw_networkx(GD, pos=nx.kamada_kawai_layout(
                    GD), arrows=True, with_labels=False)
                nx.draw_networkx_labels(GD, pos=nx.kamada_kawai_layout(
                    GD), labels=labels, font_size=12, font_family='sans-serif')

            one = True
            if(one):
                # one possible sorted directional graphs
                nlist = list(nx.topological_sort(GD))
            else:
                # all possible sorted directional graphs
                nlist = list(nx.all_topological_sorts(GD))

            f = open(
                os.path.join(path_out, glabels[p].replace(" ", "_").replace("-", "_")+'.csv'), 'w')

            if(one):
                f.write('Choice '+str(0))
                for n in range(0, len(GD)):  # display nodes for one sorted graph

                    f.write(","+G.nodes[nlist[n]]['LabelGraphics']
                            ['text'].replace(" ", "_").replace("-", "_"))

                f.write('\n')
            else:
                for m in range(10):  # process first ten sorted graphs
                    f.write('Choice '+str(m))
                    for n in range(0, len(GD)):  # display nodes for one sorted graph
                        #print(nlist[m][n],G.nodes[nlist[m][n]]['LabelGraphics']['text'].replace(" ","_").replace("-","_"))
                        f.write(","+G.nodes[nlist[m][n]]['LabelGraphics']
                                ['text'].replace(" ", "_").replace("-", "_"))
                    # if(m<len(nlist)-1):
                        # print("....")
                    f.write('\n')
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
            if(str(a_poly[c_l['g']]) == 'None'):
                grp = a_poly[c_l['c']].replace(" ", "_").replace("-", "_")
            else:
                grp = a_poly[c_l['g']].replace(" ", "_").replace("-", "_")
            # print(grp)
            if(not grp in groups):
                groups += [(grp)]

            info += [(grp, a_poly[c_l['min']], a_poly[c_l['max']])]

        # display(info)
        # display(groups)
        for j in range(0, len(groups)):
            # print(groups[j],'------------------')
            min_age = 1e10
            max_age = 0
            for i in range(0, len(info)):
                if(info[i][0] == groups[j]):
                    if(float(info[i][1]) < min_age):
                        min_age = float(info[i][1])
                        min_ind = i
                    if(float(info[i][2]) > max_age):
                        max_age = float(info[i][2])
                        max_ind = i
    #        print(groups[j],min_age,max_age,(max_age+min_age)/2)
            ages += [(groups[j], min_age, max_age, (max_age+min_age)/2)]
    #    print()
    #    for j in range(0,len(ages)):
    #        print(ages[j][0],ages[j][1],ages[j][2],ages[j][3],sep='\t')
    #    print()

        slist = sorted(ages, key=lambda l: l[3])
        f = open(os.path.join(tmp_path, 'age_sorted_groups.csv'), 'w')
        f.write('index,group_,min,max,ave\n')
        for j in range(0, len(slist)):
            f.write(str(j)+','+slist[j][0]+','+str(slist[j][1]) +
                    ','+str(slist[j][2])+','+str(slist[j][3])+'\n')
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

    def save_group(self, G, path_out, glabels, geol, c_l, quiet):
        Gp = nx.Graph().to_directed()  # New Group graph

        geology_file = gpd.read_file(os.path.join(path_out, 'geol_clip.shp'))

        self.abs_age_groups(geol, path_out, c_l)
        geology_file.drop_duplicates(subset=c_l['c'],  inplace=True)

        geology_file.set_index(c_l['c'],  inplace=True)
        # display(geology_file)

        gp_ages = pd.read_csv(os.path.join(path_out, 'age_sorted_groups.csv'))
        # display(gp_ages)
        gp_ages.set_index('group_',  inplace=True)

        if not quiet:
            display(gp_ages)
        gp_ids = []
        nlist = list(G.nodes)
        for n in nlist:  # Find out total number of groups and their names groups
            if('isGroup' in G.nodes[n]):
                Gp.add_nodes_from([n])

        if not quiet:
            display(gp_ids)
        for e in G.edges:
            if(G.nodes[e[0]]['gid'] != G.nodes[e[1]]['gid']):
                glabel_0 = G.nodes[e[0]]['LabelGraphics']['text']
                glabel_1 = G.nodes[e[1]]['LabelGraphics']['text']
                # print(glabel_0, glabel_1)
                # print(df[df.CODE == "A-FOh-xs-f"])
                # exit()
                if(str(geology_file.loc[glabel_0][c_l['g']]) == 'None'):
                    grp0 = glabel_0.replace(" ", "_").replace("-", "_")
                else:
                    grp0 = geology_file.loc[glabel_0][c_l['g']].replace(
                        " ", "_").replace("-", "_")
                if(str(geology_file.loc[glabel_1][c_l['g']]) == 'None'):
                    grp1 = glabel_1.replace(" ", "_").replace("-", "_")
                else:
                    grp1 = geology_file.loc[glabel_1][c_l['g']].replace(
                        " ", "_").replace("-", "_")

                # print(glabel_0,glabel_1,gp_ages.loc[grp0],gp_ages.loc[grp1])
                if(grp0 in glabels.values() and grp1 in glabels.values()):
                    if(gp_ages.loc[grp0]['ave'] < gp_ages.loc[grp1]['ave']):
                        Gp.add_edge(G.nodes[e[0]]['gid'], G.nodes[e[1]]['gid'])

        GpD = Gp.copy()  # temporary copy of full graph
        GpD2 = Gp.copy()  # temporary copy of full graph

        for e in GpD2.edges:  # remove duplicate edges with opposite directions
            for f in GpD.edges:
                # arbitrary choice to ensure edge is not completely removed
                if(e[0] == f[1] and e[1] == f[0] and e[0] < f[0]):
                    Gp.remove_edge(e[0], e[1])
        if not quiet:
            display(glabels)
            plt.figure(1)  # display strat graph for one group
            plt.title("groups")
            if(len(glabels) > 1):
                nx.draw_networkx(Gp, pos=nx.kamada_kawai_layout(
                    Gp), arrows=True, with_labels=False)
                nx.draw_networkx_labels(Gp, pos=nx.kamada_kawai_layout(
                    Gp),  font_size=12, font_family='sans-serif')

        # when lots of groups very large number of possibilities so short cut to first choice.
        if(len(gp_ages) > 10):
            # all possible sorted directional graphs
            glist = list(nx.topological_sort(Gp))
            print("group choices: 1 (more than 10 groups)")

            f = open(os.path.join(path_out, 'groups.csv'), 'w')
            glen = len(glist)
            f.write('Choice 0')
            for n in range(0, glen):
                f.write(','+str(glabels[glist[n]]))  # check underscore
            f.write('\n')
            f.close()
        else:
            # all possible sorted directional graphs
            glist = list(nx.all_topological_sorts(Gp))
            print("group choices:", len(glist))

            f = open(os.path.join(path_out, 'groups.csv'), 'w')
            glen = len(glist)
            if(glen > 100):
                glen = 100
            for n in range(0, glen):
                f.write('Choice '+str(n))
                for m in range(0, len(glist[0])):
                    f.write(','+str(glabels[glist[n][m]]))  # check underscore
                f.write('\n')
            f.close()

        # display(glist)
        nx.write_gml(Gp, os.path.join(path_out, 'groups.gml'))
        # plt.show()

        # g=open(os.path.join(path_out,'groups.csv'),"r")
        #contents =g.readlines()
        # g.close
        #hdr=contents[0].split(" ")
        contents = np.genfromtxt(
            os.path.join(path_out, 'groups.csv'), delimiter=',', dtype='U100')

        # display('lencon',len(contents[0]))
        k = 0
        ag = open(os.path.join(path_out, 'all_sorts.csv'), "w")
        ag.write("index,group number,index in group,number in group,code,group\n")
        if(len(contents.shape) == 1):
            for i in range(1, len(contents)):
                ucontents = np.genfromtxt(os.path.join(path_out, contents[i].replace(
                    "\n", "").replace(" ", "_")+".csv"), delimiter=',', dtype='U100')
                # f=open(os.path.join(path_out, contents[i].replace("\n","").replace(" ","_")+".csv"),"r")#check underscore
                #ucontents =f.readlines()
                # f.close
                # print(len(ucontents.shape),ucontents)
                if(len(ucontents.shape) == 1):
                    for j in range(1, len(ucontents)):
                        ag.write(str(k)+","+str(i)+","+str(j)+","+str(len(ucontents)-1)+","+ucontents[j].replace(
                            "\n", "")+","+contents[i].replace("\n", "").replace(" ", "_").replace("-", "_")+"\n")
                        k = k+1
                else:
                    for j in range(1, len(ucontents[0])):
                        ag.write(str(k)+","+str(i)+","+str(j)+","+str(len(ucontents[0])-1)+","+ucontents[0][j].replace(
                            "\n", "")+","+contents[i].replace("\n", "").replace(" ", "_").replace("-", "_")+"\n")
                        k = k+1
        else:
            for i in range(1, len(contents[0])):
                ucontents = np.genfromtxt(os.path.join(path_out, contents[0][i].replace(
                    "\n", "").replace(" ", "_")+".csv"), delimiter=',', dtype='U100')
                # f=open(os.path.join(path_out,contents[i].replace("\n","").replace(" ","_")+".csv"),"r")#check underscore
                #ucontents =f.readlines()
                # f.close
                # print(len(ucontents.shape),ucontents)
                if(len(ucontents.shape) == 1):
                    for j in range(1, len(ucontents)):
                        ag.write(str(k)+","+str(i)+","+str(j)+","+str(len(ucontents)-1)+","+ucontents[j].replace(
                            "\n", "")+","+contents[0][i].replace("\n", "").replace(" ", "_").replace("-", "_")+"\n")
                        k = k+1
                else:
                    for j in range(1, len(ucontents[0])):
                        ag.write(str(k)+","+str(i)+","+str(j)+","+str(len(ucontents[0])-1)+","+ucontents[0][j].replace(
                            "\n", "")+","+contents[0][i].replace("\n", "").replace(" ", "_").replace("-", "_")+"\n")
                        k = k+1
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
    def parse_fault_relationships(graph_path, tmp_path, output_path, quiet):
        uf = open(os.path.join(graph_path, 'unit-fault-intersection.txt'), 'r')
        contents = uf.readlines()
        uf.close()

        all_long_faults = np.genfromtxt(
            os.path.join(output_path, 'fault_dimensions.csv'), delimiter=',', dtype='U100')
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

        #display('Long Faults',unique_list)

        uf = open(os.path.join(output_path, 'unit-fault-relationships.csv'), 'w')
        uf.write('code,'+str(unique_list).replace("[", "Fault_").replace(
            ",", ",Fault_").replace("'", "").replace("]", "").replace(" ", "")+'\n')
        for row in contents:
            row = row.replace("\n", "").split("{")
            unit = row[0].split(',')
            faults = row[1].replace("}", "").replace(" ", "").split(",")
            ostr = str(unit[1]).strip().replace(" ", "_").replace("-", "_")
            for ul in unique_list:
                out = [item for item in faults if ul == item]
                if(len(out) > 0):
                    ostr = ostr+",1"
                else:
                    ostr = ostr+",0"
            uf.write(ostr+"\n")
        uf.close()

        summary = pd.read_csv(os.path.join(tmp_path, 'all_sorts_clean.csv'))
        summary.set_index("code", inplace=True)
        # display('summary',summary)
        uf_rel = pd.read_csv(os.path.join(
            output_path, 'unit-fault-relationships.csv'))

        groups = summary.group.unique()
        ngroups = len(summary.group.unique())
        # print(ngroups,'groups',groups,groups[0])
        uf_array = uf_rel.to_numpy()
        gf_array = np.zeros((ngroups, uf_array.shape[1]), dtype='U100')

        for i in range(0, ngroups):
            for j in range(0, len(uf_rel)):
                if(uf_rel.iloc[j][0] in summary.index.values):
                    gsummary = summary.loc[uf_rel.iloc[j][0]]
                    if(groups[i].replace("\n", "") == gsummary['group']):
                        for k in range(1, len(uf_rel.iloc[j])):
                            if(uf_rel.iloc[j][k] == 1):
                                gf_array[i-1, k] = '1'
                            else:
                                continue
                    else:
                        continue
                else:
                    continue

        ug = open(os.path.join(output_path, 'group-fault-relationships.csv'), 'w')
        ug.write('group')
        for k in range(1, len(uf_rel.iloc[0])):
            ug.write(','+uf_rel.columns[k])
        ug.write("\n")
        for i in range(0, ngroups):
            ug.write(groups[i].replace("\n", ""))
            for k in range(1, len(uf_rel.iloc[0])):
                if(gf_array[i-1, k] == '1'):
                    ug.write(',1')
                else:
                    ug.write(',0')
            ug.write("\n")

        ug.close()

        uf = open(os.path.join(graph_path, 'fault-fault-intersection.txt'), 'r')
        contents = uf.readlines()
        uf.close()
        # display(unique_list)
        unique_list_ff = []

        for row in contents:
            row = row.replace("\n", "").split("{")
            fault_1o = row[0].split(',')
            fault_1o = fault_1o[1]

            faults_2o = row[1].replace("(", "").replace(
                ")", "").replace("}", "").split(",")

            if ((fault_1o.replace(" ", "") not in unique_list_ff) and (fault_1o.replace(" ", "") in unique_list)):
                unique_list_ff.append(fault_1o.replace(" ", ""))
            for i in range(0, len(faults_2o), 3):
                if ((faults_2o[i].replace(" ", "") not in unique_list_ff) and (faults_2o[i].replace(" ", "") in unique_list)):
                    unique_list_ff.append(faults_2o[i].replace(" ", ""))
        # display(unique_list)

        G = nx.DiGraph()
        ff = open(os.path.join(output_path, 'fault-fault-relationships.csv'), 'w')
        ff.write('fault_id')
        for i in range(0, len(unique_list_ff)):
            ff.write(','+'Fault_'+unique_list_ff[i])
            G.add_node('Fault_'+unique_list_ff[i])
        ff.write('\n')

        for i in range(0, len(unique_list_ff)):  # loop thorugh rows
            ff.write('Fault_'+unique_list_ff[i])
            found = False
            # for j in range(0,len(unique_list)):
            for row in contents:  # loop thorugh known intersections
                row = row.replace("\n", "").split("{")
                fault_1o = row[0].split(',')
                fault_1o = fault_1o[1]
                faults_2o = row[1].replace("(", "").replace(
                    ")", "").replace("}", "").split(",")

                # correct first order fault for this row
                if(unique_list_ff[i].replace(" ", "") == fault_1o.replace(" ", "")):
                    found = True
                    for k in range(0, len(unique_list_ff)):  # loop through columns
                        found2 = False
                        if(k == i):  # no self intersections
                            ff.write(',0')
                        else:
                            # loop through second order faults for this row
                            for f2o in range(0, len(faults_2o), 3):
                                if (faults_2o[f2o].replace(" ", "") == unique_list_ff[k].replace(" ", "")):
                                    ff.write(',1')
                                    G.add_edge(
                                        'Fault_'+unique_list_ff[i], 'Fault_'+faults_2o[f2o].replace(" ", ""))
                                    found2 = True
                                    break

                        if(not found2 and k != i):
                            # this is not a second order fault for this row
                            ff.write(',0')
                if(found):
                    break
            if(not found):  # this fault is not a first order fault relative to another fault
                for i in range(0, len(unique_list_ff)):
                    ff.write(',0')

            ff.write('\n')

        ff.close()

        if not quiet:
            nx.draw(G, with_labels=True, font_weight='bold')
        
        GD = G.copy()
        edges = list(G.edges)
        cycles = list(nx.simple_cycles(G))
 
        for i in range(0, len(edges)):  # remove first edge from fault cycle
            for j in range(0, len(cycles)):
                if (edges[i][0] == cycles[j][0]
                        and edges[i][1] == cycles[j][1]):
                            if GD.has_edge(edges[i][0],edges[i][1]):
                                    GD.remove_edge(edges[i][0],edges[i][1])
                                    print('fault cycle removed:',edges[i][0],edges[i][1])



        g=open(os.path.join(graph_path,'fault-fault-intersection.txt'),"r")
        contents =g.readlines()
        for line in contents:
            parts=line.split(",")
            f1='Fault_'+parts[1]
            f1=f1.replace(" ","")
            for fn in range(int((len(parts)-2)/3)):
                f2='Fault_'+parts[2+(fn*3)].replace("{","").replace("(","")
                f2=f2.replace(" ","")
                angle=parts[4+(fn*3)].replace("}","").replace(")","")
                angle=angle.replace(" ","").replace("}","").replace(")","").rstrip()
                if GD.has_edge(f1,f2):
                    GD[f1][f2]['angle']=angle

        nx.write_gml(GD, os.path.join(tmp_path, "fault_network.gml"))

        try:
            print('cycles', list(nx.simple_cycles(GD)))
        except:
            print('no cycles')

    ####################################
    # save out geology polygons in WKT format
    #
    # save_geol_wkt(sub_geol,geology_file_csv, c_l)
    # sub_geol geopandas format geology layer geology_file_csv path to output WKT format file
    # c_l dictionary of codes and labels specific to input geo information layers
    #
    # Saves geology layer as WKT format for use by map2model c++ code
    ####################################
    def save_geol_wkt(sub_geol, geology_file_csv, c_l, hint_flag):
        # hint_flag=False
        if(hint_flag == True):
            print("Using ENS age hints")
            code_hint_file = '../test_data3/data/code_age_hint.csv'  # input code_hint file
            code_hints = pd.read_csv(code_hint_file, sep=',')
            code_hints.drop_duplicates(inplace=True)
            code_hints.set_index('code',  inplace=True)
            hint_list = []
            for indx, row in code_hints.iterrows():
                if(not indx in hint_list):
                    hint_list.append(indx)
        f = open(geology_file_csv, "w+")
        f.write('WKT\t'+c_l['o'].replace("\n", "")+'\t'+c_l['u'].replace("\n", "")+'\t'+c_l['g'].replace("\n", "")+'\t'+c_l['min'].replace("\n", "")+'\t'+c_l['max'].replace(
            "\n", "")+'\t'+c_l['c'].replace("\n", "")+'\t'+c_l['r1'].replace("\n", "")+'\t'+c_l['r2'].replace("\n", "")+'\t'+c_l['ds'].replace("\n", "")+'\n')
        # display(sub_geol)
        print(len(sub_geol), " polygons")
        # print(sub_geol)
        for i in range(0, len(sub_geol)):
            # print('**',sub_geol.loc[i][[c_l['o']]],'++')
            f.write("\""+str(sub_geol.loc[i].geometry)+"\"\t")
            f.write("\""+str(sub_geol.loc[i][c_l['o']])+"\"\t")
            f.write("\""+str(sub_geol.loc[i][c_l['c']])+"\"\t")
            # since map2model is looking for "" not "None"
            f.write(
                "\""+str(sub_geol.loc[i][c_l['g']]).replace("None", "")+"\"\t")
            if(hint_flag == True and sub_geol.loc[i][c_l['c']] in hint_list):
                hint = code_hints.loc[sub_geol.loc[i][c_l['c']]]['hint']
            else:
                hint = 0.0
            if(str(sub_geol.loc[i][c_l['min']]) == 'None'):
                min = 0.0
            else:
                min = float(sub_geol.loc[i][c_l['min']])+float(hint)
            # print(str(sub_geol.loc[i][c_l['max']]))
            if(str(sub_geol.loc[i][c_l['max']]) == 'None'):
                max = 4500000000
            else:
                max = float(sub_geol.loc[i][c_l['max']])+float(hint)
            # f.write("\""+str(sub_geol.loc[i][c_l['min']])+"\"\t")
            # f.write("\""+str(sub_geol.loc[i][c_l['max']])+"\"\t")
            f.write("\""+str(min)+"\"\t")
            f.write("\""+str(max)+"\"\t")
            f.write("\""+str(sub_geol.loc[i][c_l['u']])+"\"\t")
            f.write("\""+str(sub_geol.loc[i][c_l['r1']])+"\"\t")
            f.write("\""+str(sub_geol.loc[i][c_l['r2']])+"\"\t")
            f.write("\""+str(sub_geol.loc[i][c_l['ds']])+"\"\n")
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

        sub_pts.columns = ['WKT'] + list(sub_pts.columns[1:])
        sub_pts.to_csv(structure_file_csv, sep='\t',
                       index=False)

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

        sub_mindep.columns = ['WKT'] + list(sub_mindep.columns[1:])
        sub_mindep.to_csv(mindep_file_csv, sep='\t',
                          index=False)

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
        sub_lines.columns = ['WKT'] + list(sub_lines.columns[1:])
        # Filter cells with a feature description containing the word 'fault'
        mask = sub_lines[c_l['f']].str.contains(
            'fault', na=False, case=False, regex=True)
        sub_faults = sub_lines[mask]
        sub_faults.to_csv(fault_file_csv, sep='\t',
                          index=False)

    def check_near_fault_contacts(path_faults, all_sorts_path, fault_dimensions_path, gp_fault_rel_path, contacts_path, c_l, proj_crs):
        faults_clip = gpd.read_file(path_faults)
        gp_fault_rel = pd.read_csv(gp_fault_rel_path)
        gp_fault_rel.set_index('group',  inplace=True)
        contacts = pd.read_csv(contacts_path)
        all_sorts = pd.read_csv(all_sorts_path)
        all_sorts.set_index('code',  inplace=True)
        fault_dimensions = pd.read_csv(fault_dimensions_path)
        fault_dimensions2 = fault_dimensions.set_index('Fault')
        groups = all_sorts['group'].unique()

        for indx, flt in faults_clip.iterrows():
            #print('Fault_'+str(flt[c_l['o']]) )
            if('Fault_'+str(flt[c_l['o']]) in fault_dimensions['Fault'].values):
                #print('Fault_'+str(flt[c_l['o']]),flt.geometry.type,flt.geometry.centroid )
                if(flt.geometry.type == 'LineString'):
                    flt_ls = LineString(flt.geometry)
                    midx = flt_ls.coords[0][0]+((flt_ls.coords[0]
                                                 [0]-flt_ls.coords[len(flt_ls.coords)-1][0])/2)
                    midy = flt_ls.coords[0][1]+((flt_ls.coords[0]
                                                 [1]-flt_ls.coords[len(flt_ls.coords)-1][1])/2)
                    l, m = m2l_utils.pts2dircos(flt_ls.coords[0][0], flt_ls.coords[0][1],
                                                flt_ls.coords[len(flt_ls.coords)-1][0], flt_ls.coords[len(flt_ls.coords)-1][1])
                    angle = (360+degrees(atan2(l, m))) % 360
                    xax = fault_dimensions2.loc['Fault_' +
                                                str(flt[c_l['o']])]['HorizontalRadius']*.99*.81
                    yax = fault_dimensions2.loc['Fault_' +
                                                str(flt[c_l['o']])]['InfluenceDistance']*.99*.81
                    circle = Point(flt.geometry.centroid.x, flt.geometry.centroid.y).buffer(
                        1)  # type(circle)=polygon
                    ellipse = shapely.affinity.scale(
                        circle, xax, yax)  # type(ellipse)=polygon
                    ellipse = shapely.affinity.rotate(
                        ellipse, 90-angle, origin='center', use_radians=False)
                    splits = split(ellipse, flt.geometry)
                    # display(flt.geometry)
                    i = 0
                    for gp in groups:
                        if(not gp == 'cover'):
                            all_sorts2 = all_sorts[all_sorts["group"] == gp]
                            first = True
                            for half in splits:
                                half_poly = Polygon(half)
                                half_ellipse = gpd.GeoDataFrame(
                                    index=[0], crs=proj_crs, geometry=[half_poly])
                                has_contacts = True
                                for indx, as2 in all_sorts2.iterrows():
                                    contacts2 = contacts[contacts["formation"] == indx]
                                    if(first):
                                        first = False
                                        all_contacts = contacts2.copy()
                                    else:
                                        all_contacts = pd.concat(
                                            [all_contacts, contacts2], sort=False)
                                contacts_gdf = gpd.GeoDataFrame(all_contacts, geometry=[Point(
                                    x, y) for x, y in zip(all_contacts.X, all_contacts.Y)])
                                found = gpd.sjoin(
                                    contacts_gdf, half_ellipse, how='inner', op='within')

                                if(len(found) > 0 and has_contacts):
                                    has_contacts = True
                                else:
                                    has_contacts = False
                                i = i+1
                            # print('Fault_'+str(flt[c_l['o']]),gp,has_contacts)
                            if(not has_contacts):
                                if(gp_fault_rel.loc[gp, 'Fault_'+str(flt[c_l['o']])] == 1):
                                    print(
                                        gp, 'Fault_'+str(flt[c_l['o']]), 'combination switched OFF')
                                    gp_fault_rel.loc[gp, 'Fault_' +
                                                     str(flt[c_l['o']])] = 0

                elif(flt.geometry.type == 'MultiLineString' or flt.geometry.type == 'GeometryCollection'):
                    raise NameError('map2loop error: Fault_'+str(
                        flt[c_l['o']])+'cannot be analysed as it is a multilinestring,\n  it may be a fault that is clipped into two parts by the bounding box\nfix in original shapefile??')
                    continue
                    for pline in flt.geometry:
                        flt_ls = LineString(pline)
                        midx = flt_ls.coords[0][0]+(
                            (flt_ls.coords[0][0]-flt_ls.coords[len(flt_ls.coords)-1][0])/2)
                        midy = flt_ls.coords[0][1]+(
                            (flt_ls.coords[0][1]-flt_ls.coords[len(flt_ls.coords)-1][1])/2)
                    l, m = m2l_utils.pts2dircos(flt_ls.coords[0][0], flt_ls.coords[0][1],
                                                flt_ls.coords[len(flt_ls.coords)-1][0], flt_ls.coords[len(flt_ls.coords)-1][1])
                    angle = (360+degrees(atan2(l, m))) % 360

                    xax = fault_dimensions2.loc['Fault_' +
                                                str(flt[c_l['o']])]['HorizontalRadius']*.99*.81
                    yax = fault_dimensions2.loc['Fault_' +
                                                str(flt[c_l['o']])]['InfluenceDistance']*.99*.81
                    circle = Point(flt.geometry.centroid.x, flt.geometry.centroid.y).buffer(
                        1)  # type(circle)=polygon
                    ellipse = shapely.affinity.scale(
                        circle, xax, yax)  # type(ellipse)=polygon
                    ellipse = shapely.affinity.rotate(
                        ellipse, 90-angle, origin='center', use_radians=False)
                    splits = split(ellipse, flt.geometry)
                    # display(splits)
                    i = 0
                    for gp in groups:
                        if(not gp == 'cover'):
                            all_sorts2 = all_sorts[all_sorts["group"] == gp]
                            first = True
                            for half in splits:
                                half_poly = Polygon(half)
                                half_ellipse = gpd.GeoDataFrame(
                                    index=[0], crs=proj_crs, geometry=[half_poly])
                                has_contacts = True
                                for indx, as2 in all_sorts2.iterrows():
                                    contacts2 = contacts[contacts["formation"] == indx]
                                    if(first):
                                        first = False
                                        all_contacts = contacts2.copy()
                                    else:
                                        all_contacts = pd.concat(
                                            [all_contacts, contacts2], sort=False)
                                contacts_gdf = gpd.GeoDataFrame(all_contacts, geometry=[Point(
                                    x, y) for x, y in zip(all_contacts.X, all_contacts.Y)])
                                found = gpd.sjoin(
                                    contacts_gdf, half_ellipse, how='inner', op='within')

                                if(len(found) > 0 and has_contacts):
                                    has_contacts = True
                                else:
                                    has_contacts = False
                                i = i+1
                            # print('Fault_'+str(flt[c_l['o']]),gp,has_contacts)
                            if(not has_contacts):
                                if(gp_fault_rel.loc[gp, 'Fault_'+str(flt[c_l['o']])] == 1):
                                    print(
                                        gp, 'Fault_'+str(flt[c_l['o']]), 'combination switched OFF')
                                    gp_fault_rel.loc[gp, 'Fault_' +
                                                     str(flt[c_l['o']])] = 0

        gp_fault_rel.to_csv(gp_fault_rel_path)

    def super_groups_and_groups(group_girdle, tmp_path, misorientation):
        group_girdle = pd.DataFrame.from_dict(group_girdle, orient='index')
        group_girdle.columns = ['plunge', 'bearing', 'num orientations']
        group_girdle.sort_values(
            by='num orientations', ascending=False, inplace=True)
        display(group_girdle)

        l, m, n = m2l_utils.ddd2dircos(
            group_girdle.iloc[0]['plunge'], group_girdle.iloc[0]['bearing'])
        super_group = pd.DataFrame([[group_girdle[0:1].index[0], 'Super_Group_0', l, m, n]], columns=[
                                   'Group', 'Super_Group', 'l', 'm', 'n'])
        super_group.set_index('Group', inplace=True)
        sg_index = 0
        for i in range(1, len(group_girdle)):
            if(group_girdle.iloc[i]['num orientations'] > 3):
                l, m, n = m2l_utils.ddd2dircos(
                    group_girdle.iloc[i]['plunge'], group_girdle.iloc[i]['bearing'])

                found = False
                sg_i = 0
                for ind, sg in super_group.iterrows():

                    c = sg['l']*l + sg['m']*m + sg['n']*n
                    if c > 1:
                        c = 1
                    c = degrees(acos(c))
                    if(c < misorientation and not found):
                        found = True
                        sgname = 'Super_Group_'+str(sg_i)
                        super_group_old = pd.DataFrame(
                            [[group_girdle[i:i+1].index[0], sgname, l, m, n]], columns=['Group', 'Super_Group', 'l', 'm', 'n'])
                        super_group_old.set_index('Group', inplace=True)
                        super_group = super_group.append(super_group_old)
                    sg_i = sg_i+1
                if(not found):

                    sg_index = sg_index+1
                    #print('not found',sg_index)
                    sgname = 'Super_Group_'+str(sg_index)
                    super_group_new = pd.DataFrame(
                        [[group_girdle[i:i+1].index[0], sgname, l, m, n]], columns=['Group', 'Super_Group', 'l', 'm', 'n'])
                    super_group_new.set_index('Group', inplace=True)
                    super_group = super_group.append(super_group_new)
            else:  # not enough orientations to test, so lumped with group with most orientations
                sgname = 'Super_Group_'+str(0)
                super_group_old = pd.DataFrame(
                    [[group_girdle[i:i+1].index[0], sgname, l, m, n]], columns=['Group', 'Super_Group', 'l', 'm', 'n'])
                super_group_old.set_index('Group', inplace=True)
                super_group = super_group.append(super_group_old)

        use_gcode3 = []
        for ind, sg in super_group.iterrows():
            clean = ind.replace(" ", "_").replace("-", "_")
            use_gcode3.append(clean)

        sg2 = set(super_group['Super_Group'])
        super_groups = []
        for s in sg2:
            temp = []
            for ind, sg in super_group.iterrows():
                if(s == sg['Super_Group']):
                    temp.append(ind)
            super_groups.append(temp)

        f = open(os.path.join(tmp_path, 'super_groups.csv'), 'w')
        for sg in super_groups:
            for s in sg:
                f.write(str(s)+',')
            f.write('\n')
        f.close()
        return(super_groups, use_gcode3)

    def use_asud(strat_graph_file, graph_path):
        asud_strat_file = "https://gist.githubusercontent.com/yohanderose/3b257dc768fafe5aaf70e64ae55e4c42/raw/8598c7563c1eea5c0cd1080f2c418dc975cc5433/ASUD.csv"

        G = nx.read_gml(strat_graph_file, label='id')
        Gp = G.copy().to_directed()

        try:
            ASUD = pd.read_csv(asud_strat_file, ",")
        except Exception as e:
            print(e)
            return

        for e in G.edges:
            glabel_0 = G.nodes[e[0]]['LabelGraphics']['text']
            glabel_1 = G.nodes[e[1]]['LabelGraphics']['text']

            edge = ASUD.loc[(ASUD['over'] == glabel_0) &
                            (ASUD['under'] == glabel_1)]
            if(len(edge) > 0 and (not('isGroup' in Gp.nodes[e[0]]) and not('isGroup' in Gp.nodes[e[1]]))):
                if(Gp.has_edge(e[1], e[0])):
                    Gp.remove_edge(e[1], e[0])
                if(Gp.has_edge(e[0], e[1])):
                    Gp.remove_edge(e[0], e[1])
                Gp.add_edge(e[0], e[1])

            edge = ASUD.loc[(ASUD['under'] == glabel_0) &
                            (ASUD['over'] == glabel_1)]
            if(len(edge) > 0 and (not('isGroup' in Gp.nodes[e[0]]) and not('isGroup' in Gp.nodes[e[1]]))):
                if(Gp.has_edge(e[0], e[1])):
                    Gp.remove_edge(e[0], e[1])
                if(Gp.has_edge(e[1], e[0])):
                    Gp.remove_edge(e[1], e[0])
                Gp.add_edge(e[1], e[0])

        recode = {}
        i = 0
        for n in Gp.nodes:
            if('isGroup' in Gp.nodes[n]):
                recode[n] = i
            i = i+1

        for n in Gp.nodes:
            if(not 'isGroup' in Gp.nodes[n]):
                Gp.nodes[n]['gid'] = recode[Gp.nodes[n]['gid']]

        # check for cycle and arbitrarily remove first of the edges
        try:
            cycles = list(nx.simple_cycles(G))
            for c in cycles:
                G.remove_edge(c[0], c[1])
                warning_msg = 'map2loop warning: The stratigraphic relationship: "' + \
                    c[0]+' overlies '+c[1] + \
                    '" was removed as it conflicts with another relationship'
                warnings.warn(warning_msg)
        except:
            print('no cycles')

        nx.write_gml(Gp, os.path.join(graph_path, 'ASUD_strat.gml'))

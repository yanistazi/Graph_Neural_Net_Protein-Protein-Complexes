import sys
sys.path.append('/bgfs01/insite/yanis.tazi/Scoring_Project/Graph_Project/')
sys.path.append('/bgfs01/insite/yanis.tazi/Scoring_Project/DeepRank-GNN/')

from Model_Functions.Data_Preparation import *

graph = GraphHDF5(pdb_path="/bgfs01/insite02/taras.dauzhenka/data/docking_scoring/3HMX.1/",
         graph_type='residue', outfile="/bgfs01/insite/yanis.tazi/Scoring_Project/Graph_Project/new_decoys_scoring_predictions/3hmx/residues.hdf5", nproc=24,tmpdir="/bgfs01/insite/yanis.tazi/Scoring_Project/Graph_Project/pkl1/")

import os
import numpy as np

pdb_path = "/bgfs01/insite02/taras.dauzhenka/data/docking_scoring/3HMX.1/"
graph_path = "/bgfs01/insite/yanis.tazi/Scoring_Project/Graph_Project/new_decoys_scoring_predictions/3hmx/"
min_normalized = 0
max_normalized = 20
pdb_list = [l for l in os.listdir(pdb_path) if ".pdb" in l]
np.savetxt(graph_path+"target_list.txt",[l[:-4]+" "+l.split("_")[-1][:-4] for l in pdb_list], fmt='%s')
np.savetxt(graph_path+"target_list_normalized.txt",
           [l[:-4]+" "+str((float(l.split("_")[-1][:-4])-min_normalized)/(max_normalized-min_normalized))
            for l in pdb_list],
           fmt='%s')

CustomizeGraph.add_target(graph_path=graph_path, target_name='scoring',
                                  target_list=graph_path+"target_list.txt", sep=' ') 
CustomizeGraph.add_target(graph_path=graph_path, target_name='scoring_normalized',
                                  target_list=graph_path+"target_list_normalized.txt", sep=' ')
import sys
sys.path.append('/bgfs01/insite/yanis.tazi/Scoring_Project/Graph_Project/')
sys.path.append('/bgfs01/insite/yanis.tazi/Scoring_Project/DeepRank-GNN/')

from Model_Functions.Data_Preparation import *

dir_path = "/bgfs01/insite02/yanis.tazi/data/graph_project_data/131/"
pdb_data_dir="/bgfs01/insite02/yanis.tazi/data/graph_project_data/pdb_selected_data_target_irmsd_5/"
graph_dir = "/bgfs01/insite02/yanis.tazi/data/graph_project_data/hdf5_target_5_replicates/hdf5_pdb_graphs_copy_1/"
list_complexes = "" 
min_normalized = 0
max_normalized = 20
add_target = True
add_normalized_target = True
nproc = 1
tmpdir = "pkl/"
txt_file_name = "target_5.txt"
add_input_complex = True

prepare_hdf5_graphs(dir_path = dir_path,
                    pdb_data_dir = pdb_data_dir,
                    graph_dir = graph_dir,
                    list_complexes = list_complexes,
                    min_normalized = min_normalized,
                    max_normalized = max_normalized,
                    add_target = add_target,
                    add_normalized_target = add_normalized_target,
                    nproc = nproc,
                    tmpdir = tmpdir,
                    txt_file_name = txt_file_name,
                    add_input_complex = add_input_complex)
import sys
import os
import random

random.seed(129)

sys.path.append('/bgfs01/insite/yanis.tazi/Scoring_Project/Graph_Project/')
sys.path.append('/bgfs01/insite/yanis.tazi/Scoring_Project/DeepRank-GNN/')

from Model_Functions.Model_Training import *

dir_path_list = ["/bgfs01/insite02/yanis.tazi/data/graph_project_data/hdf5_target_5_replicates/hdf5_pdb_graphs_copy_4/"]

outdir = '/bgfs01/insite/yanis.tazi/Scoring_Project/Graph_Project/latest_gridsearch_irmsd_5/random_split_test_3PC8_3K75/'

# Specifically set database eval 

#list_complexes is training
list_complexes = ['2X9A','3HMX_AA','CP57','2W9E_AA','4HX3','BAAD','3A4S','1RKE','3DAW','3F1P','3RVW_AA','3HI6_AA','3AAA',
 '3MXW_AA','3L5W_AA','2YVJ','4IZ7','2GTP','4GAM','4GXU_AA','2A1A','4FZA','3G6D_AA','2GAF','3S9D','4H03',
 '3EO1_AA','3L89','3V6Z_AA','3EOA_AA','3SZK','3FN1','3VLB','3AAD','4DN4_AA','3H2V','4JCV','3BIW','3BX7',
 '3P57','4M76','2VXT_AA','3H11','1JTD','3LVK','4FQI_AA','4G6J_AA','3R9A','4G6M_AA','BP57','4LW4','1EXB']

#eval set
database_eval = None

percent_list = [[0.8, 0.2]]

optimizer_name_list = ["Adam"]

neuralnet_arch_list = [EGAT_Net]

lr_list = [0.0001]

nepoch_list = [1000]

keep_latest_saved_models = 300

keep_latest_saved_plots = 300

gridsearch_index = 14

grid_search(dir_path_list=dir_path_list,
            optimizer_name_list=optimizer_name_list,
            list_complexes=list_complexes,
            database_eval=database_eval,
            percent_list=percent_list,
            neuralnet_arch_list=neuralnet_arch_list,
            lr_list=lr_list,
            outdir=outdir,nepoch_list=nepoch_list,
            keep_latest_saved_models=keep_latest_saved_models,
            keep_latest_saved_plots=keep_latest_saved_plots,
            gridsearch_index=gridsearch_index)


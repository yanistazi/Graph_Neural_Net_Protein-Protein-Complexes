import sys
import os
import random

random.seed(129)

sys.path.append('/bgfs01/insite/yanis.tazi/Scoring_Project/Graph_Project/')
sys.path.append('/bgfs01/insite/yanis.tazi/Scoring_Project/DeepRank-GNN/')

from Model_Functions.Model_Testing_Predictions import *

dir_path = "/bgfs01/insite/yanis.tazi/Scoring_Project/Graph_Project/new_decoys_scoring_predictions/"

for l in ['3K75','3PC8']:

    file_dir = "/bgfs01/insite/yanis.tazi/Scoring_Project/Graph_Project/latest_gridsearch_irmsd_5/random_split_test_3PC8_3K75/12/"
    indices = [407,476] 
    pretrained_models = [file_dir+"treg_yscoring_normalized_b1024_e1000_lr0.001_"+str(index)+".pth.tar" for index in indices]

    
    for pretrain_model in pretrained_models:
        model_testing(dir_path=dir_path,
                      outdir=file_dir+"testing_results_"+pretrain_model.split("_")[-1].split(".")[0]+"_"+l+"/",
                      list_complexes=[l],                 
                      pretrained_model=pretrain_model,
                      neuralnet_arch=EGAT_Net,
                      optimizer_name="RMSprop",
                      threshold=5,
                      get_metrics=True,
                      compute_raw=True,
                      min_val=0,
                      max_val=20)    
        


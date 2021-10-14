import os
from shutil import copyfile
import shutil
import errno
import glob
import numpy as np
import random
import json
from deeprank_gnn.GraphGenMP import GraphHDF5
import deeprank_gnn.CustomizeGraph as CustomizeGraph
import warnings
from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.ginet import *
from deeprank_gnn.foutnet import FoutNet
from deeprank_gnn.EGAT_GNN import EGAT_Net
from deeprank_gnn.wgat_conv import WGATConv
import time
import datetime
from sklearn import metrics
import shutil


def compute_raw_metrics_and_raw_data(pred_normalized="",y_normalized="",outdir="",min_val=0.701,max_val=64.606):
    
    """
    Compute various regression metrics and convert normalized data to raw data using min - max provided. 
    Returns 2 .txt files ; 1. with the metrics for the raw data and 2. the predicted and output raw values.
    
        Args:
        
            - pred_normalized (list, required): list of normalized predictions
            
            - y_normalized (list, required): list of normalized targets
            
            - outdir (str, optional): output directory for the test results. Defaults to ./
            
            - min_val (float, optional): Min value used for normalization. Defaults to 0.701.
            
            - max_val (float, optional): Min value used for normalization. Defaults to 64.606.
    """    
    pred_raw = [(max_val-min_val)*pred + min_val for pred in pred_normalized]
    y_raw = [(max_val-min_val)*y + min_val for y in y_normalized]

    explained_variance = metrics.explained_variance_score(y_raw, pred_raw)

    # Max_error metric calculates the maximum residual error
    max_error = metrics.max_error(y_raw, pred_raw)

    # Mean absolute error regression loss
    mean_absolute_error = metrics.mean_absolute_error(y_raw, pred_raw)

    # Mean squared error regression loss
    mean_squared_error = metrics.mean_squared_error(y_raw, pred_raw, squared = True)

    # Root mean squared error regression loss
    root_mean_squared_error = metrics.mean_squared_error(y_raw, pred_raw, squared = False)

    try:
        # Mean squared logarithmic error regression loss
        mean_squared_log_error = metrics.mean_squared_log_error(y_raw, pred_raw)
    except ValueError:
        mean_squared_log_error="not computed"
        print ("WARNING: Mean Squared Logarithmic Error cannot be used when "
                    "targets contain negative values.")

    # Median absolute error regression loss
    median_squared_log_error = metrics.median_absolute_error(y_raw, pred_raw)

    # R^2 (coefficient of determination) regression score function
    r2_score = metrics.r2_score(y_raw, pred_raw)
    
    f = open(outdir+"test_metrics_raw.txt","w")
    f.write( str({"explained_variance":explained_variance,
           "max_error":max_error,
           "mean_absolute_error":mean_absolute_error,
           "mean_squared_error":mean_squared_error,
           "root_mean_squared_error":root_mean_squared_error,
           "mean_squared_log_error":mean_squared_log_error,
           "median_squared_log_error":median_squared_log_error,
           "r2_score":r2_score
          }) )
    f.close()
        
    f = open(outdir+"test_predictions_raw.txt","w")
    f.write( str({"prediction":pred_raw,
                  "y":y_raw}) )
    f.close()

def model_testing(dir_path="",outdir="",list_complexes="",neuralnet_arch="",pretrained_model="",optimizer_name="",
                 threshold=10,get_metrics=False,compute_raw=False,min_val=0.701,max_val=64.606):
    
    """
    Test model using saved .pth.tar in pretrained_model. Needs to map pretrained model with the correct neuralnet
    architecture and optimizer. It takes as input a path to .hdf5 graph files for the complexes and we can 
    specify a subset of complexes to be tested from that path or all the .hdf5 files if list is empty.
    
    It returns :
        1. test.hdf5 with molecule names , outputs and predictions
        If get_metrics is False
        2. .txt file with the predictions for all complexes + target values (if get_metrics is True)
           list with the predictions for all the pdb files.
        3. Returns the normalized data metrics if get_metrics is True
        4. If compute_raw is True return .txt file with the predictions and targets
        5. If compute_raw is True return .txt file with the metrics
    
        Args:
        
            - dir_path (str, required): path(s) to hdf5 dataset(s). Unique hdf5 file or list of hdf5 files.
            
            - outdir (str, optional): output directory for the test results. Defaults to ./
            
            - list_complexes (list,optional): list of names with the complexes to test (one subfolder per complex in dir_path).
                                              If not specified, we test all complexes from dir_path.
             
            - pretrained_model (str,required): path of the pretrained model to use for testing . Torch .pth.tar file
            
            - neuralnet_arch (class, required): architecture class of the graph neural network. The class should match the 
                                                architecture class of the pretrained model. If new architecture, make sure to
                                                import it.
            
            - optimizer_name (str, required): name of the optimizer used in the pretrained model. The name should match the
                                              optimizer name used in the pretrained model.
            
            - threshold (float, optional): The threshold value for binary classification (high vs low i-RMSD in this case). 
                                           This is only used for classification metrics.
                                         
            - get_metrics (boolean, optional): Whether or not to return the different test metrics . Defaults to False
            
            - compute_raw (boolean, optional): Whether or not to compute metrics for not normalized data. 
                                               Requires min and max. Default to False.
            
            - min_val (float, optional): Min value used for normalization. Used only if compute_raw is True.
                                         Defaults to 0.701.
            
            - max_val (float, optional): Min value used for normalization. Used only if compute_raw is True.
                                         Defaults to 0.701.
    """
    if(list_complexes==""): # if not specified get all complexes from dir_path
        list_complexes = os.listdir(dir_path)
        list_complexes = [ l for l in list_complexes if "." not in l]

    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    else:
        print(outdir + " exists")


    database_test = [dir_path+l+"/residues.hdf5"
                      for l in list_complexes]
    
    model = NeuralNet(database=database_test,
                      Net=neuralnet_arch,
                      pretrained_model=pretrained_model,
                      optimizer_name=optimizer_name,
                      outdir=outdir)
    
    model.test(database_test)
    
    # For this part, we need target values to compare them to predictions
    
    test_metrics = model.get_metrics('test', threshold = threshold) # Threshold is used for conversion to classification metrics
    
    if(get_metrics):
        
        dict_metrics={'accuracy' : test_metrics.accuracy,
                  'explained_variance' : test_metrics.explained_variance,
                  'FDR' : test_metrics.FDR,
                  'FNR' : test_metrics.FNR,
                  'max_error' : test_metrics.max_error,
                  'mean_absolute_error' : test_metrics.mean_absolute_error,
                  'mean_squared_error' : test_metrics.mean_squared_error,
                  'mean_squared_log_error' : test_metrics.mean_squared_log_error, 
                  'median_squared_log_error' : test_metrics.median_squared_log_error,
                  'precision' : test_metrics.precision,
                  'r2_score' : test_metrics.r2_score,
                  'root_mean_squared_error' : test_metrics.root_mean_squared_error,
                  'sensitivity' : test_metrics.sensitivity
                 }
        dict_predictions={
                  'prediction' : test_metrics.prediction,
                  'y' : test_metrics.y
        }

        f = open(outdir+"test_metrics.txt","w")
        f.write( str(dict_metrics) )
        f.close()
        
        f = open(outdir+"test_predictions.txt","w")
        f.write( str(dict_predictions) )
        f.close()
        
        if (compute_raw):
            compute_raw_metrics_and_raw_data(pred_normalized=test_metrics.prediction,
                                             y_normalized=test_metrics.y,
                                             outdir=outdir,min_val=min_val,max_val=max_val)
    else :
        dict_predictions={
                  'prediction' : test_metrics.prediction
        }
        
        f = open(outdir+"test_predictions.txt","w")
        f.write( str(dict_predictions) )
        f.close()
        
        if (compute_raw):
            pred_raw = [(max_val-min_val)*pred + min_val for pred in test_metrics.prediction]
            f = open(outdir+"test_predictions_raw.txt","w")
            dict_predictions={'prediction' : pred_raw}
            f.write( str(dict_predictions) )
            f.close()
    
    #
    return model


def convert_decoys_to_graph(path_decoys="",list_target_txt="",nproc=1,tmpdir="pkl/",
                           contact_distance=8.5, internal_contact_distance=3):
    """
    Convert pdb decoys to a graph that match the testing input required format. Output is path_decoys+"/graph/residues.hdf5".
    If list_target_txt is not empty , use it for the scoring in the graph for all the pdbs.
    If list_target_txt is empty , create fake scoring in the graph for all the pdbs.
    
        Args:
        
            - path_decoys (str, required): path(s) to decoy files. The script will convert all the .pdb files in that folder.
            
            - list_target_txt (str, optional): list to use for pdb score. Defaults to ""
            
            - nproc (int, optional): number of processors. Default to 1.

            - tmpdir (str, optional): Default to "pkl/".
            
            - contact_distance (float, optional): Contact distance to create nodes of the graph. 
                                                  Useful for new pdb (input complexes if the default cutoff is too low
                                                  and thus, graph is empty). Defaults to 8.5.
            
            - internal_contact_distance (float, optional): Contact distance to create edges of the graph. 
                                                  Useful for new pdb (input complexes if the default cutoff is too low
                                                  and thus, graph does not have edges). Defaults to 3.            
    """    
    # Make sure list_target_txt has the right format
    try:
        os.makedirs(path_decoys+"/graph/")
    except:
        pass
    

    # Quick hack if we present only one pdb file. We duplicate it so that it works with the customized graph.
    # Only done when one pdb
    list_pdbs = os.listdir(path_decoys)
    list_pdbs = [l for l in list_pdbs if ".pdb" in l]
    if(len(list_pdbs)==1):
        print("Create duplicate because only one pdb")
        source = path_decoys+list_pdbs[0]
        destination = path_decoys+list_pdbs[0].split(".")[0]+"-copy.pdb"
        shutil.copyfile(source, destination)
    # End of hack
    
    
    graph = GraphHDF5(pdb_path=path_decoys,
             graph_type='residue', outfile=path_decoys+"/graph/residues.hdf5", nproc=nproc,tmpdir=tmpdir,
                     contact_distance=contact_distance, internal_contact_distance=internal_contact_distance)


    # Create fake target so that the graph format matches
    list_pdbs = os.listdir(path_decoys)
    list_pdbs = [l for l in list_pdbs if ".pdb" in l]
    

    
    if (list_target_txt==""):
        print("Creating graph with fake scoring just to match graph format")
        tmp_target = [l[:-4]+" 0" for l in list_pdbs]
        np.savetxt(path_decoys+"/graph/target_list_normalized.txt",tmp_target, fmt='%s')
        CustomizeGraph.add_target(graph_path=path_decoys+"/graph/", target_name='scoring_normalized',
                          target_list=path_decoys+"/graph/target_list_normalized.txt", sep=' ')
    else:
        print("Creating graph with scoring from target_list input")
        CustomizeGraph.add_target(graph_path=path_decoys+"/graph/", target_name='scoring_normalized',
                          target_list=list_target_txt, sep=' ')


    
    

def model_new_predictions(path_decoys="",neuralnet_arch="",pretrained_model="",optimizer_name="",
                 threshold=10,convert_decoys=True,nproc=1,tmpdir="pkl/",outdir="",list_target_txt="",
                 compute_raw=False,min_val=0.701,max_val=64.606,contact_distance=8.5, internal_contact_distance=3):
    """
    Predict the i-RMSD on pdb files and use this as a filter to estimate high quality vs low quality new generated decoys.
    The script can convert pdb decoys to a graph that match the testing input required format. 
    Output is path_decoys+"/graph/residues.hdf5".
    If list_target_txt is not empty , use it for the scoring in the graph for all the pdbs.
    If list_target_txt is empty , create fake scoring in the graph for all the pdbs.
    
        Args:
        
            - path_decoys (str, required): path(s) to decoy files. The script will convert all the .pdb files in that folder.
                                           If the decoys are already converted into a graph, just specify the full path to the
                                           .hdf5 graph file.
              
            - neuralnet_arch (class, required): architecture class of the graph neural network. The class should match the 
                                                architecture class of the pretrained model. If new architecture, make sure to
                                                import it.
                                              
            - pretrained_model (str,required): path of the pretrained model to use for testing . Torch .pth.tar file             

            - optimizer_name (str, required): name of the optimizer used in the pretrained model. The name should match the
                                              optimizer name used in the pretrained model.
            
            - threshold (float, optional): The threshold value for binary classification (high vs low i-RMSD in this case). 
                                           This is only used for classification metrics.
            
            - convert_decoys (boolean, optional) : Whether or not to convert the .pdb decoys to graph. Defaults to True
            
            - nproc (int, optional): number of processors. Default to 1.

            - tmpdir (str, optional): Default to "./".
            
            - outdir (str,optional): path where to convert the graphs and store the pkl
            
            - list_target_txt (str, optional): list to use for pdb score. Defaults to ""
            
            - compute_raw (boolean, optional): Whether or not to return raw data. 
                                               Requires min and max. Default to False.
            
            - min_val (float, optional): Min value used for normalization. Used only if compute_raw is True.
                                         Defaults to 0.701.
            
            - max_val (float, optional): Min value used for normalization. Used only if compute_raw is True.
                                         Defaults to 0.701.
                                         
            - contact_distance (float, optional): Contact distance to create nodes of the graph. 
                                                  Useful for new pdb (input complexes if the default cutoff is too low
                                                  and thus, graph is empty). Defaults to 8.5.
            
            - internal_contact_distance (float, optional): Contact distance to create edges of the graph. 
                                                  Useful for new pdb (input complexes if the default cutoff is too low
                                                  and thus, graph does not have edges). Defaults to 3.
    """    
    # 1. Convert decoys into graph
    
    if(convert_decoys):
        convert_decoys_to_graph(path_decoys=path_decoys,list_target_txt=list_target_txt,nproc=nproc,tmpdir="pkl/",
                               contact_distance=contact_distance, internal_contact_distance=internal_contact_distance)
        database_test = path_decoys+"/graph/residues.hdf5"       
    else:
        database_test = path_decoys # give full path in this case
    
    model = NeuralNet(database=database_test,
                      Net=neuralnet_arch,
                      pretrained_model=pretrained_model,
                      optimizer_name=optimizer_name,
                      outdir=outdir,
                      threshold=threshold)
    
    model.test(database_test,threshold=threshold)
    
    if compute_raw:
        return{"normalized":model.test_out,"raw":[(max_val-min_val)*pred + min_val for pred in model.test_out]}
    else:
        return model.test_out
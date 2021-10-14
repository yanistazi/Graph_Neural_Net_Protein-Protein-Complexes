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


def create_dict(outdir="",arguments=""):
    """
        Create arguments.txt file to keep track of all parameters .

        Args:
            
            - outdir (str,required): output directory to save the file 
            
            - arguments (dict,required): dictionary with items and corresponding values to save
    """
    
    with open(outdir+"arguments.txt", "w") as f:
        for key, value in arguments.items():
            print(key, value)
            f.write('{} {}'.format(key, value)+"\n")
            
##### ALWAYS MAKE SURE YOU CHANGE PATH FOR HDF5 TO NOT CONFLICT 


                    
def training_graph(dir_path="",edge_feature=['dist'],
                   node_feature=['type', 'polarity', 'bsa','pos','chain','charge'],
                   batch_size=1024,target='scoring_normalized',task='reg',shuffle=True,
                   lr=0.001,percent = [0.8, 0.2],threshold = 5,list_complexes="",
                   database_eval = None,
                   outdir = '',
                   neuralnet_arch=GINet,nepoch=10,save_model="best",
                   save_epoch='intermediate',save_every=5,plot_every=5,plot_name='plot_loss',
                   optimizer_name="Adam",pretrained_model=None,train_using_pretrained_model=False,
                   epoch_start=0,keep_latest_saved_models=5,keep_latest_saved_plots=5):
     
    """
        Train model using specified parameters with possibility to train a new model from scratch, 
        load checkpoint and resume training or load checkpoint from pretrained model and perform 
        transfer learning.
        
        Outputs :
        1. arguments.txt to keep track of all parameters
        2. latest n best saved models .pth.tar
        3. latest N saved plots .png
        4. train_data.hdf5 with data saved every save_every epochs . It contains training and validation input names with 
           associated outputs and predictions. Helpful to recompute loss function across epochs.
        5. Train loss (train_loss.txt) to keep track of the training loss over training iterations
        6. Validation loss (valid_loss.txt) to keep track of the validation loss over training iterations

        Specificities : handle errors by creating folders of folders if path has not been changed.
    
        Args:
        
            - dir_path (str, required): path(s) to hdf5 dataset(s). Unique hdf5 file or list of hdf5 files.
            
            - edge_feature (list, optional): list of edge features to consider as input for the neural network. Default to 
                                             ['dist'].
                                             
            - node_feature (list, optional): list of node features to consider as input for the neural network. Default to 
                                             ['type', 'polarity', 'bsa','pos','chain','charge'].
                                             
            - batch_size (2**, optional): Default to min(len(data),1024), preferably power of 2.  
            
            - target (str, optional): Which score key to use to train the model and backpropagate. 
                                      Default to scoring_normalized.
            
            - task (str, optional): 'reg' for regression or 'class' for classification. Default to reg.
            
            - shuffle (bool, optional): whether or not to shuffle the training data. Default to True.
            
            - lr (float, optional): learning rate for gradient descent. Defaults to 0.001.
            
            - percent (list, optional): divides the input dataset into a training and evaluation set.
                                         If database_eval not None , it is not used.
                                         
            - threshold (int, optional): threshold to compute binary classification metrics. 
                                         Not used for regression. Defaults to 10.
            
            - list_complexes (list,optional): list of names with the complexes to train on (one subfolder per complex in 
                                              dir_path). If not specified, we test all complexes from dir_path.
                                                                        
            - database_eval (list,optional): independent list of complex names(one subfolder per complex in 
                                              dir_path). Defaults to None.                                   
                                              
            - outdir (str, required): output directory for the train results. 
            
            - neuralnet_arch (class, required): architecture class of the graph neural network.
                                                If new architecture, make sure to import it.
                                                
            - nepoch (int, optional): number of epochs to train. Defaults to 10.
            
            - save_model (last, best, optional): save the model. Defaults to 'best' based on validation loss.
            
            - save_epoch (all, intermediate, optional): Defaults to 'intermediate'.
            
            - save_every (int, optional): save data every n epoch if save_epoch == 'intermediate'. Defaults to 5.
            
            - plot_every (int, optional): plot mse loss data every n epoch . Defaults to 5.
            
            - plot_name (str,optional): name of the loss plot.
            
            - optimizer_name (str,optional): name of the optimizer to use.
            
            - pretrained_model (str,optional): path to pre-trained model. Defaults to None.
            
            - train_using_pretrained_model (bool,optional): whether or not to load the pretrained model weights and parameters
                                                            to continue training. 
                                                            Needs to specify pretrained_model path (.pth.tar). Defaults to False
                                                            
            - epoch_start (int,optional): useful if we use pretrained model and want to continue the training on the same data.
                                          For example, if training is interrupted, we can resume training from epoch_start=X
                                          using saved model .pth.tar at epoch X. 
                                          Defaults to 0.
                                         
            - keep_latest_saved_models (bool,optional): how many latest saved models .pth.tar to keep in order to remove 
                                                        the old ones. Defaults to the 5 latest.

            - keep_latest_saved_plots (bool,optional): how many latest saved plots .png to keep in order to remove 
                                                        the old ones. Defaults to the 5 latest.
    """
    


    if(list_complexes==""): # if not specified get all complexes from dir_path
            list_complexes = os.listdir(dir_path)
            list_complexes = [ l for l in list_complexes if "." not in l]

    



    database_train = [dir_path+l+"/residues.hdf5"
                      for l in list_complexes]
    
    print("Database training :")
    print(database_train)
    
    if database_eval is not None:
        print("Database testing:")
        print(database_train)
        database_eval = [dir_path+l+"/residues.hdf5"
                  for l in database_eval]
        percent = [1.0, 0.0]  # Make sure that we don't divide the training data into train and val since the val data
                              # is provided separately (otherwise part of the data is not used).
        print("percent: ",str(percent))

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    else:
        print(outdir + " exists")
        os.makedirs(outdir+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"/")
        outdir = outdir+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"/"


    create_dict(outdir=outdir,arguments=dict(locals().items()))

        

    model = NeuralNet(database=database_train, Net=neuralnet_arch,
                   node_feature=node_feature,
                   edge_feature=edge_feature,
                   target=target,
                   task=task,
                   database_eval=database_eval,
                   lr=lr,
                   batch_size=batch_size,
                   shuffle=shuffle,
                   percent=percent,
                   outdir=outdir,
                   threshold=threshold,
                   optimizer_name=optimizer_name,
                   pretrained_model=pretrained_model,
                   train_using_pretrained_model=train_using_pretrained_model)


    if pretrained_model is None:
        print("Training from Scratch")
        epoch_start=0
    model.train(nepoch=nepoch, validate=True, save_model=save_model, hdf5='train_data.hdf5', save_epoch=save_epoch,
                save_every=save_every,plot_every=plot_every,plot_name=plot_name,epoch_start=epoch_start,
                keep_latest_saved_models=keep_latest_saved_models,keep_latest_saved_plots=keep_latest_saved_plots)
    
    
def grid_search(dir_path_list="",edge_feature_list=[['dist']],
                   node_feature_list=[['type', 'polarity', 'bsa','pos','chain','charge']],
                   batch_size_list=[1024],lr_list=[0.001],percent_list = [[1.0, 0.0]],
                   neuralnet_arch_list=[""],nepoch_list=[10],optimizer_name_list=[""],
                   pretrained_model_list=[None],
                   target='scoring_normalized',task='reg',shuffle=True,
                   threshold = 5,list_complexes="",database_eval = None,
                   outdir = "",
                   save_model="best",save_epoch='intermediate',
                   save_every=5,plot_every=5,plot_name='plot_loss',
                   keep_latest_saved_models=5,keep_latest_saved_plots=5,gridsearch_index=0) :
    """
        Train the model using a gridsearch to select optimal parameter combination using training_graph function.
        The script creates one folder for each possible combination to be tested . Again, possibility to train a
        new model from scratch, load checkpoint and resume training or load checkpoint from pretrained model and
        perform transfer learning.

        Outputs :
        1. One subfolder per parameters combination . 
           Ex : If we test 2 optimizers (Adam, RMSprop) and 3 graph neural net arch (GINet,FoutNet,EGATNet) with all other
           parameters fixed, it will create 6 subfolders; one for each combination 
           (Adam-GINet ; Adam-FoutNet ; Adam-EGATNet ; RMSprop-GINet ; RMSprop-FoutNet ; RMSprop-EGATNet).
        2. arguments.txt to keep track of all parameters
        3. latest n best saved models .pth.tar
        4. latest N saved plots .png
        5. train_data.hdf5 with data saved every save_every epochs . It contains training and validation input names with 
           associated outputs and predictions. Helpful to recompute loss function across epochs.
        6. Train loss (train_loss.txt) to keep track of the training loss over training iterations
        7. Validation loss (valid_loss.txt) to keep track of the validation loss over training iterations

        Specificities : 
        1. handle misspecified parameters when pretrain model used is not consistent with neuralnet architecture
           or optimization algorithm. 
           Ex: Pretrained model is a GINet model and neuralnet architecture combination is FoutNet.
        2. handle errors by creating folders of folders if path has not been changed.
    
        Args:
            
            ### Parameters to possibly optimize in the gridsearch:
            
            - dir_path_list (list[str], required): path(s) to hdf5 dataset(s). Unique hdf5 file or list of hdf5 files.
            
            - edge_feature_list (list[list], optional): list of edge features to consider as input for the neural network. 
                                                         Default to [['dist']].
                                             
            - node_feature_list (list[list], optional): list of node features to consider as input for the neural network. 
                                                        Default to  [['type', 'polarity', 'bsa','pos','chain','charge']].
                                             
            - batch_size_list (list[2**], optional): Default to min(len(data),1024), preferably power of 2.
            
            - lr_list (list[float], optional): list of learning rate for gradient descent. Defaults to [0.001].
            
            - percent_list (list[list], optional): divides the input dataset into a training and evaluation set.
                                         If database_eval not None , it is not used.
            
            - neuralnet_arch_list (list[class], required): list of architecture classes of the graph neural network.
                                                If new architecture, make sure to import it.

            - nepoch_list (list[int], optional): list of number of epochs to train. Defaults to [10].
            
            - optimizer_name_list (list[str],optional): list of names of the optimizer to use.
            
            - pretrained_model_list (list[str],optional): list of paths to pre-trained model. Defaults to [None].
            
            ### Other parameters :
            
            - target (str, optional): Which score key to use to train the model and backpropagate. 
                                      Default to scoring_normalized.
            
            - task (str, optional): 'reg' for regression or 'class' for classification. Default to reg.
            
            - shuffle (bool, optional): whether or not to shuffle the training data. Default to True.
                                         
            - threshold (int, optional): threshold to compute binary classification metrics. 
                                         Not used for regression. Defaults to 10.
            
            - list_complexes (list,optional): list of names with the complexes to train on (one subfolder per complex in 
                                              dir_path). If not specified, we test all complexes from dir_path.
                                                                        
            - database_eval (list,optional): independent list of complex names(one subfolder per complex in 
                                              dir_path). Defaults to None.                                   
                                              
            - outdir (str, required): output directory for the train results. 
            
            - save_model (last, best, optional): save the model. Defaults to 'best' based on validation loss.
            
            - save_epoch (all, intermediate, optional): Defaults to 'intermediate'.
            
            - save_every (int, optional): save data every n epoch if save_epoch == 'intermediate'. Defaults to 5.
            
            - plot_every (int, optional): plot mse loss data every n epoch . Defaults to 5.
            
            - plot_name (str,optional): name of the loss plot.
            
            - keep_latest_saved_models (bool,optional): how many latest saved models .pth.tar to keep in order to remove 
                                                        the old ones. Defaults to the 5 latest.

            - keep_latest_saved_plots (bool,optional): how many latest saved plots .png to keep in order to remove 
                                                        the old ones. Defaults to the 5 latest.
                                                        
            - gridsearch_index (int,optional): index so that when we resume gridsearch training it is not from scratch.
                                               Defaults to 0.
    """

    # 2 params not here from training_graph : epoch_start and train_using_pretrained_model
    # CAREFUL : Make sure for transfer learning that pretrain model checkpoint have same architecture and optimizer
    count = gridsearch_index
    for pretrained_model in pretrained_model_list:
        for dir_path in dir_path_list:
            for edge_feature in edge_feature_list:
                for node_feature in node_feature_list:
                    for batch_size in batch_size_list:
                        for lr in lr_list:
                            for percent in percent_list:
                                for neuralnet_arch in neuralnet_arch_list :
                                    for nepoch in nepoch_list:
                                        for optimizer_name in optimizer_name_list:  
                                            count+=1
                                            try:
                                                # train_using_pretrained_model False if pretrained_model is None
                                                train_using_pretrained_model= pretrained_model is not None
                                                outdir_gridsearch = outdir + str(count) +"/"    
                                                training_graph(dir_path=dir_path,
                                                               edge_feature=edge_feature,
                                                               node_feature=node_feature,
                                                               batch_size=batch_size,
                                                               target=target,
                                                               task=task,
                                                               shuffle=shuffle,
                                                               lr=lr,
                                                               percent=percent,
                                                               threshold=threshold,
                                                               list_complexes=list_complexes,
                                                               database_eval=database_eval,
                                                               outdir=outdir_gridsearch,
                                                               neuralnet_arch=neuralnet_arch,
                                                               nepoch=nepoch,
                                                               save_model=save_model,
                                                               save_epoch=save_epoch,
                                                               save_every=save_every,
                                                               plot_every=plot_every,
                                                               plot_name=plot_name,
                                                               optimizer_name=optimizer_name,
                                                               pretrained_model=pretrained_model,
                                                               train_using_pretrained_model=train_using_pretrained_model,
                                                               epoch_start=0,#not for resume training
                                                               keep_latest_saved_plots=keep_latest_saved_plots,
                                                               keep_latest_saved_models=keep_latest_saved_models)

                                                print("Training over for gridsearch "+ str(count))
                                            except Exception as e:
                                                print(e)
                                                print("Training didn't work for this combination (likely due to error \
                                                      with transfer learning vs parameter combination)")


import random
import os
import json
import shutil
from shutil import copyfile
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from deeprank_gnn.GraphGenMP import GraphHDF5
import deeprank_gnn.CustomizeGraph as CustomizeGraph

import warnings
warnings.filterwarnings("ignore")


def create_dict_selected_pdbs(dir_path="",
                              seed=912,
                              threshold_irmsd=10,
                              min_number_pdb_per_stage=10,
                              list_complexes = "",
                              txt_file_name="target.txt",
                              save=False,
                              max_val=20): 
    

    """
    Select pdbs based on those criterias and create a target.txt files with the name of the selected pdbs and their 
    corresponding i-RMSD for each complex and for each stage.
    Creates inside each complex folder , inside each stage,  a dictionary with selected pdbs with corresponding i-RMSD.
    In summary , we have now 3 dictionaries for each complex (for water, it0 and it1) with the selected pdb files based
    on the threshold+sampling explained above .
    An example of dict with complex 1JTD it1 :({'1JTD_it1_complex_329.pdb': 14.98,
    '1JTD_it1_complex_359.pdb': 11.385})


    Outputs :
    1. one dictionary per stage per complex with decoy name + corresponding i-RMSD

    Args:

        - dir_path (str, required): path(s) to initial pdb files

        - seed (int, optional): for reproducibility of pdb selection

        - threshold_irmsd (float, required): create a balance dataset around the threshold

        - min_number_pdb_per_stage (int, required): If balance dataset is not created select randomly 

        - list_complexes (list,optional): list of names with the complexes we want to prepare the data
                                          (one subfolder per complex in dir_path).
                                          If not specified, we take all complexes beside input_complexes from dir_path.

        - txt_file_name (str, required): name of the dictionary to be created

        - save (boolean, required): whether or not to save the dictionary

        - max_val (float, required): maximal i-RMSD value for a pdb to be considered
    """
    
    
    if(list_complexes==""): # if not specified get all complexes from dir_path
        list_complexes = os.listdir(dir_path)
        list_complexes = [ l for l in list_complexes if ("." not in l and "input_complexes" not in l)]
        

    count = 0 

    for l in list_complexes:
        
        for sub in ["water","it0","it1"] : # 3 complexes subfolders

            datContent = [i.strip().split() for i in open(dir_path+l+"/"+sub+"/"+"i-RMSD.dat").readlines()]
            itemDict_pos = {item[0]: float(item[1])  for item in datContent[1:len(datContent)] if float(item[1]) <= threshold_irmsd} # we start from indice 1 because indice 0 is not a complex
            itemDict_neg = {item[0]: float(item[1])  for item in datContent[1:len(datContent)] if ((float(item[1]) > threshold_irmsd) & (float(item[1]) <= max_val))}

            if(len(itemDict_neg)!=0 and len(itemDict_pos)!=0 and len(itemDict_neg)>=len(itemDict_pos)):
                random.seed(seed)
                print("create dict balanced (more negatives) (sample without replacement from negatives to match positives)")
                keys = random.sample(list(itemDict_neg), max(len(itemDict_pos),min_number_pdb_per_stage)) # max(min) to handle if other way around and at least have min_number_pdb_per_stage samples per subcomplex
                itemDict_neg = {keys[i]:itemDict_neg[keys[i]] for i in range(len(keys))}
                itemDict_combined = {**itemDict_neg, **itemDict_pos}
                print(l,sub,len(itemDict_neg),len(itemDict_pos))
                prefix = l+"_"+sub+"_"
                itemDict_combined = {prefix + str(key): val for key, val in itemDict_combined.items()}
                count+=len(itemDict_combined)
                if save:
                    json.dump(itemDict_combined, open(dir_path+l+"/"+sub+"/"+txt_file_name,'w'))

            elif(len(itemDict_neg)!=0 and len(itemDict_pos)!=0 and len(itemDict_neg)<len(itemDict_pos)):
                random.seed(seed)
                print("create dict balanced (more positives) (sample without replacement from positives to match negatives)")
                keys = random.sample(list(itemDict_pos), max(len(itemDict_neg),min_number_pdb_per_stage)) # max(min) to handle if other way around and at least have min_number_pdb_per_stage samples per subcomplex
                itemDict_pos = {keys[i]:itemDict_pos[keys[i]] for i in range(len(keys))}
                itemDict_combined = {**itemDict_neg, **itemDict_pos}
                print(l,sub,len(itemDict_neg),len(itemDict_pos))
                prefix = l+"_"+sub+"_"
                itemDict_combined = {prefix + str(key): val for key, val in itemDict_combined.items()}
                count+=len(itemDict_combined)
                if save:
                    json.dump(itemDict_combined, open(dir_path+l+"/"+sub+"/"+txt_file_name,'w'))

            elif(len(itemDict_neg)==0 and len(itemDict_pos)!=0):
                random.seed(seed)
                print("create dict with only "+str(min_number_pdb_per_stage)+ " randomly selected positive examples")
                keys = random.sample(list(itemDict_pos),min_number_pdb_per_stage) # max(min) to handle if other way around and at least have min_number_pdb_per_stage samples per subcomplex
                itemDict_pos = {keys[i]:itemDict_pos[keys[i]] for i in range(len(keys))}
                itemDict_combined = {**itemDict_pos}
                print(l,sub,len(itemDict_neg),len(itemDict_pos))
                prefix = l+"_"+sub+"_"
                itemDict_combined = {prefix + str(key): val for key, val in itemDict_combined.items()}
                count+=len(itemDict_combined)
                if save:
                    json.dump(itemDict_combined, open(dir_path+l+"/"+sub+"/"+txt_file_name,'w'))

            elif(len(itemDict_neg)!=0 and len(itemDict_pos)==0):
                random.seed(seed)
                print("create dict with only "+str(min_number_pdb_per_stage)+ " randomly selected negative examples")
                keys = random.sample(list(itemDict_neg),min_number_pdb_per_stage) # max(min) to handle if other way around and at least have 10 samples per subcomplex
                itemDict_neg = {keys[i]:itemDict_neg[keys[i]] for i in range(len(keys))}
                itemDict_combined = {**itemDict_neg}
                print(l,sub,len(itemDict_neg),len(itemDict_pos))
                prefix = l+"_"+sub+"_"
                itemDict_combined = {prefix + str(key): val for key, val in itemDict_combined.items()}
                count+=len(itemDict_combined)
                if save:
                    json.dump(itemDict_combined, open(dir_path+l+"/"+sub+"/"+txt_file_name,'w'))
            else :
                print(l,sub,"No dictionary because i-RMSD is corrupted !!!!!!!!!!!!!!!")
            
            print(min([val for key,val in itemDict_combined.items()]),max([val for key,val in itemDict_combined.items()]))

    print("Total number of pdbs we use :", count)



def prepare_pdb_selected_data(parent_dir="",
                              dir_path="",
                              list_complexes="",
                              txt_file_name="target.txt",
                              add_input_complex=False) :

    """
    In the folder parent_dir,it creates as many subfolders as complex names with all selected pdb files from dictionaries             corresponding to txt_file_name.

    Outputs :
    1. one folder per selected complex with the pdb files associated to txt_file_name

    Args:

        - parent_dir (str, required): path to where to make a copy of selected pdbs

        - dir_path (str, required): path(s) to initial pdb files

        - list_complexes (list,optional): list of names with the complexes we want to prepare the data
                                          (one subfolder per complex in dir_path).
                                          If not specified, we take all complexes beside input_complexes from dir_path.

        - txt_file_name (str, required): name of the dictionary so that we only copy the corresponding pdbs

        - add_input_complex (boolean, required): whether or not to add input complexes to the selected pdbs (it will have
                                                 i-RMSD 0).

    """
    
    if(list_complexes==""): # if not specified get all complexes from dir_path
        list_complexes = os.listdir(dir_path)
        list_complexes = [ l for l in list_complexes if ("." not in l and "input_complexes" not in l)]
        
    for l in list_complexes :


        path = os.path.join(parent_dir, l)
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print(path + " exists")

        for sub in ["water","it0","it1"]:
            print(l,sub,"\n")
            if not os.path.exists(dir_path+l+"/"+sub+"/"+txt_file_name) :
                print(l,sub,"is empty ")
            else :
                with open(dir_path+l+"/"+sub+"/"+txt_file_name) as dict_target:
                    list_target = json.loads(dict_target.read())
                list_target = list(list_target.keys())
                print (len(list_target))
                for target in list_target:
                    src = dir_path+l+"/"+sub+"/"+target.split("_")[-2]+"_"+target.split("_")[-1]
                    dst = parent_dir+l+"/"+target
                    try : 
                        copyfile(src, dst)
                    except :
                        print(target, "was not found. It is weird !")
        if(add_input_complex):
            src = dir_path+"input_complexes/"+l+".pdb"
            dst = parent_dir+l+"/"+l+".pdb"
            try : 
                copyfile(src, dst)
            except :
                print(l, "input complex not found")
            
                
                        

def prepare_hdf5_graphs(dir_path="",
                       pdb_data_dir="",
                       graph_dir="",
                       list_complexes="", 
                       min_normalized = 0,
                       max_normalized = 20,
                       add_target=True,
                       add_normalized_target = True,
                       nproc=1,
                       tmpdir="pkl/",
                       txt_file_name="target.txt",
                       add_input_complex=False,
                       contact_distance=8.5,
                       internal_contact_distance=3) :


    """
        Create one graph per complex with target + normalized target values and input complex if add_input_complex is True.

        For each complex:
            - Read pdb files in selected pdb files for that specific complex
            - Create a .hdf5 graph with GraphHDF5 in graph path folder/complex each complex
            - Create and save target_list.txt file and target_normalized_list.txt with specific format ["complex_name score"]
              combining the target files for water,it0,it1 (+ reference complex(score=0))
            - Add target score for each graph using CustomizeGraph and normalized target if normalized is True
 
        Outputs :
        1. one folder per selected complex with the .hdf5 graph associated in the graph_dir path.
    
        Args:
        
            - dir_path (str, required): path(s) to initial pdb files
            
            - pdb_data_dir (str, required): path(s) to initial pdb files
            
            - graph_dir (str, required): path(s) to initial pdb files
            
            - list_complexes (list,optional): list of names with the complexes we want to prepare the data
                                              (one subfolder per complex in dir_path).
                                              If not specified, we take all complexes beside input_complexes from dir_path.
                                              
            - min_normalized (float, optional): Min value used for normalization. Defaults to 0.
            
            - max_normalized (float, optional): Min value used for normalization. Defaults to 20.
            
            - add_target (float,optional): Whether or not to add the ouput in the graph. Defaults to True.
            
            - add_normalized_target (float,optional): Whether or not to add the normalized ouput in the graph. Defaults to True.
                       
            - nproc (int, optional): number of processors. Default to 1.

            - tmpdir (str, optional): Default to "pkl/".
            
            - txt_file_name (str, required): name of the dictionary to add the target to the graph
            
            - add_input_complex (boolean, required): whether or not to add input complexes to the graphs (it will have
                                                     i-RMSD 0).
                                                     
            - contact_distance (float, optional): Contact distance to create nodes of the graph. 
                                                  Useful for new pdb (input complexes if the default cutoff is too low
                                                  and thus, graph is empty). Defaults to 8.5.
            
            - internal_contact_distance (float, optional): Contact distance to create edges of the graph. 
                                                  Useful for new pdb (input complexes if the default cutoff is too low
                                                  and thus, graph does not have edges). Defaults to 3.   
            
    """


    if(list_complexes==""): # if not specified get all complexes from dir_path
        list_complexes = os.listdir(pdb_data_dir)
        list_complexes = [ l for l in list_complexes if "." not in l]


    for l in list_complexes :
        print(l)
        pdb_path = pdb_data_dir+l+"/"


        graph_path = graph_dir+l+"/"

        if not os.path.exists(graph_path):
            os.makedirs(graph_path)
        else:
            print(graph_path + " exists")

        graph = GraphHDF5(pdb_path=pdb_path,
                 graph_type='residue', outfile=graph_path+"residues.hdf5", nproc=nproc,tmpdir=tmpdir,
                 contact_distance=contact_distance, internal_contact_distance=internal_contact_distance)
        

        if(add_target):
            # Add target file in each complex folder with all target labels
            target_list = []
            for sub in ["water","it0","it1"]:
                try:
                    with open(dir_path+l+"/"+sub+"/"+txt_file_name) as dict_target:
                        tmp = json.loads(dict_target.read())   
                        tmp = [k[:-4]+" "+str(v) for k,v in tmp.items()]        
                    target_list.extend(tmp)   
                except:
                    print("There is no dictionary associated for", l,sub)
            
            if add_input_complex:
                target_list.extend([l+" 0"])
            np.savetxt(graph_path+"target_list.txt",target_list, fmt='%s')

            CustomizeGraph.add_target(graph_path=graph_path, target_name='scoring',
                                  target_list=graph_path+"target_list.txt", sep=' ') 
            
        if(add_normalized_target):
            # Add normalized target file in each complex folder with all target labels
            target_list_normalized = [] 
            for sub in ["water","it0","it1"]:
                try:
                    with open(dir_path+l+"/"+sub+"/"+txt_file_name) as dict_target:
                        tmp_normalized = json.loads(dict_target.read())
                        for k in tmp_normalized:
                            tmp_normalized[k] = (tmp_normalized[k]-min_normalized)/(max_normalized-min_normalized)    
                        tmp_normalized = [k[:-4]+" "+str(v) for k,v in tmp_normalized.items()]
                    target_list_normalized.extend(tmp_normalized)
                except:
                    print("There is no dictionary associated for", l,sub)
                    
            if add_input_complex:
                target_list_normalized.extend([l+" 0"])
            np.savetxt(graph_path+"target_list_normalized.txt",target_list_normalized, fmt='%s')

            CustomizeGraph.add_target(graph_path=graph_path, target_name='scoring_normalized',
                                  target_list=graph_path+"target_list_normalized.txt", sep=' ')


            
            
            
            

def pdb_cif_to_pandas_dataframe(path='/projects2/common/DATA/alphafold_data/pdb_mmcif/mmcif_files/1jtg.cif'):
    """
        Convert any .cif file to a pandas format
 
        Outputs :
        1. one pandas data frame
    
        Args:
        
            - path to .cif file that needs to be converted
    """            
    pdb = pd.read_fwf(path)
    col_names = ['group_PDB', 'id', 'element', 'name', 'label_alt_id', 'resname', 'chain',
             'label_entity_id', 'label_seq_id', 'pdbx_PDB_ins_code','x','y','z', 'occupancy', 'tempfactor',
             'charge', 'resseq','auth_comp_id','auth_asym_id','auth_atom_id','pdbx_PDB_model_num']

    pdb_atom = pd.DataFrame(columns=col_names)

    to_convert = pdb[pdb.iloc[:,0].str[0:4]=="ATOM"].iloc[:,0].tolist()
    for i in range(len(to_convert)):
        tmp_str=to_convert[i]
        tmp_str = tmp_str.split(" ")
        list_str = [i for i in tmp_str if i!=""]
        list_str = list_str if len(list_str)==len(col_names) else list_str.append([""])
        pdb_atom.loc[i] = np.array(list_str)
        pdb_atom.drop(['auth_comp_id', 'auth_asym_id','auth_atom_id'], axis=1)

    pdb_atom["id"] = pd.to_numeric(pdb_atom["id"]) ## Make sure we convert to numeric

    return pdb_atom

def create_labeled_datas_positives(dataset_names=[""],dist_threshold = 6,
                         folder_name=""):
    # Return as many dataframes as combination of chains (chainA_B; chainC_E ; ...) with all the elements that have distance less than threshold
    list_name = []
    for count in range(len(dataset_names)):   ## Iterate over the pdb file names from dips or db5
        path= dataset_names[count]  ## Get path name
        if(not(dataset_names[count] in list_name)): ## To make sure we don't repeat twice with same name (to gain some time)
            list_name.append(dataset_names[count])
            print(path)
            pdb_atom = pdb_cif_to_pandas_dataframe(path)
           # Dataframe created

            list_chains = sorted(pdb_atom.chain.unique()) ## Sort the chains so that
                                                          ## it is easy to compare distances between neighboring chains
            if(len(list_chains)>1): ## Make sure more than one chain
                for i in range(len(list_chains)-1):
                    for k in range(i+1,len(list_chains)):
                        print(list_chains[i],list_chains[k])
                        d1 = pdb_atom[pdb_atom.chain==list_chains[i]]
                        d2 = pdb_atom[pdb_atom.chain==list_chains[k]]


                        ## Compute distance between all elements of chain 1 with all elements of chain 2 . It gives us a matrix
                        ## with element of chain 1 in rows and element of chain 2 in columns with the distance
                        dist_df = pd.DataFrame(cdist(d1.loc[:,["x","y","z"]], d2.loc[:,["x","y","z"]], metric='euclidean'))
                        dist_df.set_axis(d1.id.values,axis=0,inplace=True)
                        dist_df.set_axis(d2.id.values,axis=1,inplace=True)


                        chain1_ids = np.where(dist_df<dist_threshold)[0] ## Get identifier for chain 1 where threshold is met
                        chain2_ids = np.where(dist_df<dist_threshold)[1] ## Get identifier for chain 2 where threshold is met

                        if(len(chain1_ids)>0): ## Make sure there is contact otherwise move on to next chain pair
                            # Create empty dataframe to store atoms pairs neighbors and their features
                            df_interaction = pd.DataFrame(columns=["id1","name1","resname1","chain1","resseq1",
                                                                   "x1","y1","z1","occupancy1","tempfactor1","element1",
                                                                   "label_alt_id1","label_entity_id1", "label_seq_id1",
                                                                   "pdbx_PDB_ins_code1","pdbx_PDB_model_num1",
                                                                   "id2","name2","resname2","chain2","resseq2",
                                                                   "x2","y2","z2","occupancy2","tempfactor2","element2",
                                                                   "label_alt_id2","label_entity_id2", "label_seq_id2",
                                                                   "pdbx_PDB_ins_code2","pdbx_PDB_model_num2"])

                            # One dataframe per two neighboring chains
                            for j in range(len(chain1_ids)):
                                l1 = d1[["id","name",
                                "resname","chain","resseq",
                                "x","y","z","occupancy","tempfactor","element",
                                "label_alt_id","label_entity_id", "label_seq_id",
                                "pdbx_PDB_ins_code","pdbx_PDB_model_num"]][d1.id==dist_df.index[chain1_ids[j]]].values.tolist()[0]
                                l2 = d2[["id","name",
                                "resname","chain","resseq",
                                "x","y","z","occupancy","tempfactor","element",
                                "label_alt_id","label_entity_id", "label_seq_id",
                                "pdbx_PDB_ins_code","pdbx_PDB_model_num"]][d2.id==dist_df.columns[chain2_ids[j]]].values.tolist()[0]


                                df_interaction.loc[j] = np.append(l1,l2).tolist()

                            file_path = folder_name+dataset_names[count].split("/")[-1][:-4]+"/"   #[:-4] to remove .cif
                            directory = os.path.dirname(file_path)
                            print(directory+"/chain"+list_chains[i]+"_"+list_chains[k]+".csv")
                            try:
                                os.stat(directory)
                            except:
                                os.mkdir(directory)
                            df_interaction.to_csv(directory+"/chain"+list_chains[i]+"_"+list_chains[k]+".csv")

def create_labeled_datas_negatives(dataset_names=[""],dist_threshold = 6,
                                   folder_name=""):
    
    # Return as many dataframes as combination of chains (chainA_B; chainC_E ; ...) with all the elements that have distance greater than threshold and balance it with positives
    list_name = []
    for count in range(len(dataset_names)):   ## Iterate over the pdb file names from dips or db5
        path= dataset_names[count]  ## Get path name
        if(not(dataset_names[count] in list_name)): ## To make sure we don't repeat twice with same name (to gain some time)
            list_name.append(dataset_names[count])
            print(path)
            pdb_atom = pdb_cif_to_pandas_dataframe(path)
           # Dataframe created

            list_chains = sorted(pdb_atom.chain.unique()) ## Sort the chains so that
                                                          ## it is easy to compare distances between neighboring chains
            if(len(list_chains)>1): ## Make sure more than one chain
                for i in range(len(list_chains)-1):
                    for k in range(i+1,len(list_chains)):
                        d1 = pdb_atom[pdb_atom.chain==list_chains[i]]
                        d2 = pdb_atom[pdb_atom.chain==list_chains[k]]


                        ## Compute distance between all elements of chain 1 with all elements of chain 2 . It gives us a matrix
                        ## with element of chain 1 in rows and element of chain 2 in columns with the distance
                        dist_df = pd.DataFrame(cdist(d1.loc[:,["x","y","z"]], d2.loc[:,["x","y","z"]], metric='euclidean'))
                        dist_df.set_axis(d1.id.values,axis=0,inplace=True)
                        dist_df.set_axis(d2.id.values,axis=1,inplace=True)


                        chain1_ids = np.where(dist_df>dist_threshold)[0] ## Get identifier for chain 1 where threshold is met
                        chain2_ids = np.where(dist_df>dist_threshold)[1] ## Get identifier for chain 2 where threshold is met

                        if(len(np.where(dist_df<dist_threshold)[0])>0): ## Make sure there is contact otherwise move on to next chain pair
                           
                            np.random.seed(17)
                            len_pos_samples = len(np.where(dist_df<dist_threshold)[0]) # sample same length as positive labels
                            len_neg_samples = len(np.where(dist_df>dist_threshold)[0])  # sample them from all negative labels
                            ids_to_sample = np.random.choice(len_neg_samples,len_pos_samples,replace=False) # sample N= len_pos_samples from all neg samples 

                            chain1_ids = chain1_ids[ids_to_sample]
                            chain2_ids = chain2_ids[ids_to_sample]
                            
                            
                            # Create empty dataframe to store atoms pairs neighbors and their features
                            df_interaction = pd.DataFrame(columns=["id1","name1","resname1","chain1","resseq1",
                                                                   "x1","y1","z1","occupancy1","tempfactor1","element1",
                                                                   "label_alt_id1","label_entity_id1", "label_seq_id1",
                                                                   "pdbx_PDB_ins_code1","pdbx_PDB_model_num1",
                                                                   "id2","name2","resname2","chain2","resseq2",
                                                                   "x2","y2","z2","occupancy2","tempfactor2","element2",
                                                                   "label_alt_id2","label_entity_id2", "label_seq_id2",
                                                                   "pdbx_PDB_ins_code2","pdbx_PDB_model_num2"])

                            # One dataframe per two neighboring chains
                            for j in range(len(chain1_ids)):
                                l1 = d1[["id","name",
                                "resname","chain","resseq",
                                "x","y","z","occupancy","tempfactor","element",
                                "label_alt_id","label_entity_id", "label_seq_id",
                                "pdbx_PDB_ins_code","pdbx_PDB_model_num"]][d1.id==dist_df.index[chain1_ids[j]]].values.tolist()[0]
                                l2 = d2[["id","name",
                                "resname","chain","resseq",
                                "x","y","z","occupancy","tempfactor","element",
                                "label_alt_id","label_entity_id", "label_seq_id",
                                "pdbx_PDB_ins_code","pdbx_PDB_model_num"]][d2.id==dist_df.columns[chain2_ids[j]]].values.tolist()[0]


                                df_interaction.loc[j] = np.append(l1,l2).tolist()

                            file_path = folder_name+dataset_names[count].split("/")[-1][:-4]+"/"   #[:-4] to remove .cif
                            directory = os.path.dirname(file_path)

                            try:
                                os.stat(directory)
                            except:
                                os.mkdir(directory)
                            df_interaction.to_csv(directory+"/chain"+list_chains[i]+"_"+list_chains[k]+".csv")



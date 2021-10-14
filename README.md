# Graph Neural Network + Attention mechanism to predict scoring functions (i-RMSD) for protein complexes and decoys.

## Tutorial for the data preparation, gridsearch training , testing and inference are available in this repository
![alt text](images/graph_intro.png?raw=true "Title")


## 1. Fully Automated data preparation pipeline that creates balanced graph datasets from PDB protein complexes and decoys files
![alt text](images/data_prep.png?raw=true "Title")

## 2. Automated gridsearch for graph neural net architecture selection (Convolution, Node Attention, Edge Attention, Node+Edge Attention, customizable); 
optimizer selection; possibility to train from scratch/resume training/transfer learning; feature selection
![alt text](images/training.png?raw=true "Title")
![alt text](images/gridsearch.png?raw=true "Title")
## 3. Automated testing pipeline that returns summary of the output, predictions and metrics.
![alt text](images/testing.png?raw=true "Title")
## 4. Inference / Scoring pipeline returning the prediction on raw pdb files.
![alt text](images/inference.png?raw=true "Title")

## Conclusion : Precision within 2 A is reached using attention at both the node and the edge level to leverage complex interaction patterns between the nodes. Further training and architectures/hyperparameter exploration are required and might lead to performance improvement.

![alt text](images/architecture_comparison.png?raw=true "Title")
![alt text](images/testing_results.png?raw=true "Title")

## Future work : 
- augmented training dataset
- introduction of more features (pssm,depth,hse)
- hyperparameter gridsearch exploration
- use pretrained model as a feature embedding
- deeper version of Edge + Node attention network
- energy scoring functions

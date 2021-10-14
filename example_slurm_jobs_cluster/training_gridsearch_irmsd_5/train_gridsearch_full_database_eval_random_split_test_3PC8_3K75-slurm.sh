#!/bin/bash
#SBATCH --reservation=advsim
#SBATCH -N 1 # 1 node
#SBATCH -J test #job name
#SBATCH -o testLog_gridsearch_full_database_eval_random_split_EGATNet3.out #output name
#SBATCH -e testLog_gridsearch_full_database_eval_random_split_EGATNet3.err #error file name
#SBATCH -n 1 #number of processes
##SBATCH -c 24 #number of Intel threads per task;cores is half that
#SBATCH -p project #partition name; 4 partitions on Neo: project: drug discovery related projects; gpu: GPU jobs; cpu: CPU jobs; debug: < 1 hour jobs.
#SBATCH --gres=gpu:1 #comment out if GPU is not needed
#SBATCH --qos=maxjobs #partition qos for project queue; each partition has a QOS; project: maxjobs; cpu: cpuonly; gpu: maxjobs; debug: restrained.
#SBATCH --time=240:00:00
#SBATCH --no-requeue
#SBATCH --exclusive
source /projects2/insite/yanis.tazi/anaconda3/bin/activate /projects2/insite/yanis.tazi/anaconda3/envs/graph_predictions
python train_gridsearch_full_database_eval_random_split_test_3PC8_3K75.py

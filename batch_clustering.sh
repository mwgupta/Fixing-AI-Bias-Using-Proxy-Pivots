#!/bin/bash

#job name
#SBATCH -J batch_clustering

#number of nodes
#SBATCH -N 1

#walltime (set to 12 hrs)
#SBATCH -t 12:00:00

#memory size
#SBATCH --mem=128gb

#output file
#SBATCH --output=output/clustering/batch_clustering.log

#SBATCH -n 2
#SBATCH -p dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:4

cd ../
ns=(8)
filenums=(1 2 3 4 5 6 7 8 9 10)


for x in "${filenums[@]}"
do
	for n in "${ns[@]}"
	do
		python fairness-c1-v1.py --filenums $x --n $n
	done
	
done

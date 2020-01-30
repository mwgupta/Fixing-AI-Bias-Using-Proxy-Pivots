#!/bin/bash

#job name
#SBATCH -J liljob

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 3-00:00:00

#memory size
#SBATCH --mem=128gb

#output file
#SBATCH --output=logs/run_related_analogy_gen2.log

#SBATCH -n 2
#SBATCH -p scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:4

cd ../
n_arr=(1)
# n =(2 5 10 15 20 25)
x_arr=(1)

for x in "${x_arr[@]}"
	do
	python fairness-c1.py ()

	
done


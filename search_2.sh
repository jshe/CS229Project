#!/usr/bin/env bash
#SBATCH --time=0-23:0:0
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
source activate pytorch
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

for i in {0.0001,0.00025,0.001};
	do for j in {5,10,20};
        	do for n in {0.0001,0.001,0.01};
			do sbatch runner.sh python main.py -a ES -d ./ES/exp_0 --lr=$i --population_size=$j --sigma=$n
		done;
	done;
done


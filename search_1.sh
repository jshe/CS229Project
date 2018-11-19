#!/usr/bin/env bash
#SBATCH --time=0-23:0:0
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
source activate pytorch
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

for i in {0.0001,0.00025,0.001};
	do for j in {128,256,512};
        	do for n in {0.01,0.02,0.2};
			do sbatch runner.sh python main.py -a PPO -d ./PPO/exp_0 --lr=$i --max_steps=$j --n_updates=3 --batch_size=32 --gamma=0.99 --clip=$n --ent_coeff=0.0
		done;
	done;
done


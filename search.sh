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
			do for k in {3,5};
        			do for l in {16, 32, 64};
       					do for n in {0.9,0.99};
        					do for m in {0.01,0.02,0.2};
							do for o in {0.0,0.001,0.01,0.02,0.2};
								do sbatch runner.sh python main.py -a PPO --lr=$i --max_steps=$j --n_updates=$k --batch_size=$l --gamma=$m --clip=$n --ent_coeff=$o
							done;
						done;
					done;
				done;
			done;
		done;
	done;
done


#!/usr/bin/env bash
#SBATCH --time=1-23:0:0
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
source activate pytorch
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

exec $@


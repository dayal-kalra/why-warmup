#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --time=8:00:00
#SBATCH --job-name=fcn_mech
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out
###SBATCH --mem=10000
#SBATCH --gpus=a100:1
####SBATCH --gpus=a100_1g.5gb:1
#SBATCH --partition=gpu
####SBATCH --partition=scavenger

source ~/.bashrc
conda activate jax

cd $SLURM_SUBMIT_DIR

time srun python3 train_resnet_grads_adam_lobpcg.py --cluster zaratan --abc sp --scale 0.0 --loss_name xent --widening_factor 1 --depth 16 --num_examples 50000 --batch_size 512 --opt_name base_adam --lr_trgt 0.001 --warmup_steps 1

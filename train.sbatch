#!/bin/bash
#SBATCH --job-name=train_resnet
#SBATCH --partition=ENSTA-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:59:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source /home/ensta/ensta-rian.santos/.venv/bin/activate

cd /home/ensta/ensta-rian.santos/ood-neural-collapse-cifar100 || exit 1

python3 train.py --model resnet_cifar --no-stop --best-optim --suffix adam_no_decay_1bs

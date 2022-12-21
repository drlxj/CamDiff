#!/bin/bash
# export PYTHONPATH="${PYTHONPATH}:/cluster/home/denfan/xueluo/stable-diffusion"
source /cluster/home/denfan/venv/diff/bin/activate
# python inpainting_diff.py --indir /cluster/scratch/denfan/TrainDataset/ --outdir /cluster/scratch/denfan/TrainDataset/extend
python inpainting_diff.py --indir /cluster/work/cvl/denfan/Train/ --outdir /cluster/work/cvl/denfan/Train/out
# srun -n 20 --mem-per-cpu=8000 --gres=gpumem:20g -G 1 --time=12:00:00 --pty bash -i
# sbatch -n 20 --mem-per-cpu=8000 --gres=gpumem:20g -G 1 --time=24:00 --wrap="bash inpaint.sh"
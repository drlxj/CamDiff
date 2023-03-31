#!/bin/bash
source /cluster/home/denfan/venv/diff/bin/activate

python inpainting_diff.py --indir TrainDataset --outdir result

#!/bin/bash

# get needed modules
module load gcc
module load python3
source .venv/bin/activate

# make plot
echo "Starting training..."
python3 Code/train_rPPGnet.py --save_folder $EXP_DIR --output_files $OUT_DIR
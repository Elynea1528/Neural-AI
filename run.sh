#!/bin/bash
source /home/elynea/miniconda3/bin/activate mt5linux
export PYTHONPATH="/home/elynea/projects/Neural-AI:$PYTHONPATH"
python main.py
conda deactivate
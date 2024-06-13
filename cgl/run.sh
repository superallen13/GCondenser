#!/bin/bash
# pass dataset and condenser from cli
dataset=$1
condenser=$2

# Check if the dataset is one of the specified options and set experiment_name accordingly
if [[ "$dataset" == "citeseer" || "$dataset" == "cora" || "$dataset" == "pubmed" ]]; then
    experiment_name="${condenser}_a"
else
    experiment_name="${condenser}_b"
fi
file_name="${dataset}_${condenser}"
memory_bank_path="./data/memory_banks/${file_name}.pt"

echo "Dataset: $dataset"
echo "Condenser: $condenser"

if [[ "$condenser" == "whole" ]] || [[ "$condenser" == "random" ]] || [[ "$condenser" == "kcenter" ]]; then
    python cgl/test.py --dataset "$dataset" --condenser $condenser
else
    echo "Hparams: $experiment_name"
    echo "Memory bank path will be saved at: $memory_bank_path"
    # Run training and testing scripts using the constructed experiment name and other parameters
    python cgl/train.py +experiment="$experiment_name" dataset_name="$dataset"
    python cgl/test.py --dataset "$dataset" --condenser "$condenser" --memory_bank_path "$memory_bank_path"
fi

# bash scripts/wiener/submit.sh -c "bash cgl/run.sh products gcond"
# bash scripts/wiener/submit.sh -c "python cgl/test.py --dataset products --mb_type original"

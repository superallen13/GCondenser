#!/bin/bash

# Default settings for the model configuration
backbone="sgc"
selected_dataset=""
budget_index=""
condenser="gcond"  # Default graph condenser

# Processing command line options
while getopts "m:d:b:c:" opt; do
    case $opt in
    m)
        backbone=${OPTARG,,}
        ;;
    d)
        selected_dataset=$OPTARG
        ;;
    b)
        budget_index=$OPTARG
        ;;
    c)
        condenser=${OPTARG,,}  
        ;;
    \?)
        echo "Invalid option: -$OPTARG" >&2
        exit 1
        ;;
    esac
done

# Setting hyperparameter search method based on condenser choice
if [[ "$condenser" == "gcond" || "$condenser" == "doscond" || "$condenser" == "sgdd" || "$condenser" == "gcdm" ]]; then
    hparams_search="adj_feat_optuna"
elif [[ "$condenser" == "gcondx" || "$condenser" == "doscondx" || "$condenser" == "dm" ]]; then
    hparams_search="feat_optuna"
else
    echo "Invalid graph condenser: $condenser. Available condensers: gcond, doscond, gcondx, doscondx, sgdd, gcdm, dm"
    exit 1
fi

# Budget definitions for different datasets as arrays of three levels
declare -A budgets=(
    ["citeseer"]="30 60 120"
    ["cora"]="35 70 140"
    ["pubmed"]="15 30 60"
    ["arxiv"]="90 454 909"
    ["flickr"]="44 223 446"
    ["reddit"]="153 769 1539"
)

# Validating the selected dataset
if [[ -z "${budgets[$selected_dataset]}" ]]; then
    echo "Invalid or missing dataset. Available datasets: citeseer, cora, pubmed, arxiv, flickr, reddit"
    exit 1
fi

# Converting budget_index to an actual budget
budget_values=($({
    read -r
    echo "$REPLY"
} <<<"${budgets[$selected_dataset]}"))
selected_budget="${budget_values[$((budget_index - 1))]}"

# Validating the selected budget index
if [[ -z "$selected_budget" ]]; then
    echo "Invalid or missing budget index. Choose 1, 2, or 3 for: ${budgets[$selected_dataset]}"
    exit 1
fi

# Determine the mode based on the dataset
observe_mode="inductive"
if [[ "$selected_dataset" == "citeseer" || "$selected_dataset" == "cora" ||
    "$selected_dataset" == "pubmed" || "$selected_dataset" == "arxiv" ]]; then
    observe_mode="transductive"
elif [[ "$selected_dataset" == "flickr" || "$selected_dataset" == "reddit" ]]; then
    observe_mode="inductive"
fi

# Confirmation of the settings before execution
echo "Running training with the following settings:"
echo "Graph Condenser: $condenser"
echo "Backbone Model: $backbone"
echo "Dataset: $selected_dataset"
echo "Budget Index: $budget_index -> Budget: $selected_budget"
echo "Observation Mode: $observe_mode"

# Configure model targets based on the backbone choice
if [[ "$backbone" == "sgc" ]]; then
    target_gnn="graph_condenser.models.backbones.sgc.SGC"
    target_val="graph_condenser.models.evaluators.sgc.SGCEvaluator"
    target_test="graph_condenser.models.backbones.sgc.SGC"
elif [[ "$backbone" == "gcn" ]]; then
    target_gnn="graph_condenser.models.backbones.gcn.GCN"
    target_val="graph_condenser.models.backbones.gcn.GCN"
    target_test="graph_condenser.models.backbones.gcn.GCN"
else
    echo "Invalid backbone model: $backbone. Available backbone model: SGC, GCN"
    exit 1
fi

# Executing the Python training script with model configuration
python graph_condenser/train.py \
    dataset.dataset_name=$selected_dataset \
    condenser=$condenser \
    condenser.budget=$selected_budget \
    condenser.observe_mode=$observe_mode \
    condenser.gnn._target_=$target_gnn \
    condenser.gnn_val._target_=$target_val \
    condenser.gnn_test._target_=$target_test \
    hparams_search=$hparams_search

#!/bin/bash

# settings for the model configuration
declare -A valid_backbones=([sgc]=1 [gcn]=1)
declare -A valid_datasets=([citeseer]=1 [cora]=1 [pubmed]=1 [arxiv]=1 [flickr]=1 [reddit]=1, [products]=1)
declare -a valid_budgets=(1 2 3)
declare -A valid_condensers=([gcond]=1 [doscond]=1 [gcondx]=1 [doscondx]=1 [sgdd]=1 [gcdm]=1 [dm]=1 [sfgc]=1)

# command line options
while getopts "m:d:b:c:" opt; do
    case $opt in
    m)
        backbone=${OPTARG,,} # convert to lowercase
        if [[ -z ${valid_backbones[$backbone]} ]]; then
            echo "Invalid backbone model (-m): $OPTARG" >&2
            exit 1
        fi
        ;;
    d)
        selected_dataset=${OPTARG,,} # convert to lowercase
        if [[ -z ${valid_datasets[$selected_dataset]} ]]; then
            echo "Invalid dataset (-d): $OPTARG" >&2
            exit 1
        fi
        ;;
    b)
        budget_index=$OPTARG
        if [[ ! " ${valid_budgets[*]} " =~ " $budget_index " ]]; then
            echo "Invalid budget index (-b): $OPTARG" >&2
            exit 1
        fi
        ;;
    c)
        condenser=${OPTARG,,} # convert to lowercase
        if [[ -z ${valid_condensers[$condenser]} ]]; then
            echo "Invalid condenser (-c): $OPTARG" >&2
            exit 1
        fi
        ;;
    \?)
        echo "Invalid option: -$OPTARG" >&2
        exit 1
        ;;
    esac
done

# Budget definitions for different datasets as arrays of three levels
declare -A budgets=(
    ["citeseer"]="30 60 120"
    ["cora"]="35 70 140"
    ["pubmed"]="15 30 60"
    ["arxiv"]="90 454 909"
    ["flickr"]="44 223 446"
    ["reddit"]="153 769 1539"
    ["products"]="612 1225 2449"
)

# Converting budget_index to an actual budget
budget_values=($({
    read -r
    echo "$REPLY"
} <<<"${budgets[$selected_dataset]}"))
selected_budget="${budget_values[$((budget_index - 1))]}"

# Determine the mode based on the dataset
observe_mode="inductive"
if [[ "$selected_dataset" == "citeseer" || "$selected_dataset" == "cora" ||
    "$selected_dataset" == "pubmed" || "$selected_dataset" == "arxiv" || "$selected_dataset" == "products" ]]; then
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
echo "Hparams Search: ${selected_dataset}_${condenser}_grid"

# Configure model targets based on the backbone choice
if [[ "$backbone" == "sgc" ]]; then
    target_gnn="graph_condenser.models.backbones.sgc.SGC"
    target_val="graph_condenser.models.evaluators.sgc.SGCEvaluator"
    target_test="graph_condenser.models.backbones.sgc.SGC"
elif [[ "$backbone" == "gcn" ]]; then
    target_gnn="graph_condenser.models.backbones.gcn.GCN"
    target_val="graph_condenser.models.backbones.gcn.GCN"
    target_test="graph_condenser.models.backbones.gcn.GCN"
fi

base_command="python graph_condenser/train.py \
    dataset.dataset_name=$selected_dataset \
    condenser=$condenser \
    condenser.budget=$selected_budget \
    condenser.observe_mode=$observe_mode \
    condenser.gnn._target_=$target_gnn \
    condenser.gnn_val._target_=$target_val \
    condenser.gnn_test._target_=$target_test \
    hparams_search=${selected_dataset}_${condenser}_grid \
    logger=wandb \
    logger.wandb.project=GCB-overall \
    logger.wandb.group=$selected_dataset-$observe_mode-$selected_budget-$backbone-$condenser"

# Adding condition specific command
if [ "$condenser" == "sfgc" ]; then
    $base_command condenser.traj_buffer.model=$backbone
elif [ "$selected_dataset" == "products" ]; then
    $base_command condenser.init_method=randomChoice +condenser.batch_training=true
else
    $base_command
fi

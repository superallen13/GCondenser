#!/bin/bash

# settings for the model configuration
declare -A valid_backbones=( [sgc]=1 [gcn]=1 )
declare -A valid_datasets=( [citeseer]=1 [cora]=1 [pubmed]=1 [arxiv]=1 [flickr]=1 [reddit]=1 )
declare -a valid_budgets=(1 2 3)
declare -A valid_condensers=( [gcond]=1 [doscond]=1 [gcondx]=1 [doscondx]=1 [sgdd]=1 [gcdm]=1 [dm]=1 [sfgc]=1 )
declare -A valid_label_distributions=( [original]=1 [balanced]=1 )
declare -A valid_feat_inits=( [randomchoice]=1 [randomnoise]=1 [kcenter]=1 )

# command line options
while getopts "m:d:b:c:y:x:" opt; do
    case $opt in
    m)
        backbone=${OPTARG,,}  # convert to lowercase
        if [[ -z ${valid_backbones[$backbone]} ]]; then
            echo "Invalid backbone model (-m): $OPTARG" >&2
            exit 1
        fi
        ;;
    d)
        selected_dataset=${OPTARG,,}  # convert to lowercase
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
        condenser=${OPTARG,,}  # convert to lowercase
        if [[ -z ${valid_condensers[$condenser]} ]]; then
            echo "Invalid condenser (-c): $OPTARG" >&2
            exit 1
        fi
        ;;
    y)
        label_distribution=${OPTARG,,}  # convert to lowercase
        if [[ -z ${valid_label_distributions[$label_distribution]} ]]; then
            echo "Invalid label distribution (-y): $OPTARG" >&2
            exit 1
        fi
        ;;
    x)
        feat_init=${OPTARG,,}  # convert to lowercase
        if [[ -z ${valid_feat_inits[$feat_init]} ]]; then
            echo "Invalid feature initialization (-x): $OPTARG" >&2
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
)

# convert budget_index to node budget
IFS=' ' read -ra budget_values <<< "${budgets[$selected_dataset]}"
selected_budget="${budget_values[$((budget_index - 1))]}"

# determine the default observation mode based on the dataset
observe_mode="inductive"
if [[ "$selected_dataset" == "citeseer" || "$selected_dataset" == "cora" ||
    "$selected_dataset" == "pubmed" || "$selected_dataset" == "arxiv" ]]; then
    observe_mode="transductive"
elif [[ "$selected_dataset" == "flickr" || "$selected_dataset" == "reddit" ]]; then
    observe_mode="inductive"
fi

# present of the settings before execution
echo "Running training with the following settings:"
echo "Graph Condenser: $condenser"
echo "Backbone Model: $backbone"
echo "Dataset: $selected_dataset"
echo "Node Budget: $selected_budget"
echo "Observation Mode: $observe_mode"
echo "Label Distribution: $label_distribution"
echo "Feature Initialization: $feat_init"

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
    condenser.label_distribution=$label_distribution \
    condenser.init_method=$feat_init \
    hparams_search=init_${selected_dataset}_${condenser} \
    logger=wandb \
    logger.wandb.project=GCB_Init \
    logger.wandb.group=${selected_dataset}_${condenser}_${backbone}_${selected_budget}_${label_distribution}_${feat_init}


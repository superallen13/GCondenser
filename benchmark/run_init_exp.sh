#!/bin/bash

backbone=gcn
selected_dataset=arxiv
budget_index=1

for condenser in dm doscond gcond; do
    for label_distribution in original balanced; do
        for feat_init in randomchoice randomnoise kcenter; do
            echo "Running experiment with backbone=$backbone, dataset=$selected_dataset, budget=$budget_index, condenser=$condenser, label_distribution=$label_distribution, feat_init=$feat_init"
            bash scripts/wiener/submit.sh -c "bash benchmark/initialisation.sh -m $backbone -d $selected_dataset -b $budget_index -c $condenser -y $label_distribution -x $feat_init"
        done
    done
done

#!/bin/bash

log_file="benchmark_log.csv"

if [ ! -f $log_file ]; then
    echo "dataset,budget_index,init_method,seed,output" > $log_file
fi

for data in citeseer cora pubmed; do
    for budget_index in 1 2 3; do
        for init_method in randomChoice kCenter; do
            for seed in 1 2 3 4 5; do
                python ./benchmark/run_sampling.py --dataset $data --budget_index $budget_index --init_method $init_method --seed $seed | while read output; do
                    echo "$data,$budget_index,$init_method,$seed,\"$output\"" >> $log_file
                done
            done
        done
    done
done

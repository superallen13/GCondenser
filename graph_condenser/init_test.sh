declare -A budgets=(
    ["citeseer"]="30 60 120"
    ["cora"]="35 70 140"
    ["pubmed"]="15 30 60"
    ["arxiv"]="90 454 909"
    ["flickr"]="44 223 446"
    ["reddit"]="153 769 1539"
)

for dataset in citeseer cora pubmed arxiv flickr reddit; do
    for sampling_method in randomchoice kcenter; do
        for budget_index in 1 2 3; do
            budget_values=($({
                read -r
                echo "$REPLY"
            } <<<"${budgets[$dataset]}"))
            selected_budget="${budget_values[$((budget_index - 1))]}"

            if [[ "$dataset" == "citeseer" || "$dataset" == "cora" ||
                "$dataset" == "pubmed" || "$dataset" == "arxiv" ]]; then
                observe_mode="transductive"
            elif [[ "$dataset" == "flickr" || "$dataset" == "reddit" ]]; then
                observe_mode="inductive"
            fi
            echo "Running $dataset $sampling_method $selected_budget $observe_mode"
            python graph_condenser/init_test.py \
                --dataset $dataset \
                --observe_mode $observe_mode \
                --budget $selected_budget \
                --sampling_method $sampling_method
        done
    done
done

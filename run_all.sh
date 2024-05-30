for dataset in flickr; do
    for budget in 1 2 3; do
        for backbone in sgc gcn; do
            for condenser in gcond gcondx doscond doscondx sgdd gcdm dm sfgc; do
                bash scripts/wiener/submit.sh -c "bash scripts/experiment.sh -m $backbone -d $dataset -b $budget -c $condenser"
                sleep 5
            done
        done
    done
done

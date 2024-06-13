for dataset in products reddit flickr cora arxiv; do
    for condenser in whole random kcenter gcond gcondx doscond doscondx sgdd gcdm gcdmx dm; do
        bash scripts/wiener/submit.sh -c "bash cgl/run.sh $dataset $condenser"
        sleep 3
    done
done

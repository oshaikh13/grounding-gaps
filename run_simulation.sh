for dataset in "tscc" "persuasion" "esconv"
do
    python src/simulator.py \
        --data-path ./datasets/${dataset}/processed \
        --output-path ./outputs/${dataset}/out_${dataset}.json \
        --model "gpt-3.5-turbo" \
        --limit 100
done


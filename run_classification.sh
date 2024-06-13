

task="combined"

for item in "tscc" "persuasion" "esconv"
do
    echo "RUNNING FOR $item $task" 

    python src/gold_classifier.py \
        --data-path ./datasets/$item/processed \
        --output-path ./outputs/$item/out_gold_test_$task.json \
        --prompt-path ./prompts/${task}_classification.txt \
        --limit 100 \
        --skip-test 20 \
        --rev-test

    python src/gpt_classifier.py \
        --gpt-data-path ./outputs/$item/out_${item}.json \
        --gold-cls-path ./outputs/$item/out_${task}.json \
        --output-path ./outputs/$item/out_text_label.json  \
        --prompt-path ./prompts/${task}_classification.txt \
        --skip-test 20

done

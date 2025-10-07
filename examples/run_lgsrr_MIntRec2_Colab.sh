%%bash

# Define the path to the Python executable 'lgsrr' conda environment
PYTHON_EXEC="/usr/local/envs/lgsrr/bin/python"

run_third_script() {
    for method in 'lgsrr'
    do
        for dataset in 'MIntRec2.0'
        do
            for text_backbone in 'bert-large-uncased'
            do
                # Execute the Python script with all arguments
                $PYTHON_EXEC run.py \
                --dataset $dataset \
                --data_path 'data/data_text' \
                --desc_path 'data/data_desc/mintrec2.0' \
                --rank_path 'data/data_rank/rank_mintrec2_train.tsv' \
                --logger_name ${method} \
                --method ${method} \
                --train \
                --save_results \
                --gpu_id '2' \
                --text_backbone $text_backbone \
                --config_file_name ${method}_MIntRec2 \
                --results_file_name "results_mintrec2_${method}.csv"
            done
        done
    done
}
run_third_script
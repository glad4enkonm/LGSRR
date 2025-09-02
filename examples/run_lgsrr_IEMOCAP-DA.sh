#!/usr/bin/bash

run_third_script() {
    for method in 'lgsrr'
    do
        for dataset in 'IEMOCAP-DA'
        do
            for text_backbone in 'bert-large-uncased'
            do
                python run.py \
                --dataset $dataset \
                --data_path 'data/data_text' \
                --desc_path 'data/data_desc/iemocap_da' \
                --rank_path 'data/data_rank/rank_iemocap_da_train.tsv' \
                --logger_name ${method} \
                --method ${method} \
                --train \
                --save_results \
                --gpu_id '2' \
                --text_backbone $text_backbone \
                --config_file_name ${method}_IEMOCAP-DA \
                --results_file_name "results_iemocap-da_${method}.csv"
            done
        done
    done
}
run_third_script
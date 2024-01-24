#!/bin/bash

export TOKENIZERS_PARALLELISM=false
DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node 4 --nnodes ${SLURM_NNODES} --node_rank ${SLURM_PROCID} --master_addr localhost --master_port 55555"

python ${DISTRIBUTED_ARGS} jpr.py \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --metric_for_best_model reranked-avg \
    --label_names retr_bm25_labels \
    --preprocessing_num_workers 10 \
    --dataloader_num_workers 10 \
    --dataloader_drop_last \
    --has_label True \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --train_bm25_data_path downloads/data/retriever-outputs/bm25/nq-train-50.json \
    --train_dpr_data_path downloads/data/retriever/biencoder-nq-train.json \
    --eval_bm25_data_path downloads/data/retriever-outputs/bm25/nq-dev.json \
    --corpus_path downloads/data/wikipedia-split/psgs_w100.tsv \
    --top_k_passages 24 \
    --gen_top_k_passages 16 \
    --prediction_output_path tmp/pred_train.json \
    --generator_model_name $GEN_MODEL \
    --retriever_model_name $RETR_MODEL \
    --output_dir tmp \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --loss_weights 1,1,0.001 \
    --nce_loss binary \
    --dist_loss kl_div \
    --warmup_ratio 0.1 \
    --load_best_model_at_end True \
    --logging_steps 500  \
    --save_strategy steps \
    --save_steps 1500 \
    --evaluation_strategy steps \
    --eval_steps 1500 \
    --save_total_limit 2 \
    --bf16 True \
    --eval_lmbda 0.5 \
    --do_train \
    --do_eval

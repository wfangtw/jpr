#!/bin/bash

export TOKENIZERS_PARALLELISM=false
DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node 4 --nnodes ${SLURM_NNODES} --node_rank ${SLURM_PROCID} --master_addr localhost --master_port 55555"

python ${DISTRIBUTED_ARGS} joint_train.py \
    --per_device_eval_batch_size 1 \
    --metric_for_best_model reranked-avg \
    --label_names retr_bm25_labels \
    --preprocessing_num_workers 10 \
    --dataloader_num_workers 10 \
    --dataloader_drop_last \
    --has_label True \
    --remove_unused_columns False \
    --corpus_path beir_datasets/nfcorpus/corpus.jsonl \
    --test_bm25_data_path downloads/data/retriever-outputs/bm25/beir/beir-nfcorpus-test.json \
    --top_k_passages 100 \
    --gen_top_k_passages 100 \
    --prediction_output_path tmp/pred_beir.json \
    --generator_model_name $GEN_MODEL \
    --retriever_model_name $RETR_MODEL \
    --output_dir tmp \
    --eval_lmbda 0.5 \
    --do_predict
python evaluate_beir_rerank.py --dataset nfcorpus --beir_data_dir beir_datasets --split test --orig_result_path downloads/data/retriever-outputs/bm25/beir/beir-nfcorpus-test.json --rerank_result_path tmp/pred_beir.json --k_values 10
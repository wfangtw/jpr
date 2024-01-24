JPR: Joint Passage Re-ranking
===
<a id="contents"></a>
# Contents
<!-- MarkdownTOC -->

- [Setup](#setup)
- [Data](#data)
- [Training](#training)
- [Inference](#inference)
- [Citation](#citation)


<!-- /MarkdownTOC -->

This repository contains the implementation of the JPR (Joint Passage Re-ranking) algorithm,
described in the paper "Joint Inference of Retrieval and Generation for Passage
Re-ranking".

<a id="setup"></a>
# Setup
Please install [PyTorch](https://pytorch.org/) and [Pyserini](https://github.com/castorini/pyserini).
Other dependencies are listed in `requirements.txt`.

<a id="data"></a>
# Data
## OpenQA Retrieval
We use 100-word long passages from [DPR](https://arxiv.org/abs/2004.04906),
which can be downloaded by running:
```python
python utils/download_data.py --resource data.wikipedia-split.psgs_w100
```
This evidence file contains tab-separated fields for passage id, passage text, and passage title. 

The top-1000 retrieved passages from BM25 for the dev/test splits of NaturalQuestions-Open 
(NQ) and TriviaQA can be downloaded by running:
```python
python utils/download_data.py \
	--resource data.retriever-outputs.bm25  \
	[optional --output_dir {your location}]
```
Additionally, the original NQ and TriviaQA data are required for training, which can be
downloaded by running:
```bash
python utils/download_data.py \
	--resource data.retriever  \
	[optional --output_dir {your location}]
```
## BEIR
Please follow [BEIR](https://github.com/beir-cellar/beir) to install and
download the resources.
We suggest to create a soft link to BEIR datasets named `beir_datasets`.

Download Pyserini indexes with `utils/download_pyserini.py`, and then run 
`utils/generate_pyserini_input.py` and `evaluate_beir.py` to get BM25 results on BEIR.

<a id="training"></a>
# Training
For training, run:
```
DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node ${NPROC_PER_NODE} --nnodes ${SLURM_NNODES} --node_rank ${SLURM_PROCID} --master_addr ${PARENT} --master_port ${MPORT}"

python ${DISTRIBUTED_ARGS} joint_train.py \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --warmup_ratio 0.1 \
    --metric_for_best_model reranked-avg \
    --label_names retr_bm25_labels \
    --preprocessing_num_workers $NCPU \
    --dataloader_num_workers $NCPU \
    --dataloader_drop_last \
    --save_strategy epoch \
    --has_label True \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --train_bm25_data_path $DATA_PATH \
    --train_dpr_data_path $DPR_DATA_PATH \
    --corpus_path $CORPUS_PATH \
    --eval_bm25_data_path $EVAL_DATA_PATH \
    --top_k_passages $TOP_K \
    --gen_top_k_passages $GEN_TOP_K \
    --generator_model_name $GEN_MODEL \
    --retriever_model_name $RETR_MODEL \
    --output_dir $MODEL_DIR \
    --num_train_epochs $EPOCHS \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --eval_lmbda $LMBDA \
    --loss_weights $LOSS_WEIGHTS \
    --bf16 True \
    --load_best_model_at_end True \
    --logging_steps 500  \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --do_train \
    --do_eval
```
Remove the flag `--bf16` if not training on Ampere or newer machines.

<a id="inference"></a>
# Inference (Re-ranking)
For OpenQA retrieval, run:
```
python ${DISTRIBUTED_ARGS} joint_train.py \
    --per_device_eval_batch_size 1 \
    --metric_for_best_model reranked-avg \
    --label_names retr_bm25_labels \
    --preprocessing_num_workers $NCPU \
    --dataloader_num_workers $NCPU \
    --dataloader_drop_last \
    --has_label True \
    --remove_unused_columns False \
    --corpus_path $CORPUS_PATH \
    --test_bm25_data_path $TEST_DATA_PATH \
    --top_k_passages $TOP_K \
    --gen_top_k_passages $GEN_TOP_K \
    --prediction_output_path $PREDICTION_PATH \
    --generator_model_name $GEN_MODEL \
    --retriever_model_name $RETR_MODEL \
    --output_dir $MODEL_DIR \
    --eval_lmbda $LMBDA \
    --do_predict
```

For BEIR, run the command above and then run `evaluate_beir_after_rerank.py` to call
TREC's official evaluation.

<a id="citation"></a>
# Citation
If you find our method or code useful, please consider citing our paper as:
```
@article{fang-etal-2024-joint,
  title = "Joint Inference of Retrieval and Generation for Passage
Re-ranking",
  author = "Fang, Wei and Chuang, Yung-Sung and Glass, James",
  journal={preprint}
  year = "2024"
}
```

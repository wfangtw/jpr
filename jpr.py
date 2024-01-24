#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import sys
import logging
import math
import random
import os
import json
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union, List, Dict, Tuple
from collections import defaultdict

import numpy as np
import torch
# torch.backends.cuda.matmul.allow_tf32 = True
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase


from data import JointDataset, JointOpenQADataset
from joint_trainer import JointTrainer
from joint_model import JointModel


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.25.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    generator_model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    retriever_model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Teacher model path."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    use_bf16: bool = field(default=False)

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.generator_model_name is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --generator_model_name"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(default=None, metadata={"help": "The name of the dataset to use."})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_data_path: Optional[str] = field(default=None, metadata={"help": "The input training data path."})
    train_bm25_data_path: Optional[str] = field(default=None, metadata={"help": "BM25 Json file"})
    corpus_path: Optional[str] = field(default='downloads/data/wikipedia-split/psgs_w100.tsv', metadata={"help": "Retrieval database"})
    train_dpr_data_path: Optional[str] = field(default=None, metadata={"help": "DPR Json file"})
    train_data_split: Optional[str] = field(default='test', metadata={"help": "data split for training"})
    eval_bm25_data_path: Optional[str] = field(default=None, metadata={"help": "BM25 Json file"})
    test_bm25_data_path: Optional[str] = field(default=None, metadata={"help": "BM25 Json file"})
    prediction_output_path: Optional[str] = field(default=None, metadata={"help": "BM25 Json file"})
    top_k_passages: Optional[int] = field(default=32, metadata={'help': 'only use top-k from bm25 results'})
    gen_top_k_passages: Optional[int] = field(default=8, metadata={'help': 'only use top-k from ground truth results'})
    has_label: bool = field(default=False, metadata={"help": "data split contains label"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
    )


@dataclass
class JointTrainingArguments(Seq2SeqTrainingArguments):
    loss_alpha: Optional[float] = field(
        default=0.0, metadata={"help": "Loss weighting"}
    )
    temperature: Optional[float] = field(
        default=1.0, metadata={"help": "Softmax temperature"}
    )
    loss_weights: Optional[str] = field(
        default=None, metadata={"help": "Loss weightings for [Generation_loss, Retrieval_loss, dist_loss]"}
    )
    nce_loss: Optional[str] = field(
        default='binary', metadata={"help": "binary or rank"}
    )
    dist_loss: Optional[str] = field(
        default='kl_div', metadata={"help": "kl_div, js_div, mse"}
    )
    eval_top_k: Optional[int] = field(
        default=100,
        metadata={"help": 'Eval top k retrieval'}
    )
    report_top_k_accuracies: Optional[str] = field(
        default="1,5,10,20,50,100",
        metadata={"help": 'Report top k retrieval accs'}
    )
    eval_lmbda: Optional[float] = field(
        default=0.5,
        metadata={"help": 'lmbda weighting'}
    )


def calculate_top_k_hits(scores, max_k):
    top_k_hits = [0] * max_k
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits[:max_k]) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
    return top_k_hits


def compute_top_k_recall(answers_list, max_k):
    topk_hits = calculate_top_k_hits(answers_list, max_k=max_k)

    topk_hits = torch.FloatTensor(topk_hits).cuda()
    n_docs = torch.FloatTensor([len(answers_list)]).cuda()
    torch.distributed.all_reduce(topk_hits, torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(n_docs, torch.distributed.ReduceOp.SUM)

    topk_hits /= n_docs
    return topk_hits


def compute_top_k_accs(eval_top_k, report_top_k_accuracies):
    def topk_acc(results: EvalPrediction):
        original_answers_list = []
        reranked_answers_list = []
        for i in range(len(results.predictions)): 
            scores = results.predictions[i, :eval_top_k] # only up to eval top k
            lab = results.label_ids[0][i, :eval_top_k]

            contexts = []
            for j in range(eval_top_k):
                contexts.append({
                    'score': scores[j].item() if j < eval_top_k else -100.,
                    'has_answer': bool(lab[j].item()),
                })

            topk_scores, indexes = torch.topk(torch.Tensor(scores), k=eval_top_k)
            ranked_answers = torch.BoolTensor(lab[:eval_top_k])[indexes]

            original_answers_list.append(lab[:eval_top_k].tolist())
            reranked_answers_list.append(ranked_answers.tolist())

        max_k = report_top_k_accuracies[-1]
        original_accs = compute_top_k_recall(original_answers_list, max_k=max_k)
        reranked_accs = compute_top_k_recall(reranked_answers_list, max_k=max_k)
        metrics = {}
        for k in report_top_k_accuracies:
            metrics['original-top-{:03d}'.format(k)] = original_accs[k - 1] * 100
        for k in report_top_k_accuracies:
            metrics['reranked-top-{:03d}'.format(k)] = reranked_accs[k - 1] * 100
        metrics['reranked-avg'] = sum(reranked_accs[k - 1] for k in report_top_k_accuracies[:3]) * 100 / 3
        return metrics
    return topk_acc


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, JointTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.per_device_train_batch_size = 1
    training_args.report_top_k_accuracies = [int(x) for x in training_args.report_top_k_accuracies.split(',')]

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        gen_config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.generator_model_name:
        gen_config = AutoConfig.from_pretrained(model_args.generator_model_name, **config_kwargs)
    else:
        gen_config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            gen_config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {gen_config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        gen_tokenizer = T5Tokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
        # tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.generator_model_name:
        gen_tokenizer = T5Tokenizer.from_pretrained(model_args.generator_model_name, **tokenizer_kwargs)
        # tokenizer = AutoTokenizer.from_pretrained(model_args.generator_model_name, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.generator_model_name:
        gen_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.generator_model_name,
            from_tf=bool(".ckpt" in model_args.generator_model_name),
            config=gen_config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in gen_model.parameters()).values())
        logger.info(f"Loaded model - Total size={n_params/10**6:.2f}M params")
    else:
        gen_model = AutoModelForSeq2SeqLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    retr_config = AutoConfig.from_pretrained(model_args.retriever_model_name)
    retr_config.num_labels = 1
    retr_tokenizer = AutoTokenizer.from_pretrained(model_args.retriever_model_name)
    retr_model = AutoModelForSequenceClassification.from_pretrained(model_args.retriever_model_name, config=retr_config, ignore_mismatched_sizes=True)
    n_params = sum(dict((p.data_ptr(), p.numel()) for p in retr_model.parameters()).values())
    logger.info(f"Loaded model - Total size={n_params/10**6:.2f}M params")

    training_args.loss_weights = [float(x) for x in training_args.loss_weights.split(',')] if training_args.loss_weights is not None else None
    model = JointModel(
        gen_model, 
        retr_model, 
        gen_config, 
        retr_config, 
        gen_tokenizer,
        retr_tokenizer,
        loss_weights=training_args.loss_weights,
        eval_lmbda=training_args.eval_lmbda,
        nce_loss=training_args.nce_loss,
        dist_loss=training_args.dist_loss,
    )

    # Preprocessing the datasets.
    # First we tokenize all the texts.

    train_dataset = None
    eval_dataset = None
    if training_args.do_train:
        if 'nq' in data_args.train_bm25_data_path or 'trivia' in data_args.train_bm25_data_path:
            train_dataset = JointOpenQADataset(
                data_args.train_bm25_data_path, 
                dpr_json_data_path=data_args.train_dpr_data_path,
                psgs_path=data_args.corpus_path,
                top_k=data_args.top_k_passages,
                max_samples=data_args.max_train_samples,
            )
        else:
            train_dataset = JointDataset(data_args.train_data_path, data_args.train_bm25_data_path, split=data_args.train_data_split, top_k=data_args.top_k_passages)

    if training_args.do_eval:
        if 'nq' in data_args.train_bm25_data_path or 'trivia' in data_args.train_bm25_data_path:
            if train_dataset is not None:
                eval_dataset = JointOpenQADataset(
                    data_args.eval_bm25_data_path, 
                    corpus=train_dataset.corpus,
                    max_samples=data_args.max_eval_samples,
                    top_k=training_args.eval_top_k,
                    for_eval=True,
                )
            else:
                eval_dataset = JointOpenQADataset(
                    data_args.eval_bm25_data_path, 
                    psgs_path=data_args.corpus_path, 
                    max_samples=data_args.max_eval_samples,
                    top_k=training_args.eval_top_k,
                    for_eval=True,
                )
        else:
            raise NotImplementedError

    is_openqa = True
    if training_args.do_predict:
        if 'nq' in data_args.test_bm25_data_path or 'trivia' in data_args.test_bm25_data_path:
            if train_dataset is not None:
                test_dataset = JointOpenQADataset(
                    data_args.test_bm25_data_path, 
                    corpus=train_dataset.corpus,
                    max_samples=data_args.max_eval_samples,
                    top_k=training_args.eval_top_k,
                    for_eval=True,
                )
            elif eval_dataset is not None:
                test_dataset = JointOpenQADataset(
                    data_args.test_bm25_data_path, 
                    corpus=eval_dataset.corpus,
                    max_samples=data_args.max_eval_samples,
                    top_k=training_args.eval_top_k,
                    for_eval=True,
                )
            else:
                test_dataset = JointOpenQADataset(
                    data_args.test_bm25_data_path, 
                    psgs_path=data_args.corpus_path, 
                    max_samples=data_args.max_eval_samples,
                    top_k=training_args.eval_top_k,
                    for_eval=True,
                )
        else:
            test_dataset = JointDataset(
                os.path.dirname(data_args.corpus_path),
                data_args.test_bm25_data_path,
                split='test',
                top_k=training_args.eval_top_k,
            )
            is_openqa = False


    @dataclass
    class JointDataCollator:
        gen_tokenizer: PreTrainedTokenizerBase
        retr_tokenizer: PreTrainedTokenizerBase
        batch_size: int
        gen_batch_size: int
        padding: Union[bool, str, PaddingStrategy] = True
        gen_context_max_length: Optional[int] = 256
        gen_query_max_length: Optional[int] = 64
        retr_max_length: Optional[int] = 320
        pad_to_multiple_of: Optional[int] = 8
        pad_to_max_length: bool = data_args.pad_to_max_length
        use_gold: bool = False
        max_gold_num: Optional[int] = 1
        verbalizer: str = None

        def __call__(self, examples: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
            assert len(examples) == 1
            example = examples[0]
            question = example['question']
            bm25_contexts = []
            bm25_labels = []

            batch_size = len(example['bm25_contexts'])
            for ctx in example['bm25_contexts']: #[:self.batch_size]:
                bm25_contexts.append(
                    '{} {}'.format(ctx['doc_title'], ctx['doc_text']))
                bm25_labels.append(ctx['label'])

            gold_contexts = []
            perm = list(np.random.permutation(len(example['gold_contexts'])))
            for idx in perm[:self.gen_batch_size]:
                gold_doc = example['gold_contexts'][idx]
                gold_contexts.append('{} {}'.format(gold_doc['doc_title'], gold_doc['doc_text']))

            retr_bm25_inputs = self.retr_tokenizer(
                [question for _ in range(batch_size)], 
                bm25_contexts, 
                max_length=self.retr_max_length,
                padding=True, 
                truncation='longest_first',
                return_token_type_ids=True,
                return_tensors='pt')

            retr_bm25_labels = torch.Tensor(bm25_labels) # floattensor since reranker has single-dim output (bceloss)

            if self.verbalizer is not None:
                gen_contexts = [f"{self.verbalizer} {ctx}" for ctx in bm25_contexts]
            else:
                gen_contexts = bm25_contexts

            gen_bm25_ctxs = self.gen_tokenizer(
                gen_contexts,
                padding='longest',
                max_length=self.gen_context_max_length,
                truncation=True,
                return_tensors='pt')

            if len(gold_contexts) > 0:
                gen_gold_ctxs = self.gen_tokenizer(
                    gold_contexts,
                    padding='longest',
                    max_length=self.gen_context_max_length,
                    truncation=True,
                    return_tensors='pt')
            else:
                gen_gold_ctxs = {'input_ids': None, 'attention_mask': None}

            gen_decoder_ids = self.gen_tokenizer(
                [question for _ in range(batch_size)], 
                padding='longest',
                max_length=self.gen_query_max_length,
                truncation=True,
                return_tensors='pt').input_ids

            gen_decoder_ids[gen_decoder_ids == self.gen_tokenizer.pad_token_id] = -100

            batch = {}
            for k, v in retr_bm25_inputs.items():
                batch['retr_bm25_inputs_'+k] = v

            batch['retr_bm25_labels'] = retr_bm25_labels.unsqueeze(0)

            for k, v in gen_bm25_ctxs.items():
                batch['gen_bm25_ctxs_'+k] = v

            for k, v in gen_gold_ctxs.items():
                batch['gen_gold_ctxs_'+k] = v

            batch['gen_decoder_ids'] = gen_decoder_ids
            batch['question'] = question

            return batch

    data_collator_cls = JointDataCollator

    # Initialize our Trainer
    trainer = JointTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=gen_tokenizer,
        retr_tokenizer=retr_tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator_cls(
            gen_tokenizer, 
            retr_tokenizer, 
            batch_size=data_args.top_k_passages, 
            gen_batch_size=data_args.gen_top_k_passages, 
            use_gold=data_args.has_label,
            verbalizer="Passage:" if not is_openqa else None),
        compute_metrics=compute_top_k_accs(training_args.eval_top_k, training_args.report_top_k_accuracies),
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        best_ckpt_path = trainer.state.best_model_checkpoint
        print(best_ckpt_path)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Prediction ***")
        logits, labels, metrics = trainer.predict(test_dataset)
        metrics["test_samples"] = len(test_dataset)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if data_args.prediction_output_path is not None:
            save_predictions(test_dataset, logits, data_args.prediction_output_path)


def save_predictions(dataset, all_logits, output_path):
    results = []
    for inst, logits in zip(dataset, all_logits):
        sample = {}
        if 'q_id' in inst:
            sample['qid'] = inst['q_id']
        sample['question'] = inst['question']
        sample['answers'] = ['dummy']
        ctxs = []
        for k, ctx in enumerate(inst['bm25_contexts']):
            if k >= 1001:
                break
            score = logits[k]
            ctxs.append({
                'score': str(score),
                'has_answer': bool(ctx['label']),
                'id': ctx['doc_id'],
            })
        ctxs = sorted(ctxs, reverse=True, key=lambda x: float(x['score']))
        for k, ctx in enumerate(ctxs, 1):
            ctx['rank'] = k

        sample['ctxs'] = ctxs
        results.append(sample)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from transformers import Trainer, __version__
from transformers import AdamW, get_polynomial_decay_schedule_with_warmup, get_constant_schedule
from transformers.utils import logging
from transformers.utils.import_utils import (
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tensorrt_fx_available,
    is_torch_tpu_available,
    is_torchdynamo_available,
)
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import ShardedDDPOption, PredictionOutput, EvalPrediction
from transformers.training_args import TrainingArguments
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.configuration_utils import PretrainedConfig
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    nested_detach,
)
from transformers.trainer_callback import TrainerCallback


logger = logging.get_logger(__name__)
# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
CONFIG_NAME = "config.json"

class JointTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        retr_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset,
                         tokenizer, model_init, compute_metrics, callbacks,
                         optimizers, preprocess_logits_for_metrics)
        self.retr_tokenizer = retr_tokenizer

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        optimizer = self.optimizer
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = ['past_key_values', 'encoder_last_hidden_state']

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            gen_model = self.model.gen_model
            retr_model = self.model.retr_model

        gen_ckpt_path = os.path.join(resume_from_checkpoint, 'gen')
        retr_ckpt_path = os.path.join(resume_from_checkpoint, 'retr')
        if (not os.path.isfile(os.path.join(gen_ckpt_path, WEIGHTS_NAME)) and 
            not os.path.isfile(os.path.join(gen_ckpt_path, WEIGHTS_INDEX_NAME)) and
            not os.path.isfile(os.path.join(retr_ckpt_path, WEIGHTS_NAME)) and 
            not os.path.isfile(os.path.join(retr_ckpt_path, WEIGHTS_INDEX_NAME))):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(os.path.join(gen_ckpt_path, CONFIG_NAME)):
            gen_config = PretrainedConfig.from_json_file(os.path.join(gen_ckpt_path, CONFIG_NAME))
            checkpoint_version = gen_config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if self.args.deepspeed:
            # will be resumed in deepspeed_init
            pass
        elif os.path.isfile(os.path.join(gen_ckpt_path, WEIGHTS_NAME)) and os.path.isfile(os.path.join(retr_ckpt_path, WEIGHTS_NAME)):
            # If the model is on the GPU, it still works!
            if is_sagemaker_mp_enabled():
                raise NotImplementedError
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                gen_state_dict = torch.load(os.path.join(gen_ckpt_path, WEIGHTS_NAME), map_location="cpu")
                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                gen_load_result = gen_model.load_state_dict(gen_state_dict, False)
                # release memory
                del gen_state_dict
                self._issue_warnings_after_load(gen_load_result)

                retr_state_dict = torch.load(os.path.join(retr_ckpt_path, WEIGHTS_NAME), map_location="cpu")
                retr_load_result = retr_model.load_state_dict(retr_state_dict, False)
                # release memory
                del retr_state_dict
                self._issue_warnings_after_load(retr_load_result)
        else:
            raise NotImplementedError

    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        gen_best_model_path = os.path.join(self.state.best_model_checkpoint, 'gen', WEIGHTS_NAME)
        retr_best_model_path = os.path.join(self.state.best_model_checkpoint, 'retr', WEIGHTS_NAME)
        model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        gen_model = model.gen_model
        retr_model = model.retr_model

        if os.path.exists(gen_best_model_path) and os.path.exists(retr_best_model_path):
            if self.deepspeed:
                raise NotImplementedError
            else:
                if is_sagemaker_mp_enabled():
                    raise NotImplementedError
                else:
                    # We load the model state dict on the CPU to avoid an OOM error.
                    gen_state_dict = torch.load(gen_best_model_path, map_location="cpu")
                    # If the model is on the GPU, it still works!
                    # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                    # which takes *args instead of **kwargs
                    gen_load_result = gen_model.load_state_dict(gen_state_dict, False)
                    self._issue_warnings_after_load(gen_load_result)

                    retr_state_dict = torch.load(retr_best_model_path, map_location="cpu")
                    retr_load_result = retr_model.load_state_dict(retr_state_dict, False)
                    self._issue_warnings_after_load(retr_load_result)

        elif os.path.exists(os.path.join(self.state.best_model_checkpoint, WEIGHTS_INDEX_NAME)):
            raise NotImplementedError
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.
        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            raise NotImplementedError
        elif is_sagemaker_mp_enabled():
            raise NotImplementedError
        elif (
            ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
            or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
            or self.fsdp is not None
        ):
            raise NotImplementedError
        elif self.deepspeed:
            raise NotImplementedError
        elif self.args.should_save:
            self._save(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info(f"Saving model checkpoint to {output_dir}")
        gen_output_dir = os.path.join(output_dir, 'gen')
        retr_output_dir = os.path.join(output_dir, 'retr')
        os.makedirs(gen_output_dir, exist_ok=True)
        os.makedirs(retr_output_dir, exist_ok=True)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            unwrapped_model = unwrap_model(self.model)
            gen_model = unwrapped_model.gen_model
            gen_state_dict = gen_model.state_dict()
            if isinstance(gen_model, PreTrainedModel):
                gen_model.save_pretrained(gen_output_dir, state_dict=gen_state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                torch.save(gen_state_dict, os.path.join(gen_output_dir, WEIGHTS_NAME))

            retr_model = unwrapped_model.retr_model
            retr_state_dict = retr_model.state_dict()
            if isinstance(retr_model, PreTrainedModel):
                retr_model.save_pretrained(retr_output_dir, state_dict=retr_state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                torch.save(retr_state_dict, os.path.join(retr_output_dir, WEIGHTS_NAME))
        else:
            self.model.gen_model.save_pretrained(gen_output_dir)
            self.model.retr_model.save_pretrained(retr_output_dir)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(gen_output_dir)
        if self.retr_tokenizer is not None:
            self.retr_tokenizer.save_pretrained(retr_output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(gen_output_dir, TRAINING_ARGS_NAME))
        torch.save(self.args, os.path.join(retr_output_dir, TRAINING_ARGS_NAME))

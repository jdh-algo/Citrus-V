# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import os
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from peft import PeftModel
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import RandomSampler
from transformers import EvalPrediction
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers.integrations.deepspeed import is_deepspeed_available
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.training_args import OptimizerNames
from transformers.utils import (is_accelerate_available, is_apex_available, is_peft_available, is_sagemaker_mp_enabled,
                                is_torch_hpu_available, is_torch_mlu_available, is_torch_mps_available,
                                is_torch_musa_available, is_torch_npu_available, is_torch_xpu_available)

from swift.utils import JsonlWriter, Serializer, gc_collect, get_logger, unwrap_model_for_generation
from .arguments import Seq2SeqTrainingArguments, TrainingArguments
from .mixin import DataLoaderMixin, SwiftMixin

if is_accelerate_available():
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedType, )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse('1.3.0'):
        from accelerate.utils import TorchTensorParallelPlugin
    if version.parse(accelerate_version) > version.parse('0.23.0'):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse('1.10')

    from transformers.trainer_pt_utils import smp_forward_backward
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

logger = get_logger()


class Trainer(SwiftMixin, HfTrainer):
    args: TrainingArguments

    @contextmanager
    def _patch_loss_function(self):
        model = self.model
        if isinstance(model, PeftModel):
            model = model.model
        model_cls = model.__class__
        if not hasattr(model_cls, 'loss_function'):
            yield
            return

        loss_function = model.loss_function
        _old_loss_function = model_cls.loss_function

        @staticmethod
        @wraps(loss_function)
        def new_loss_function(logits, labels, **kwargs):
            labels = labels.to(logits.device)  # fix device_map
            return loss_function(logits=logits, labels=labels, **kwargs)

        model_cls.loss_function = new_loss_function
        try:
            yield
        finally:
            model_cls.loss_function = _old_loss_function

    def train(self, *args, **kwargs):
        with self._patch_loss_function():
            return super().train(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if inputs.get('labels') is not None:
            self._compute_acc(outputs, inputs['labels'])
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = loss / self.args.gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss


class EmbeddingTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self.calculate_metric
        self.preprocess_logits_for_metrics = None
        self.label_names = ['labels']

    def calculate_metric(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        from swift.plugin.loss import infonce_loss, calculate_paired_metrics, calculate_infonce_metrics
        if self.compute_loss_func is infonce_loss:
            return calculate_infonce_metrics(eval_prediction.predictions, eval_prediction.label_ids)
        else:
            return calculate_paired_metrics(eval_prediction.predictions, eval_prediction.label_ids)


class RerankerTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self.calculate_metric
        self.label_names = ['labels']

        # Set up preprocess_logits_for_metrics to reduce memory usage for generative reranker
        from swift.plugin.loss import get_loss_func, LossType
        if self.compute_loss_func in [
                get_loss_func(LossType.generative_reranker),
                get_loss_func(LossType.listwise_generative_reranker)
        ]:
            self.preprocess_logits_for_metrics = self._preprocess_generative_reranker_logits
        else:
            self.preprocess_logits_for_metrics = None

    def _preprocess_generative_reranker_logits(self, logits, labels):
        """
        Preprocess logits for generative reranker to reduce memory usage.
        Extract only the yes/no token logits instead of keeping the full vocab logits.
        """
        import torch
        import os

        # Get token IDs for positive and negative tokens
        positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
        negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')

        tokenizer = getattr(self, 'processing_class', None)
        if tokenizer is None:
            # Fallback: return full logits if tokenizer not available
            return logits

        try:
            positive_token_id = tokenizer.convert_tokens_to_ids(positive_token)
            negative_token_id = tokenizer.convert_tokens_to_ids(negative_token)
        except Exception:
            # Fallback: return full logits if token conversion fails
            return logits

        # Extract only the yes/no token logits from the last position
        # This dramatically reduces memory usage
        if len(logits.shape) == 3:
            # Extract directly from last position: [batch_size, seq_len, vocab_size] -> [batch_size, 2]
            positive_logits = logits[:, -1, positive_token_id]  # [batch_size]
            negative_logits = logits[:, -1, negative_token_id]  # [batch_size]
            # Return as [batch_size, 2] tensor instead of full [batch_size, seq_len, vocab_size]
            logits = torch.stack([negative_logits, positive_logits], dim=1)
            return logits
        else:
            # Unexpected shape, return as-is
            return logits

    def calculate_metric(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        from swift.plugin.loss import (get_loss_func, LossType, calculate_reranker_metrics)

        # Check if we're using generative reranker (point-wise or list-wise)
        if self.compute_loss_func in [
                get_loss_func(LossType.generative_reranker),
                get_loss_func(LossType.listwise_generative_reranker)
        ]:
            # For generative reranker, predictions are now [batch_size, 2] from preprocessing
            # We need to handle this differently
            predictions = eval_prediction.predictions
            if len(predictions.shape) == 2 and predictions.shape[1] == 2:
                # Predictions are already preprocessed [batch_size, 2] format
                # Apply softmax to get probabilities
                import numpy as np
                exp_logits = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
                probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                relevance_scores = probabilities[:, 1]  # Positive class probability
                return calculate_reranker_metrics(relevance_scores, eval_prediction.label_ids)
            else:
                # Fallback to original method if preprocessing didn't work
                raise ValueError('Unexpected predictions shape')
        else:
            # For standard reranker (point-wise or list-wise)
            return calculate_reranker_metrics(eval_prediction.predictions, eval_prediction.label_ids)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Check if we have a custom loss function
        if self.compute_loss_func is not None:
            from swift.plugin.loss import get_loss_func, LossType
            loss_kwargs = {}

            if self.compute_loss_func in [
                    get_loss_func(LossType.generative_reranker),
                    get_loss_func(LossType.listwise_generative_reranker)
            ]:
                loss_kwargs['trainer'] = self

            # Get labels and compute outputs
            labels = inputs.get('labels')
            if labels is not None:
                labels = inputs.pop('labels')

            outputs = model(**inputs)

            if labels is not None:
                # Call custom loss function
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, **loss_kwargs)
            else:
                # Fallback to model's loss
                loss = outputs.loss

            if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
                loss = loss / self.args.gradient_accumulation_steps

            if labels is not None:
                self._compute_acc(outputs, labels)

            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


class Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer):
    args: Seq2SeqTrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2
        if self.args.predict_with_generate:
            from swift.llm import PtEngine
            self.infer_engine = PtEngine.from_model_template(
                self.model, self.template, max_batch_size=self.args.per_device_eval_batch_size)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'predict.jsonl'))

    @staticmethod
    def _predict_data_collator(batch):
        return {'_data': batch}

    @contextmanager
    def _patch_predict_with_generate(self):
        origin_data_collator = self.data_collator
        self.data_collator = self._predict_data_collator
        try:
            yield
        finally:
            self.data_collator = origin_data_collator

    def evaluate(self, *args, **kwargs):
        context = self._patch_predict_with_generate() if self.args.predict_with_generate else nullcontext()
        with context:
            res = super().evaluate(*args, **kwargs)
            gc_collect()
            return res

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            with self.template.forward_context(self.model, inputs):
                return super().prediction_step(
                    model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        from swift.llm import RequestConfig, InferRequest
        data_list = inputs['_data']
        labels_list = [InferRequest.remove_response(data['messages']) for data in data_list]
        with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation), self.template.generate_context():
            resp_list = self.infer_engine.infer(
                data_list,
                RequestConfig(max_tokens=self.model.generation_config.max_new_tokens),
                use_tqdm=False,
                template=self.template)

        response_list = []
        jsonl_cache = []
        device = self.args.device
        for data, resp, labels in zip(data_list, resp_list, labels_list):
            response = resp.choices[0].message.content
            jsonl_cache.append({'response': response, 'labels': labels, **data})
            response_list.append(Serializer.to_tensor(resp.choices[0].message.content).to(device=device))
        self.jsonl_writer.append(jsonl_cache, gather_obj=True)
        labels_list = [Serializer.to_tensor(labels).to(device=device) for labels in labels_list]
        response_list = pad_sequence(response_list, batch_first=True, padding_value=0)
        labels_list = pad_sequence(labels_list, batch_first=True, padding_value=0)
        return None, response_list, labels_list

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, num_masks_in_batch=None):
        # print(f"======using swift seq2seq trainer compute_loss.....")

        from swift.plugin.loss import get_loss_func
        loss_kwargs = {}
        labels = None
        compute_loss_func = self.compute_loss_func
        # print(f"============ inputs in swift.seq2seatrainer compute_loss: {inputs}")

        loss_scale = inputs.pop('loss_scale', None)
        if loss_scale is not None:
            loss_kwargs['loss_scale'] = loss_scale
            if compute_loss_func is None:
                compute_loss_func = get_loss_func('loss_scale')

        sample_channels = inputs.pop('channel', None)
        if sample_channels is not None and self.args.channels is not None:
            state = self.state
            setattr(state, 'local_step', getattr(state, 'local_step', 0))
            setattr(state, 'ch_loss_steps', getattr(state, 'ch_loss_steps', {}))

            loss_kwargs['sample_channels'] = sample_channels
            loss_kwargs['trainer'] = self
            if inputs.get('position_ids') is not None:
                loss_kwargs['position_ids'] = inputs['position_ids']

        if (self.label_smoother is not None or compute_loss_func is not None) and 'labels' in inputs:
            labels = inputs.pop('labels')

        use_logits_to_keep = self.get_use_logits_to_keep('labels' in inputs)
        if use_logits_to_keep:
            inputs['labels'], logits_to_keep = self.get_logits_to_keep(inputs['labels'])
            if logits_to_keep is not None:
                inputs['logits_to_keep'] = logits_to_keep

        # Remove input_ids from inputs to avoid duplicate arguments
        # since we're already passing it explicitly to model()
        # if 'input_ids' in inputs:
        #     inputs.pop('input_ids')

        outputs = model(**inputs)

        # print(f"===== outputs: {outputs}")
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is None:
            labels = inputs['labels']
            outputs.loss = outputs.loss.to(labels.device)

            # fix https://github.com/huggingface/transformers/issues/34263
            if num_items_in_batch is not None:
                outputs.loss = outputs.loss * ((labels[:, 1:] != -100).sum() / (num_items_in_batch + 1e-5))

            if num_masks_in_batch is not None:
                outputs.loss_dice = outputs.loss_dice * (outputs.cur_num_masks / (num_masks_in_batch + 1e-5))
                outputs.loss_mask = outputs.loss_mask * (outputs.cur_num_masks / (num_masks_in_batch + 1e-5))

            if isinstance(outputs, dict) and 'loss' not in outputs:
                raise ValueError(
                    'The model did not return a loss from the inputs, only the following keys: '
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        else:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if compute_loss_func is not None:
                loss = compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, **loss_kwargs)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)

        if outputs.loss_dice is not None:
            outputs.loss_dice = outputs.loss_dice.to(labels.device)
            loss = loss + outputs.loss_dice
        if outputs.loss_mask is not None:
            outputs.loss_mask = outputs.loss_mask.to(labels.device)
            loss = loss + outputs.loss_mask

        # visualization
        loss_token = outputs.loss.detach()
        if outputs.loss_dice is not None:
            loss_dice = outputs.loss_dice.detach()
        else:
            loss_dice = torch.tensor(0, device=loss.device)
        if outputs.loss_mask is not None:
            loss_mask = outputs.loss_mask.detach()
        else:
            loss_mask = torch.tensor(0, device=loss.device)

        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            loss = sequence_parallel.reduce_outputs(loss, labels)

            loss_token = sequence_parallel.reduce_outputs(loss_token, labels)  # labels 没用上
            loss_dice = sequence_parallel.reduce_outputs(loss_dice, labels)
            loss_mask = sequence_parallel.reduce_outputs(loss_mask, labels)

        if getattr(self.args, 'average_tokens_across_devices', False) and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

            loss_token *= self.accelerator.num_processes
            loss_dice *= self.accelerator.num_processes
            loss_mask *= self.accelerator.num_processes

        if outputs.logits is not None and labels is not None and not return_outputs:
            # Liger does not have logits
            self._compute_acc(outputs, labels)
        return (loss, outputs) if return_outputs else (loss, loss_token, loss_dice, loss_mask)

    # def training_step(self, model, inputs, *args, **kwargs):
    #     with self.template.forward_context(self.model, inputs):
    #         # print(f"======= inputs in training_step in swift seq2seq trainer: {inputs}")
    #         return super().training_step(model, inputs, *args, **kwargs)

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
        num_masks_in_batch=None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        with self.template.forward_context(self.model, inputs):
            model.train()
            if hasattr(self.optimizer, 'train') and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)
            # print(f"============ inputs in after prepare_inputs in transformers.training step: {inputs}")
            # print(f"is_sagemaker_mp_enabled(): {is_sagemaker_mp_enabled()}")
            if is_sagemaker_mp_enabled():  # False
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                # print(f"============ inputs in training step 111111: {inputs}")
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss, loss_token, loss_dice, loss_mask = self.compute_loss(
                    model, inputs, num_items_in_batch=num_items_in_batch, num_masks_in_batch=num_masks_in_batch)

            del inputs
            if (self.args.torch_empty_cache_steps is not None
                    and self.state.global_step % self.args.torch_empty_cache_steps == 0):
                if is_torch_xpu_available():
                    torch.xpu.empty_cache()
                elif is_torch_mlu_available():
                    torch.mlu.empty_cache()
                elif is_torch_musa_available():
                    torch.musa.empty_cache()
                elif is_torch_npu_available():
                    torch.npu.empty_cache()
                elif is_torch_mps_available():
                    torch.mps.empty_cache()
                elif is_torch_hpu_available():
                    logger.warning(
                        '`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache().')
                else:
                    torch.cuda.empty_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learnign rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs['learning_rate'] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel trainino

                loss_token = loss_token.mean()
                loss_dice = loss_dice.mean()
                loss_mask = loss_mask.mean()

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
                if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                    loss = loss / self.args.gradient_accumulation_steps

                    loss_token = loss_token / self.args.gradient_accumulation_steps
                    loss_mask = loss_mask / self.args.gradient_accumulation_steps
                    loss_dice = loss_dice / self.args.gradient_accumulation_steps

                # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
                # https://github.com/huggingface/transformers/pull/35808
                if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                    kwargs['scale_wrt_gas'] = False

                self.accelerator.backward(loss, **kwargs)

                return loss.detach(), loss_token, loss_dice, loss_mask

# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import contextlib
import copy
import functools
import inspect
import logging
import os
import shutil
import time
from contextlib import contextmanager
from copy import copy
from functools import partial, wraps
from types import MethodType
from typing import Callable, Dict, List, Optional, Tuple, Union

import safetensors
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint
import transformers
from datasets import Dataset as HfDataset
from modelscope import check_local_model_is_latest
from packaging import version
from peft import PeftModel
from torch.nn import Module
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modeling_utils import unwrap_model
from transformers.trainer import TrainerCallback, _is_peft_model
from transformers.trainer_callback import ExportableState, TrainerState
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import EvalPrediction, IntervalStrategy, TrainOutput, speed_metrics
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.utils import (XLA_FSDPV2_MIN_VERSION, is_accelerate_available, is_apex_available,
                                is_sagemaker_mp_enabled, is_torch_npu_available, is_torch_xla_available, logging)

from swift.hub import get_hub
from swift.llm import BatchSamplerShard, DataLoaderDispatcher, DataLoaderShard, Template
from swift.llm.utils import update_generation_config_eos_token
from swift.plugin import MeanMetric, compute_acc, extra_tuners
from swift.tuners import SwiftModel
from swift.utils import get_logger, is_dist, is_mp, is_mp_ddp, ms_logger_context, seed_worker, use_torchacc
from swift.utils.torchacc_utils import ta_trim_graph
from ..utils.torch_utils import get_device_count
from .arguments import TrainingArguments
from .utils import can_return_loss, find_labels, get_function, is_instance_of_ms_model

try:
    from trl import AutoModelForCausalLMWithValueHead
except (ImportError, RuntimeError):
    AutoModelForCausalLMWithValueHead = None

if is_accelerate_available():
    from accelerate import skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedType, )
if is_apex_available():
    from apex import amp

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.runtime as xr
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
else:
    IS_XLA_FSDPV2_POST_2_2 = False
# Name of the files used for checkpointing
TRAINING_ARGS_NAME = 'training_args.bin'
TRAINER_STATE_NAME = 'trainer_state.json'
OPTIMIZER_NAME = 'optimizer.pt'
SCALER_NAME = 'scaler.pt'
OPTIMIZER_NAME_BIN = 'optimizer.bin'
SCHEDULER_NAME = 'scheduler.pt'
FSDP_MODEL_NAME = 'pytorch_model_fsdp'

logger = get_logger()


class SwiftMixin:

    def __init__(self,
                 model: Union[PreTrainedModel, Module] = None,
                 args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[HfDataset] = None,
                 eval_dataset: Optional[Union[HfDataset, Dict[str, HfDataset]]] = None,
                 template: Optional[Template] = None,
                 model_init: Optional[Callable[[], PreTrainedModel]] = None,
                 compute_loss_func: Optional[Callable] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 **kwargs) -> None:
        if not hasattr(train_dataset, '__len__') and args.dataloader_num_workers > 1:
            args.dataloader_num_workers = 1
            logger.warning('Using IterableDataset, setting args.dataloader_num_workers to 1.')

        if args.check_model and hasattr(model, 'model_dir'):
            with ms_logger_context(logging.CRITICAL):
                check_local_model_is_latest(
                    model.model_dir, user_agent={
                        'invoked_by': 'local_trainer',
                        'third_party': 'swift',
                    })
        if eval_dataset is None and args:
            if getattr(args, 'eval_dataset', None):
                # Avoid trainer throwing errors.
                eval_dataset = []
            else:
                args.evaluation_strategy = IntervalStrategy.NO
                args.eval_strategy = IntervalStrategy.NO

        self._custom_metrics = {}
        self.template = template
        self.max_memory = 0
        self.hub = get_hub()

        self.model_meta = model.model_meta
        with self.hub.patch_hub():
            super().__init__(
                model=model,
                args=args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=template.tokenizer,
                model_init=model_init,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
                optimizers=optimizers,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                **kwargs)

        self.compute_loss_func = compute_loss_func
        if get_function(model.__class__.forward) is not get_function(model.forward):
            self.label_names = find_labels(model)
            self.can_return_loss = can_return_loss(model)
        self.label_names = self.label_names or ['labels']
        self.start_time = time.time()
        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            sequence_parallel.prepare_trainer(self)
        self._fix_gradient_checkpointing()
        update_generation_config_eos_token(self.model.generation_config, self.template)
        if getattr(self.model, 'origin_generation_config', None):
            self.model.origin_generation_config.eos_token_id = self.model.generation_config.eos_token_id

    def get_use_logits_to_keep(self, default_value: bool = True):
        use_logits_to_keep = self.args.use_logits_to_keep
        if use_logits_to_keep is None:
            base_model = self.template.get_base_model(self.model)
            use_logits_to_keep = (not self.model.model_meta.is_multimodal
                                  and 'logits_to_keep' in inspect.signature(base_model.forward).parameters
                                  and default_value)
        logger.info_once(f'use_logits_to_keep: {use_logits_to_keep}')
        return use_logits_to_keep

    def _save_initial_model(self, output_dir):
        # pissa/olora/lora-ga
        model = unwrap_model(self.model)
        if isinstance(model, PeftModel):
            config = model.peft_config.get('default')
            init_lora_weights = getattr(config, 'init_lora_weights', None)
            if (isinstance(init_lora_weights, str)
                    and any(s in init_lora_weights for s in ('pissa', 'olora', 'lora-ga'))):
                config.init_lora_weights = True
                model.save_pretrained(os.path.join(output_dir, 'initial_model'))
                config.init_lora_weights = init_lora_weights

    def _save_converted_model(self, output_dir):
        # pissa/olora/lora-ga
        model = unwrap_model(self.model)
        if isinstance(model, PeftModel):
            config = model.peft_config.get('default')
            init_lora_weights = getattr(config, 'init_lora_weights', None)
            if isinstance(init_lora_weights, str):
                config = copy(config)
                os.makedirs(os.path.join(output_dir, 'converted'), exist_ok=True)
                if 'lora-ga' in init_lora_weights:
                    try:
                        from lora_ga.entrypoint import LoraGAContext
                        with LoraGAContext(model):
                            model.save_pretrained(
                                os.path.join(output_dir, 'converted', 'default'),
                                path_initial_model_for_weight_conversion=os.path.join(
                                    os.path.dirname(output_dir), 'initial_model'),
                            )
                            model.peft_config['default'] = config
                    except ImportError as e:
                        error_message = """
                        Since 'LoRA-GA' is not implemented by PEFT, you will need to install it directly from GitHub.
                        Command: 'pip install git+https://github.com/lxline/LoRA-GA.git'.
                        """
                        logger.info(error_message)
                        raise RuntimeError(error_message) from e
                elif 'pissa' in init_lora_weights or 'olora' in init_lora_weights:
                    model.save_pretrained(
                        os.path.join(output_dir, 'converted', 'default'),
                        path_initial_model_for_weight_conversion=os.path.join(
                            os.path.dirname(output_dir), 'initial_model'),
                    )
                    model.peft_config['default'] = config

    def _load_optimizer_and_scheduler(self, *args, **kwargs):
        super()._load_optimizer_and_scheduler(*args, **kwargs)
        if is_mp_ddp():
            # fix mp+ddp adamw
            for v in self.optimizer.state.values():
                if 'step' in v:
                    # not on the same device
                    device_set = set([t.device for t in v.values()]) - {v['step'].device, torch.device('cpu')}
                    if len(device_set) >= 1:
                        v['step'] = v['step'].to('cpu')

    def _save_model(self, output_dir: Optional[str] = None, state_dict=None):
        # model
        supported_classes = (SwiftModel, PreTrainedModel, PeftModel)
        supported_names = ('SentenceTransformer', )
        if AutoModelForCausalLMWithValueHead is not None:
            supported_classes = supported_classes + (AutoModelForCausalLMWithValueHead, )
        save_safetensors = self.args.save_safetensors
        if not isinstance(self.model, supported_classes) and self.model.__class__.__name__ not in supported_names:
            if state_dict is None:
                state_dict = self.model.state_dict()

            _unwrap_model = unwrap_model(self.model)
            if isinstance(_unwrap_model, supported_classes):
                _unwrap_model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=save_safetensors)
            else:
                logger.info('Trainer.model is not a `PreTrainedModel`, only saving its state dict.')
                if save_safetensors:
                    safetensors.torch.save_file(state_dict, os.path.join(output_dir, 'model.safetensors'))
                else:
                    torch.save(state_dict, os.path.join(output_dir, 'pytorch_model.bin'))
        elif AutoModelForCausalLMWithValueHead and isinstance(self.model, AutoModelForCausalLMWithValueHead):
            # print(f"====== 001 ===== ")
            # save reward model
            state_dict = self.model.state_dict()
            decoder_state_dict, v_head_state_dict = {}, {}
            for name, param in state_dict.items():
                if name.startswith('v_head.'):
                    v_head_state_dict[name] = param
                else:
                    decoder_state_dict[name.replace('pretrained_model.', '', 1)] = param
            self.model.pretrained_model.save_pretrained(
                output_dir, state_dict=decoder_state_dict or None, safe_serialization=save_safetensors)
            if save_safetensors:
                from safetensors.torch import save_file
                save_file(
                    v_head_state_dict, os.path.join(output_dir, 'value_head.safetensors'), metadata={'format': 'pt'})
            else:
                torch.save(v_head_state_dict, os.path.join(output_dir, 'value_head.bin'))
        elif is_instance_of_ms_model(self.model):
            # print(f"====== 002 ===== ")
            PreTrainedModel.save_pretrained(
                self.model, output_dir, state_dict=state_dict, safe_serialization=save_safetensors)
        elif self.args.train_type in extra_tuners:
            # print(f"====== 003 ===== ")
            extra_tuners[self.args.train_type].save_pretrained(
                self.model, output_dir, state_dict=state_dict, safe_serialization=save_safetensors)
        else:
            # print(f"====== 004 ===== {self.model.__class__.__name__}")
            if self.model.__class__.__name__ != 'SentenceTransformer':
                # print(f"========= state_dict: {state_dict}")
                # print(f"======= save pretrained ======= save_safetensors: {save_safetensors} ==== state_dict is None: {state_dict is None}")
                self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=save_safetensors)
            else:
                # print(f"======= save sentencetransformers ======= ")
                @contextmanager
                def save_context():
                    save_pretrained = self.model[0].auto_model.save_pretrained
                    _state_dict = {
                        key[len('0.auto_model.'):] if 'auto_model' in key else key: value
                        for key, value in state_dict.items()
                    }
                    self.model[0].auto_model.save_pretrained = partial(
                        self.model[0].auto_model.save_pretrained, state_dict=_state_dict)
                    yield
                    self.model[0].auto_model.save_pretrained = save_pretrained

                with save_context():
                    self.model.save_pretrained(output_dir, safe_serialization=save_safetensors)
                    # copy sentencetransformers files
                    from swift.utils import copy_files_by_pattern
                    copy_files_by_pattern(self.model.model_dir, output_dir, '*.py')
                    copy_files_by_pattern(self.model.model_dir, output_dir, '*.json')

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Compatible with swift and peft"""
        # print(f"============ _save in swiftMinxin .....")
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._save_model(output_dir, state_dict)
        # training_args.bin
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        self._save_converted_model(output_dir)
        # args.json
        args_path = os.path.join(os.path.dirname(output_dir), 'args.json')
        if os.path.exists(args_path):
            shutil.copy(args_path, os.path.join(output_dir, 'args.json'))
        # predict.jsonl
        predict_jsonl = os.path.join(os.path.dirname(output_dir), 'predict.jsonl')
        if os.path.exists(predict_jsonl):
            shutil.move(predict_jsonl, os.path.join(output_dir, 'predict.jsonl'))

        is_adapter = isinstance(self.model, (SwiftModel, PeftModel))
        # tokenizer
        if not is_adapter:
            from swift.llm import save_checkpoint
            additional_saved_files = self.model_meta.additional_saved_files
            save_checkpoint(
                None,
                self.template.processor,
                output_dir,
                model_dirs=[self.model.model_dir],
                additional_saved_files=additional_saved_files)
            if getattr(self.model, 'origin_generation_config', None):
                self.model.origin_generation_config.save_pretrained(output_dir)

    def _fix_zero3_gather_all_parameters(self) -> None:
        if is_deepspeed_zero3_enabled() and not hasattr(self.deepspeed, '_zero3_consolidated_16bit_state_dict_origin'):
            parameters = inspect.signature(self.deepspeed._zero3_consolidated_16bit_state_dict).parameters
            if 'exclude_frozen_parameters' in parameters:

                def _zero3_consolidated_16bit_state_dict(model, exclude_frozen_parameters=False):
                    unwrapped = unwrap_model(model)
                    exclude_frozen_parameters = False
                    if isinstance(unwrapped, SwiftModel) and unwrapped.has_additional_modules:
                        exclude_frozen_parameters = True
                    if isinstance(unwrapped, PeftModel):
                        exclude_frozen_parameters = True
                    return model._zero3_consolidated_16bit_state_dict_origin(exclude_frozen_parameters)

                self.deepspeed._zero3_consolidated_16bit_state_dict_origin = (
                    self.deepspeed._zero3_consolidated_16bit_state_dict)
                self.deepspeed._zero3_consolidated_16bit_state_dict = MethodType(_zero3_consolidated_16bit_state_dict,
                                                                                 self.deepspeed)

    def _save_checkpoint(self, *args, **kwargs):
        self.state.last_model_checkpoint = os.path.join(self.args.output_dir, f'checkpoint-{self.state.global_step}')
        self._fix_zero3_gather_all_parameters()
        result = super()._save_checkpoint(*args, **kwargs)
        logger.info(f'Saving model checkpoint to {self.state.last_model_checkpoint}')
        return result

    @staticmethod
    @contextmanager
    def _fix_grad_norm_nan():
        from accelerate import Accelerator
        origin_clip_grad_norm_ = Accelerator.clip_grad_norm_

        def clip_grad_norm_(self, parameters, *args, **kwargs):
            # If NaN occurs, ignore weight updates.
            parameters = list(parameters)
            grad_norm = origin_clip_grad_norm_(self, parameters, *args, **kwargs)
            if isinstance(grad_norm, torch.Tensor) and grad_norm.isnan().item():
                for p in parameters:
                    p.grad = None
            return grad_norm

        Accelerator.clip_grad_norm_ = clip_grad_norm_
        try:
            yield
        finally:
            Accelerator.clip_grad_norm_ = origin_clip_grad_norm_

    def _fix_gradient_checkpointing(self):
        # fix use_reentrant
        if hasattr(torch.utils.checkpoint, '_old_checkpoint'):  # avoid double patching
            return
        args = self.args
        if args.gradient_checkpointing_kwargs:
            use_reentrant_ = args.gradient_checkpointing_kwargs.get('use_reentrant')
        else:
            use_reentrant_ = None
        if use_reentrant_ is None:
            if is_dist() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                use_reentrant_ = False
            else:
                use_reentrant_ = True
        logger.info(f'use_reentrant: {use_reentrant_}')
        _old_checkpoint = torch.utils.checkpoint.checkpoint

        @wraps(_old_checkpoint)
        def _new_checkpoint(*args, use_reentrant=None, **kwargs):
            return _old_checkpoint(*args, use_reentrant=use_reentrant_, **kwargs)

        torch.utils.checkpoint._old_checkpoint = _old_checkpoint
        torch.utils.checkpoint.checkpoint = _new_checkpoint
        try:
            # Fix the old version of transformers.
            import transformers.modeling_utils
            transformers.modeling_utils.checkpoint = _new_checkpoint
        except (ImportError, AttributeError):
            pass

    def _prepare_gradient_checkpointing(self, model) -> None:
        from swift.llm import HfConfigFactory, get_model_arch, deep_getattr, dynamic_gradient_checkpointing
        args = self.args
        HfConfigFactory.set_model_config_attr(model, 'use_cache', False)
        if args.gradient_checkpointing or args.vit_gradient_checkpointing:
            dynamic_gradient_checkpointing(model, args.vit_gradient_checkpointing)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)
            model.enable_input_require_grads()

        model_meta = model.model_meta
        model_arch = get_model_arch(model_meta.model_arch)
        if model_meta.is_multimodal and model_arch:
            for vision_tower_name in model_arch.vision_tower:
                vision_tower = deep_getattr(model, vision_tower_name)
                if hasattr(vision_tower, 'enable_input_require_grads'):
                    try:
                        if args.vit_gradient_checkpointing:
                            vision_tower.gradient_checkpointing_enable(
                                gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)
                            vision_tower.enable_input_require_grads()
                        else:
                            vision_tower.gradient_checkpointing_disable()
                            vision_tower.disable_input_require_grads()
                    except (NotImplementedError, AttributeError):
                        pass
        # Avoid vit_gradient_checkpointing being overwritten by transformers.Trainer.gradient_checkpointing_enable.
        self.args.gradient_checkpointing = False

    def train(self, *args, **kwargs):
        # print(f"===== trainer.mixin.train")
        if self.model_meta.is_multimodal:
            models = []
            for model_name in ['model', 'ref_model', 'value_model', 'teacher_model']:
                model = getattr(self, model_name, None)
                if isinstance(model, nn.Module):
                    models.append(model)

            reward_model = getattr(self, 'reward_model', None)
            if reward_model is not None:
                if isinstance(reward_model, list):
                    models.extend([m for m in reward_model if isinstance(m, nn.Module)])
                elif isinstance(reward_model, nn.Module):
                    models.append(reward_model)

            models = list(set(self.accelerator.unwrap_model(model) for model in models))  # Deduplicate
            self.template.register_post_encode_hook(models)
            logger.info(f'Successfully registered post_encode hook: {[model.__class__.__name__ for model in models]}.')
        self._save_initial_model(self.args.output_dir)

        # gradient_checkpointing
        gradient_checkpointing = self.args.gradient_checkpointing
        self._prepare_gradient_checkpointing(self.accelerator.unwrap_model(self.model))
        with self.hub.patch_hub(), self._fix_grad_norm_nan(), self._patch_skip_first_batches():
            res = super().train(*args, **kwargs)
            # print(f"===== trainer.mixin.train res: {super}")
        self.template.remove_post_encode_hook()
        self.args.gradient_checkpointing = gradient_checkpointing  # recover
        return res

    def push_to_hub(self, *args, **kwargs):
        with self.hub.patch_hub():
            return super().push_to_hub(*args, **kwargs)

    def get_max_cuda_memory(self, device: Optional[Union[torch.device, int]] = None) -> float:
        if device is None:
            mems = [torch.cuda.max_memory_reserved(device=device) for device in range(get_device_count())]
        else:
            mems = [torch.cuda.max_memory_reserved(device=device)]
        mem = sum(mems) / 1024**3
        self.max_memory = max(self.max_memory, mem)
        return mem

    def _maybe_log_save_evaluate(self, tr_loss, *args, **kwargs):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            self.control.should_log = False

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            loss = tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged)
            logs: Dict[str, float] = {'loss': loss}  # loss first

            if kwargs.get('tr_loss_token', None) is not None:
                tr_loss_token_scalar = self._nested_gather(kwargs.get('tr_loss_token')).mean().item()
                loss_token = tr_loss_token_scalar / (self.state.global_step - self._globalstep_last_logged)
                logs['loss_token'] = loss_token
            if kwargs.get('tr_loss_dice', None) is not None:
                tr_loss_dice_scalar = self._nested_gather(
                    kwargs.get('tr_loss_dice')).mean().item()  # 可能只有某一些进程有loss dice
                loss_dice = tr_loss_dice_scalar / (self.state.global_step - self._globalstep_last_logged)
                logs['loss_dice'] = loss_dice
            if kwargs.get('tr_loss_mask', None) is not None:
                tr_loss_mask_scalar = self._nested_gather(kwargs.get('tr_loss_mask')).mean().item()  # 只有部分进程有loss dice
                loss_mask = tr_loss_mask_scalar / (self.state.global_step - self._globalstep_last_logged)
                logs['loss_mask'] = loss_mask

            for k, metric in self._custom_metrics.items():
                value = metric.compute()
                if len(value) == 1:
                    val = list(value.values())[0]
                    logs[k] = val
                else:
                    for k_suffix, val in value.items():
                        new_k = f'{k}_{k_suffix}'
                        logs[new_k] = val
                metric.reset()

            if version.parse(transformers.__version__) >= version.parse('4.38'):
                grad_norm = args[0]
                if grad_norm is not None:
                    logs['grad_norm'] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs['learning_rate'] = self._get_learning_rate()
            if not is_torch_npu_available():
                logs['memory(GiB)'] = round(self.get_max_cuda_memory(), 2)

            elapse_time = time.time() - self.start_time
            logs['train_speed(iter/s)'] = round(self.state.global_step / elapse_time, 6)
            for k in list(logs.keys()):
                if logs[k] is None:
                    logs.pop(k)
            tr_loss -= tr_loss
            kwargs['tr_loss_token'] -= kwargs['tr_loss_token']
            kwargs['tr_loss_dice'] -= kwargs['tr_loss_dice']
            kwargs['tr_loss_mask'] -= kwargs['tr_loss_mask']

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs)

        if self.args.eval_use_evalscope and self.control.should_evaluate:
            self._evalscope_eval()
            if not self.eval_dataset:
                self.control.should_evaluate = False

        kwargs.pop('tr_loss_token', None)  # super不接受kwargs参数，因此需要去除
        kwargs.pop('tr_loss_dice', None)
        kwargs.pop('tr_loss_mask', None)
        super()._maybe_log_save_evaluate(tr_loss, *args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.args.optimizer is not None:
            from swift.plugin import optimizers_map
            optimizer_callback = optimizers_map[self.args.optimizer]
            self.optimizer, self.lr_scheduler = optimizer_callback(self.args, self.model, self.train_dataset)
            if self.optimizer is None:
                self.create_optimizer()
            if self.lr_scheduler is None:
                self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)
        else:
            super().create_optimizer_and_scheduler(num_training_steps=num_training_steps)

    def _compute_acc(self, outputs, labels) -> None:
        args = self.args
        acc_steps = args.acc_steps
        preds = outputs.logits.argmax(dim=-1)
        if self.state.global_step % acc_steps == 0:
            if use_torchacc():
                ta_trim_graph()
                preds = preds.to('cpu')
                labels = labels.to('cpu')
            metrics = compute_acc(
                preds, labels, acc_strategy=args.acc_strategy, is_encoder_decoder=self.template.is_encoder_decoder)
            for k, v in metrics.items():
                if k not in self._custom_metrics:
                    self._custom_metrics[k] = MeanMetric(nan_value=None)
                self._custom_metrics[k].update(v)

    @torch.no_grad()
    def _evalscope_eval(self):
        from ..llm.eval.utils import EvalModel
        from evalscope import TaskConfig, run_task
        from evalscope.constants import EvalType

        self.model.eval()
        max_batch_size = self.args.per_device_eval_batch_size
        custom_model = EvalModel(
            self.model, self.template, max_batch_size=max_batch_size, model_name=f'model-step{self.state.global_step}')
        task_config = TaskConfig(
            model=custom_model,
            eval_type=EvalType.CUSTOM,
            datasets=self.args.eval_dataset,
            dataset_args=self.args.eval_dataset_args,
            limit=self.args.eval_limit,
            work_dir=os.path.join(self.args.output_dir, 'eval'),
            eval_batch_size=max_batch_size,
            generation_config=self.args.eval_generation_config or {'max_tokens': 512},
        )
        # start evaluation
        eval_report = run_task(task_config)
        # convert to dict
        eval_dict = {f'test_{k}': v.score for k, v in eval_report.items()}
        self.log(eval_dict)

        self.model.train()
        return eval_dict

    def get_logits_to_keep(self, labels):
        if labels.shape[0] == 1 and not is_mp():
            # device_map may encounter device mismatch issues.
            loss_mask = (labels != -100)[0]
            labels = labels[:, loss_mask]
            labels = nn.functional.pad(labels, (1, 0), value=-100)
            logits_to_keep = nn.functional.pad(loss_mask[1:], (0, 1), value=True)
        else:
            logits_to_keep = labels.shape[-1] - ((labels != -100).int().argmax(-1).min().item()) + 1
            assert logits_to_keep > 0
            labels = labels[:, -logits_to_keep:]
        return labels, logits_to_keep

    def get_cu_seqlens(self, position_ids, logits_to_keep) -> torch.Tensor:
        assert position_ids.shape[0] == 1
        position_ids = position_ids[0]
        indices = torch.arange(position_ids.shape[0], device=position_ids.device)
        cu_seqlens = torch.concat([
            indices[position_ids == 0],
            torch.tensor(position_ids.shape, device=position_ids.device),
        ])
        res_cu_seqlens = cu_seqlens.clone()
        if isinstance(logits_to_keep, torch.Tensor):
            for i in range(cu_seqlens.shape[0] - 1):
                start, end = cu_seqlens[i], cu_seqlens[i + 1]
                res_cu_seqlens[i + 1:] -= (~logits_to_keep[start:end]).sum()
        elif isinstance(logits_to_keep, int):
            res_cu_seqlens[1:] -= position_ids.shape[0] + 1 - logits_to_keep
        return res_cu_seqlens

    def get_batch_samples(self, *args, **kwargs):
        res = super().get_batch_samples(*args, **kwargs)
        from swift.trainers.sequence_parallel import sequence_parallel

        batch_samples, num_items_in_batch = res
        num_masks_in_batch = None
        count_num_masks_in_batch = (len(batch_samples) > 0 and 'g_pixel_values' in batch_samples[0])
        if count_num_masks_in_batch:
            try:
                num_masks_in_batch = 0
                if len(batch_samples[0]['g_pixel_values'].shape) == 5:  # packing
                    for item in batch_samples:
                        num_masks_in_batch += item['g_pixel_values'].shape[1]
                else:  # no packing
                    for item in batch_samples:
                        num_items_in_batch += item['g_pixel_values'].shape[0]
                    # num_masks_in_batch = sum(item["masks"].any() for item in batch_samples)
                res = (batch_samples, num_items_in_batch, num_masks_in_batch)
            except (TypeError, AttributeError):
                logger.info(f'error in num_masks_in batch')
                pass

        if (self.template.sequence_parallel_size == 1 or 'Ulysses' == sequence_parallel.__class__.__name__
                or 'RingAttention' == sequence_parallel.__class__.__name__):
            # ulysses and ring attention split inputs in the model hook, so no need to gather num_items_in_batch
            return res

        if num_items_in_batch is None:
            num_items_in_batch = torch.tensor(0).to(args[2])
        if num_masks_in_batch is None:
            num_masks_in_batch = torch.tensor(0).to(args[2])
        from swift.trainers.sequence_parallel import sequence_parallel
        dist.all_reduce(num_items_in_batch, dist.ReduceOp.SUM, sequence_parallel.sp_group)
        dist.all_reduce(num_masks_in_batch, dist.ReduceOp.SUM, sequence_parallel.sp_group)
        return batch_samples, num_items_in_batch, num_masks_in_batch

    @contextmanager
    def _patch_skip_first_batches(self):
        from transformers import trainer
        origin_skip_first_batches = trainer.skip_first_batches

        def skip_first_batches(dataloader, num_batches=0):
            if isinstance(dataloader, (DataLoaderShard, DataLoaderDispatcher)):
                # DataLoaderMixin
                return self.get_train_dataloader(skip_batches=num_batches)
            else:
                return origin_skip_first_batches(dataloader, num_batches)

        trainer.skip_first_batches = skip_first_batches
        try:
            yield
        finally:
            trainer.skip_first_batches = origin_skip_first_batches

    def _inner_training_loop(self,
                             batch_size=None,
                             args=None,
                             resume_from_checkpoint=None,
                             trial=None,
                             ignore_keys_for_eval=None):
        # print(f"====== inner training loop")
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped, ) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f'Currently training with a batch size of: {self._train_batch_size}')
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps
        # print(f"====== num_train_tokens: {num_train_tokens}")

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError('Currently --debug underflow_overflow is not supported under DP. Please use DDP'
                                 ' (torchrun or torch.distributed.launch (deprecated)).')
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, 'fsdp_version', 1) == 2)
        if is_fsdp2:
            delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(stateful_callbacks=[
            cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
        ])
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != 'fp8':
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, 'step'):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer,
                                                                                    self.lr_scheduler)
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model))
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        print(f'********** running .......')
        logger.info('***** Running training *****')
        logger.info(f'  Num examples = {num_examples:,}')
        logger.info(f'  Num Epochs = {num_train_epochs:,}')
        logger.info(f'  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}')
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f'  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}')
        logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}')
        logger.info(f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
        logger.info(f'  Total optimization steps = {max_steps:,}')
        logger.info(f'  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}')

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info('  Continuing training from checkpoint, will skip to saved global_step')
            logger.info(f'  Continuing training from epoch {epochs_trained}')
            logger.info(f'  Continuing training from global step {self.state.global_step}')
            if not args.ignore_data_skip:
                logger.info(f'  Will skip the first {epochs_trained} epochs then the first'
                            f' {steps_trained_in_current_epoch} batches in the first epoch.')

        # Update the references
        for attr in ('model', 'optimizer', 'lr_scheduler'):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        tr_loss_token = torch.tensor(0.0, device=args.device)
        tr_loss_dice = torch.tensor(0.0, device=args.device)
        tr_loss_mask = torch.tensor(0.0, device=args.device)

        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, 'set_epoch'):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader) if len_dataloader is not None else args.max_steps
                * args.gradient_accumulation_steps)
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)

            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
                remainder < args.gradient_accumulation_steps)
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch, num_masks_in_batch = self.get_batch_samples(
                    epoch_iterator, num_batches, args.device)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, 'main_input_name', 'input_ids')
                        if main_input_name not in inputs:
                            logger.warning('Tried to track the number of tokens seen, however the current model is '
                                           'not configured properly to know what item is the input. To fix this, add '
                                           'a `main_input_name` attribute to the model class you are using.')
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model) if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type != DistributedType.DEEPSPEED else contextlib.nullcontext)
                    with context():
                        tr_loss_step, tr_loss_token_step, tr_loss_dice_step, tr_loss_mask_step = self.training_step(
                            model, inputs, num_items_in_batch, num_masks_in_batch)

                    if (args.logging_nan_inf_filter and not is_torch_xla_available()
                            and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)

                        tr_loss_token = tr_loss_token + tr_loss_token / (1 + self.state.global_step
                                                                         - self._globalstep_last_logged)
                        tr_loss_dice = tr_loss_dice + tr_loss_dice / (1 + self.state.global_step
                                                                      - self._globalstep_last_logged)
                        tr_loss_mask = tr_loss_mask + tr_loss_mask / (1 + self.state.global_step
                                                                      - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f'Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}'
                            )
                        tr_loss = tr_loss + tr_loss_step

                        tr_loss_token = tr_loss_token + tr_loss_token_step
                        tr_loss_dice = tr_loss_dice + tr_loss_dice_step
                        tr_loss_mask = tr_loss_mask + tr_loss_mask_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (is_accelerate_available()
                                    and self.accelerator.distributed_type == DistributedType.DEEPSPEED):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, 'item'):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        # get leaning rate before update
                        learning_rate = self._get_learning_rate()

                        if not self.accelerator.optimizer_step_was_skipped:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss,
                            grad_norm,
                            model,
                            trial,
                            epoch,
                            ignore_keys_for_eval,
                            start_time,
                            learning_rate=learning_rate,
                            tr_loss_token=tr_loss_token,
                            tr_loss_dice=tr_loss_dice,
                            tr_loss_mask=tr_loss_mask,
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning('There seems not to be a single sample in your epoch_iterator, stopping training at step'
                               f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                               f' num_steps ({max_steps}) higher than the number of available samples.')
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(
                tr_loss,
                grad_norm,
                model,
                trial,
                epoch,
                ignore_keys_for_eval,
                start_time,
                learning_rate=learning_rate,
                tr_loss_token=tr_loss_token,
                tr_loss_dice=tr_loss_dice,
                tr_loss_mask=tr_loss_mask,
            )

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning("You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                                   'configured. Check your training configuration if this is unexpected.')
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, '_past'):
            # Clean the state at the end of training
            delattr(self, '_past')

        logger.info('\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n')
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous('load_best_model_at_end')
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            'train',
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics['total_flos'] = self.state.total_flos
        metrics['train_loss'] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f'Deleting older checkpoint [{checkpoint}] due to args.save_total_limit')
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)


class DataLoaderMixin:

    def get_train_dataloader(self, skip_batches=0):
        dataloader = None
        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            dataloader = sequence_parallel.get_dataloader(
                self, self.train_dataset, self._train_batch_size, skip_batches=skip_batches)
        if dataloader is None:
            # Higher efficiency
            if self.train_dataset is None:
                raise ValueError('Trainer: training requires a train_dataset.')
            args = self.args
            train_dataset = self.train_dataset

            dataloader_params = {
                'collate_fn': self.data_collator,
                'num_workers': args.dataloader_num_workers,
                'pin_memory': args.dataloader_pin_memory,
                'persistent_workers': args.dataloader_persistent_workers,
                'prefetch_factor': args.dataloader_prefetch_factor
            }
            batch_sampler_params = {
                'drop_last': args.dataloader_drop_last,
                'shuffle': args.train_dataloader_shuffle,
                'data_seed': args.data_seed,
            }

            if hasattr(train_dataset, '__len__'):
                batch_sampler = BatchSamplerShard(
                    len(train_dataset), batch_size=self._train_batch_size, **batch_sampler_params)
                dataloader_params['worker_init_fn'] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index)
                if skip_batches > 0:
                    from accelerate.data_loader import SkipBatchSampler
                    batch_sampler = SkipBatchSampler(batch_sampler, skip_batches=skip_batches)
                dataloader_params['batch_sampler'] = batch_sampler
                dataloader = DataLoaderShard(train_dataset, device=self.accelerator.device, **dataloader_params)
            else:
                # IterableDataset
                if dist.is_initialized() and dataloader_params['prefetch_factor']:
                    dataloader_params['prefetch_factor'] = dataloader_params['prefetch_factor'] * dist.get_world_size()
                dataloader = DataLoader(train_dataset, batch_size=self._train_batch_size, **dataloader_params)
                dataloader = DataLoaderDispatcher(dataloader, self.accelerator.device, skip_batches=skip_batches)
        return dataloader

    def get_eval_dataloader(self, eval_dataset=None):
        dataloader = None
        if self.template.sequence_parallel_size > 1:
            from swift.trainers.sequence_parallel import sequence_parallel
            if eval_dataset is None and self.eval_dataset is None:
                raise ValueError('Trainer: evaluation requires an eval_dataset.')
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            dataloader = sequence_parallel.get_dataloader(self, eval_dataset, self.args.eval_batch_size)
        if dataloader is None:
            return super().get_eval_dataloader(eval_dataset=eval_dataset)
        return dataloader

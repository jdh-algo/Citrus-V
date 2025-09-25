import os
from collections import OrderedDict
from dataclasses import dataclass
from optparse import Option
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures.sam2_train import SAM2TrainRunner
from architectures.third_parts.mmdet.models.losses import CrossEntropyLoss, DiceLoss
from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig, Qwen2_5_VLVisionConfig
from transformers.utils import ModelOutput


class CitrusVConfig(PretrainedConfig):
    """Configuration class for CitrusV model."""

    model_type = 'citrus_v'
    sub_configs = {'vision_config': Qwen2_5_VLVisionConfig, 'text_config': Qwen2_5_VLTextConfig}

    def __init__(self,
                 text_config=None,
                 vision_config=None,
                 image_token_id=151655,
                 seg_token_idx=151665,
                 loss_mask_config: Dict[str, Any] = {},
                 loss_dice_config: Dict[str, Any] = {},
                 grounding_encoder_config: Dict[str, Any] = {},
                 frozen_sam2_decoder: bool = False,
                 frozen_sam2_image_encoder: bool = False,
                 frozen_sam2_prompt_encoder: bool = False,
                 use_seg_loss: bool = True,
                 use_res_seg_projector: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs['vision_config'](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs['vision_config']()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs['text_config'](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs['text_config'](**kwargs)

        self.image_token_id = image_token_id
        self.seg_token_idx = seg_token_idx

        self.grounding_encoder_config = grounding_encoder_config
        self.frozen_sam2_decoder = frozen_sam2_decoder
        self.frozen_sam2_image_encoder = frozen_sam2_image_encoder
        self.frozen_sam2_prompt_encoder = frozen_sam2_prompt_encoder

        self.loss_mask_config = loss_mask_config
        self.loss_dice_config = loss_dice_config

        self.use_seg_loss = use_seg_loss
        self.use_res_seg_projector = use_res_seg_projector


@dataclass
class CitrusVCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None

    # added by xuyangcao
    loss_mask: Optional[torch.FloatTensor] = None
    loss_dice: Optional[torch.FloatTensor] = None
    cur_num_masks: Optional[torch.FloatTensor] = None


class CitrusVOutput(CitrusVCausalLMOutputWithPast):
    """
    Output class for SamQwen2_5VL model.
    Extends CausalLMOutputWithPast to include segmentation losses.
    """

    def __init__(
        self,
        loss=None,
        loss_mask=None,
        loss_dice=None,
        cur_num_masks=None,
        logits=None,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
    ):
        super().__init__(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
            loss_mask=loss_mask,
            loss_dice=loss_dice,
            cur_num_masks=cur_num_masks,
        )


class ResidualSegProjector(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super(ResidualSegProjector, self).__init__()

        self.linear1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        if in_dim != out_dim:
            self.residual_mapping = nn.Linear(in_dim, out_dim)
        else:
            self.residual_mapping = nn.Identity()

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.dropout(out)
        identity = self.residual_mapping(identity)
        out += identity
        return out


class CitrusV(PreTrainedModel, GenerationMixin):
    """
    CitrusV model transformed to be compatible with the Hugging Face Transformers library.
    """
    config_class = CitrusVConfig
    base_model_prefix = 'citrus_v'
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    _checkpoint_conversion_mapping = {
        '^visual': 'model.visual',
        r'^model(?!\.(language_model|visual))': 'model.language_model',
        r'^lm_head': 'lm_head',
        'lm_head.weight': 'lm_head.weight'
    }
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config):
        super().__init__(config)

        if config.torch_dtype == 'float32':
            self.torch_dtype = torch.float32
        elif config.torch_dtype == 'float16':
            self.torch_dtype = torch.float16
        elif config.torch_dtype == 'bfloat16':
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.bfloat16

        self.seg_token_idx = config.seg_token_idx

        # mllm
        self.mllm = Qwen2_5_VLForConditionalGeneration(config)

        # grounding encoder
        self.grounding_encoder = SAM2TrainRunner(**config.grounding_encoder_config)
        self.grounding_encoder.requires_grad_(True)
        if not config.frozen_sam2_decoder:
            self.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)
        if not config.frozen_sam2_image_encoder:
            self.grounding_encoder.sam2_model.image_encoder.requires_grad_(True)
        if not config.frozen_sam2_prompt_encoder:
            self.grounding_encoder.sam2_model.sam_prompt_encoder.requires_grad_(True)

        # text hidden fcs
        self.use_res_seg_projector = config.use_res_seg_projector
        in_dim = self.mllm.model.config.hidden_size
        out_dim = self.grounding_encoder.hidden_dim
        if self.use_res_seg_projector:
            self.seg_projector = ResidualSegProjector(in_dim, out_dim, dropout=0.0)
        else:
            self.seg_projector = nn.Sequential(
                nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim), nn.Dropout(0.0))

        # loss
        self.loss_mask = CrossEntropyLoss(**config.loss_mask_config)
        self.loss_dice = DiceLoss(**config.loss_dice_config)
        self.use_seg_loss = config.use_seg_loss

    @property
    def language_model(self):
        return self.mllm.language_model

    @property
    def model(self):
        return self.mllm.model

    @property
    def visual(self):
        return self.mllm.visual

    @property
    def grounding_mask_decoder(self):
        return self.grounding_encoder.sam2_model.sam_mask_decoder

    @property
    def grounding_image_encoder(self):
        return self.grounding_encoder.sam2_model.image_encoder

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def load_ori_state_dict(self, mllm_path, sam2_path):
        # load original model state dict
        from transformers import Qwen2_5_VLForConditionalGeneration
        original_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(mllm_path)

        # get original model state dict
        original_state_dict = original_model.state_dict()

        # create mapping dictionary
        mapping = {}
        for key in original_state_dict.keys():
            if key.startswith('model.visual'):
                mapping[key] = f'model.{key}'
            elif key.startswith('model.language_model'):
                mapping[key] = f'model.{key}'
            elif key.startswith('lm_head'):
                mapping[key] = key
            else:
                mapping[key] = f'model.{key}'

        # load weights
        self.mllm.load_state_dict(original_state_dict, strict=False)

        print(f'lm_head weight loaded, sum: {self.mllm.lm_head.weight.sum()}')
        print(f'embed_tokens weight loaded, sum: {self.mllm.model.language_model.embed_tokens.weight.sum()}')
        print(f'q_proj weight loaded, sum: {self.mllm.model.language_model.layers[0].self_attn.q_proj.weight.sum()}')

        del original_model  # release memory

        self.grounding_encoder.load_ori_state_dict(sam2_path)

    def get_input_embeddings(self):
        return self.mllm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.mllm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.mllm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.mllm.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None):
        self.mllm.resize_token_embeddings(new_num_tokens)

    def check_obj_number(self, pred_embeddings_list, gt_masks, fix_number=5):
        assert len(pred_embeddings_list) == len(gt_masks)
        ret_pred_embeddings_list = []
        ret_gt_masks = []
        for pred_mebeds, gt_masks in zip(pred_embeddings_list, gt_masks):
            if len(pred_mebeds) != len(gt_masks):
                min_num = min(len(pred_mebeds), len(gt_masks))
                pred_mebeds = pred_mebeds[:min_num]
                gt_masks = gt_masks[:min_num]

            if len(pred_mebeds) != fix_number:
                if len(pred_mebeds) > fix_number:
                    _idxs = torch.randperm(pred_mebeds.shape[0])
                    _idxs = _idxs[:fix_number]
                    pred_mebeds = pred_mebeds[_idxs]
                    gt_masks = gt_masks[_idxs]
                else:
                    n_repeat = fix_number // len(pred_mebeds) + 1
                    pred_mebeds = torch.cat([pred_mebeds] * n_repeat, dim=0)[:fix_number]
                    gt_masks = torch.cat([gt_masks] * n_repeat, dim=0)[:fix_number]
            ret_pred_embeddings_list.append(pred_mebeds)
            ret_gt_masks.append(gt_masks)
        return ret_pred_embeddings_list, ret_gt_masks

    def _get_pesudo_data(self, dtype, device, batch_size):
        g_pixel_values = torch.zeros((batch_size, 3, 1024, 1024), dtype=dtype, device=device)
        gt_masks = [torch.zeros((5, 256, 256), dtype=torch.LongTensor, device=device)] * batch_size
        return g_pixel_values, gt_masks

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        g_pixel_values: Optional[torch.FloatTensor] = None,
        return_seg_masks: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CitrusVOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # mllm
        output = self.mllm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always need hidden states for segmentation
            return_dict=True,  # Always use return_dict=True internally
        )
        loss_mask = torch.tensor(0.0, device=output.loss.device)
        loss_dice = torch.tensor(0.0, device=output.loss.device)
        cur_num_mask = torch.tensor(0, device=loss_mask.device)

        if self.use_seg_loss:
            gt_masks = masks
            hidden_states = output.hidden_states  # [B, seq_len, num_embendings]
            hidden_states = self.seg_projector(hidden_states[-1])
            ########### hiddenstate grad hook
            # hidden_states_lm = hidden_states[-1]
            # hidden_states = self.seg_projector(hidden_states_lm)
            # hidden_states.register_hook(lambda grad: grad * 0.001)
            ###########

            seg_token_mask = (input_ids == self.seg_token_idx)  # [batch, seq_len]
            seg_token_counts = seg_token_mask.int().sum(-1)  # [batch]

            has_seg = seg_token_mask.any(dim=1)
            selected_indices = has_seg.cpu().nonzero(as_tuple=True)[0]

            if len(selected_indices) > 0:
                selected_seg_token_mask = seg_token_mask[selected_indices].cpu()  # [B, seq_len]
                selected_seg_token_counts = seg_token_counts[selected_indices]

                selected_hidden_states = hidden_states[selected_indices]  # [B, seq_len, 256]
                selected_gt_masks = [gt_masks[i] for i in selected_indices.tolist()]
                selected_g_pixel_values = g_pixel_values[selected_indices]

                _zero = hidden_states.mean() * 0.0
                pred_embeddings = selected_hidden_states[selected_seg_token_mask] + _zero
                pred_embeddings_list_ = torch.split(pred_embeddings, selected_seg_token_counts.tolist(), dim=0)
                pred_embeddings_list = [item for item in pred_embeddings_list_ if len(item) != 0]

                if len(selected_g_pixel_values.shape) == 5:
                    ret_pred_embeddings_list = []
                    ret_gt_masks = []
                    for pred_mebeds, gt_masks in zip(pred_embeddings_list, gt_masks):
                        if len(pred_mebeds) != len(gt_masks):
                            min_num = min(len(pred_mebeds), len(gt_masks))
                            pred_mebeds = pred_mebeds[:min_num]
                            gt_masks = gt_masks[:min_num]
                        ret_pred_embeddings_list.append(pred_mebeds)
                        ret_gt_masks.append(gt_masks)
                    pred_embeddings_list, selected_gt_masks = ret_pred_embeddings_list, ret_gt_masks
                    language_embeddings = torch.stack(pred_embeddings_list)

                    pred_masks = []
                    for i in range(language_embeddings.shape[1]):
                        cur_language_embeddings = language_embeddings[:, i:i + 1, :]
                        cur_g_pixel_values = g_pixel_values[:, i, :, :, :]
                        sam_states = self.grounding_encoder.get_sam2_embeddings(cur_g_pixel_values)
                        cur_pred_masks = self.grounding_encoder.inject_language_embd(
                            sam_states, cur_language_embeddings)
                        pred_masks.append(cur_pred_masks)

                    pred_masks = torch.cat(pred_masks, dim=0)
                    selected_gt_masks = torch.cat(selected_gt_masks, dim=0).flatten(0, 1)
                else:
                    pred_embeddings_list, selected_gt_masks = self.check_obj_number(pred_embeddings_list,
                                                                                    selected_gt_masks)
                    language_embeddings = torch.stack(pred_embeddings_list)

                    sam_states = self.grounding_encoder.get_sam2_embeddings(selected_g_pixel_values)
                    pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings)

                    selected_gt_masks = [
                        F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0)
                        for gt_mask in selected_gt_masks
                    ]
                    selected_gt_masks = torch.cat(selected_gt_masks, dim=0)
                cur_num_mask = torch.tensor(selected_g_pixel_values.shape[0], device=output.loss.device)

                # postprocess
                pred_masks = pred_masks.flatten(0, 1).contiguous()

                if len(pred_masks) != len(selected_gt_masks):
                    min_num = min(len(pred_masks), len(selected_gt_masks))
                    pred_masks = pred_masks[:min_num]
                    selected_gt_masks = selected_gt_masks[:min_num]

                if self.loss_mask is not None:
                    selected_gt_masks_long = selected_gt_masks
                    sam_loss_mask = self.loss_mask(pred_masks, selected_gt_masks_long)
                    loss_mask = loss_mask + sam_loss_mask

                if self.loss_dice is not None:
                    selected_gt_masks_float = selected_gt_masks
                    sam_loss_dice = self.loss_dice(pred_masks, selected_gt_masks_float)
                    loss_dice = loss_dice + sam_loss_dice

        output = CitrusVOutput(
            loss=output.loss,
            loss_mask=loss_mask,
            loss_dice=loss_dice,
            logits=output.logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            cur_num_masks=cur_num_mask)

        return output

    @staticmethod
    def mask_to_rle(mask):
        """
        mask: numpy array, shape (H, W), dtype uint8, 0/1
        return: COCO RLE dict, counts为str
        """
        import numpy as np
        from pycocotools import mask as mask_utils
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle['counts'] = rle['counts'].decode() if isinstance(rle['counts'], bytes) else rle['counts']
        return rle

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        masks=None,
        image_grid_thw=None,
        g_pixel_values=None,
        attention_mask=None,
        return_seg_masks=True,
        **kwargs
    ):
        with torch.no_grad():
            device = self.device
            if pixel_values is not None:
                pixel_values=pixel_values.to(device)
            if image_grid_thw is not None:
                image_grid_thw=image_grid_thw.to(device)

            useargs = True
            kwargs["use_cache"] = True
            generate_kwargs = deepcopy_generate_kwargs(input_ids, pixel_values, image_grid_thw, attention_mask, device, useargs, **kwargs)
            generate_output1 = self.mllm.generate(
                                    output_hidden_states=False,
                                    return_dict_in_generate=True,
                                    **generate_kwargs,
                                    )
            seg_mask = generate_output1.sequences[0][:-1] == self.seg_token_idx
            has_seg = seg_mask.any()
            if not has_seg:
                return {
                    "sequences": generate_output1.sequences,
                    "seg_masks": torch.empty(0),
                    "seg_masks_rle": [],
                }
        
            useargs = False
            generate_kwargs = deepcopy_generate_kwargs(input_ids, pixel_values, image_grid_thw, attention_mask, device, useargs, **kwargs)
            generate_output = self.mllm.generate(
                                                output_hidden_states=True,
                                                return_dict_in_generate=True,
                                                **generate_kwargs,
                                                )
            hidden_states = generate_output.hidden_states
            last_hidden_states = [item[-1][0] for item in hidden_states]
            last_hidden_states = torch.cat(last_hidden_states, dim=0)
            seg_mask = generate_output.sequences[0][:-1] == self.seg_token_idx
    
            seg_hidden_states = get_seg_hidden_states(last_hidden_states, generate_output.sequences[0][:-1], self.seg_token_idx)
            
            all_seg_hidden_states = self.seg_projector(seg_hidden_states)
            ret_masks = []
            for seg_hidden_states in all_seg_hidden_states:
                seg_hidden_states = seg_hidden_states.unsqueeze(0)
                sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values.to(device))
                pred_masks = self.grounding_encoder.inject_language_embd(sam_states, seg_hidden_states.unsqueeze(1))
    
                masks = pred_masks[:, 0]
                masks = masks.sigmoid() > 0.5
                masks = masks.cpu().numpy()
                ret_masks.append(masks)
            if ret_masks:
                all_masks = torch.from_numpy(np.concatenate(ret_masks, axis=0))
            else:
                all_masks = torch.empty(0)

            seg_masks_rle = []
            for mask in ret_masks:
                for single_mask in mask:
                    mask_np = single_mask.astype('uint8')
                    seg_masks_rle.append(self.mask_to_rle(mask_np))
    
            return {
                "sequences": generate_output.sequences,
                "seg_masks": all_masks,
                "seg_masks_rle": seg_masks_rle,
            }

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorder cache for beam search.
        """
        # Delegate to the MLLM model's _reorder_cache
        return self.mllm._reorder_cache(past_key_values, beam_idx)

    def extract_seg_masks(self, forward_output, seg_token_mask, g_pixel_values):
        """
        Extract segmentation masks from forward_output and seg_token_mask.
        """
        # Extract segmentation features
        hidden_states = forward_output.hidden_states[-1]  # Last layer hidden states
        hidden_states = self.seg_projector(hidden_states)

        # Find [SEG] token positions
        seg_token_mask = seg_token_mask.to(hidden_states.device)

        if seg_token_mask.any():
            # Extract embeddings for [SEG] tokens
            pred_embeddings = hidden_states[seg_token_mask]

            # Count [SEG] tokens per sequence
            seg_token_counts = seg_token_mask.int().sum(-1)

            # Split embeddings by sequence
            pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
            pred_embeddings_list = []
            for item in pred_embeddings_list_:
                if len(item) > 0:
                    pred_embeddings_list.append(item)

            if pred_embeddings_list:
                # Pad embeddings to same length for batching
                max_objs = max(len(emb) for emb in pred_embeddings_list)
                if max_objs > 0:
                    padded_embeddings = []
                    for emb in pred_embeddings_list:
                        if len(emb) < max_objs:
                            # Pad with repeated embeddings
                            n_repeat = max_objs // len(emb) + 1
                            emb = torch.cat([emb] * n_repeat, dim=0)[:max_objs]
                        else:
                            emb = emb[:max_objs]
                        padded_embeddings.append(emb)

                    language_embeddings = torch.stack(padded_embeddings)

                    sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
                    seg_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings)
                    seg_masks = [mask for mask in seg_masks]

        return seg_masks


def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    seg_mask = seg_mask.to(hidden_states.device)
    return hidden_states[-n_out:][seg_mask]

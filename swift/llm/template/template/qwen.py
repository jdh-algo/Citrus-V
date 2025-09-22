# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
# from ..vision_utils import rescale_image, DirectResize, DirectResize_maks
from PIL import Image

from swift.llm import to_device, to_float_dtype
from swift.utils import get_env_args, get_logger, is_deepspeed_enabled
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..template_meta import TemplateMeta
from ..utils import Context, Word, findall
from ..vision_utils import load_audio, load_batch, load_video_ovis2
from .llama import Llama3TemplateMeta
from .utils import DEFAULT_SYSTEM, ChatmlTemplateMeta

logger = get_logger()


@dataclass
class QwenTemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = DEFAULT_SYSTEM
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])
    agent_template: str = 'hermes'


@dataclass
class Qwen2_5TemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'


@dataclass
class Qwen2_5MathTemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = 'Please reason step by step, and put your final answer within \\boxed{}.'


qwq_preview_system = ('You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
                      'You should think step-by-step.')

register_template(QwenTemplateMeta(LLMTemplateType.qwen))
register_template(Qwen2_5TemplateMeta(LLMTemplateType.qwen2_5))
register_template(QwenTemplateMeta(LLMTemplateType.qwq_preview, default_system=qwq_preview_system))


class ThinkingTemplate(Template):

    def _swift_prepare_messages(self, messages):
        super()._swift_prepare_messages(messages)
        # Only during inference or training, and only if the loss_scale is set to 'last_round',
        # will the previous 'think' entries be deleted.
        if not self.is_training or self.loss_scale.name == 'last_round':
            for i, message in enumerate(messages):
                # Delete the content before '</think>' in all assistant turns except the last round.
                if message['role'] == 'assistant' and isinstance(message['content'], str) and i != len(messages) - 1:
                    message['content'] = message['content'].split('</think>')[-1].strip()


register_template(
    QwenTemplateMeta(
        LLMTemplateType.qwq, default_system=None, response_prefix='<think>\n', template_cls=ThinkingTemplate))

# '<think>\n\n</think>\n\n'
register_template(QwenTemplateMeta(LLMTemplateType.qwen3, default_system=None, template_cls=ThinkingTemplate))


class Qwen3RerankerTemplate(Template):
    instruction = 'Given a web search query, retrieve relevant passages that answer the query'

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        super()._preprocess_inputs(inputs)
        query = inputs.messages[-2]['content']
        user_message = '<Instruct>: ' + self.instruction + '\n' + '<Query>: ' + query + '\n' + '<Document>: {doc}'
        inputs.messages[-2]['content'] = user_message


qwen3_reranker_system = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be \"yes\" or \"no\".')

register_template(
    QwenTemplateMeta(
        LLMTemplateType.qwen3_reranker,
        default_system=qwen3_reranker_system,
        response_prefix='<think>\n\n</think>\n\n',
        template_cls=Qwen3RerankerTemplate))

register_template(Qwen2_5MathTemplateMeta(LLMTemplateType.qwen2_5_math))


class QwenPRMTemplate(Template):
    cot_process_placeholder = '<extra_0>'

    def _preprocess_inputs(
        self,
        inputs: StdTemplateInputs,
    ) -> None:
        super()._preprocess_inputs(inputs)
        total_content = '\n'.join([message['content'] or '' for message in inputs.messages])
        if self.cot_process_placeholder not in total_content:
            inputs.messages[-1]['content'] = inputs.messages[-1]['content'] + self.cot_process_placeholder

    @staticmethod
    def make_step_rewards(logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res

    def decode_prm(self, input_ids: torch.Tensor, logits: torch.Tensor) -> Any:
        step_sep_id = self.tokenizer.encode(self.cot_process_placeholder)[0]
        token_masks = (input_ids == step_sep_id)
        return self.make_step_rewards(logits, token_masks)


register_template(Qwen2_5MathTemplateMeta(LLMTemplateType.qwen2_5_math_prm, template_cls=QwenPRMTemplate))


class QwenVLTemplate(Template):
    load_images = False

    @staticmethod
    def _load_image(image, load_images: bool):
        if not load_images and isinstance(image, str) and (image.startswith('data:') or len(image) > 200):
            load_images = True
        return Template._load_image(image, load_images)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        if self.mode == 'lmdeploy':
            return [f'Picture {index + 1}: ', [-100], '\n']
        else:
            image = inputs.images[index]
            if self.mode == 'vllm':
                return [f'Picture {index + 1}: <img></img>\n']
            else:
                assert isinstance(image, str)
                return [f'Picture {index + 1}: <img>{image}</img>\n']

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<ref>{ref}</ref>']

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<box>{self._get_bbox_str(bbox)}</box>']


register_template(QwenTemplateMeta(MLLMTemplateType.qwen_vl, template_cls=QwenVLTemplate))


class QwenAudioTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        audios = inputs.audios
        audio = audios[index]
        assert isinstance(audio, str)
        return [f'Audio {index + 1}:<audio>{audio}</audio>\n']

    def _tokenize(self, context, **tokenizer_kwargs):
        audio_info = self.processor.process_audio(context)
        return super()._tokenize(context, audio_info=audio_info)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        text = ''.join([f'<audio>{audio}</audio>' for audio in inputs.audios])
        audio_info = self.processor.process_audio(text)
        if audio_info:
            tokenizer_kwargs = {'audio_info': audio_info}
            encoded.update(tokenizer_kwargs)
            encoded['tokenizer_kwargs'] = tokenizer_kwargs
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        if batch[0].get('audio_info') is not None:
            res['audio_info'] = [b['audio_info'] for b in batch]
        return res


register_template(QwenTemplateMeta(MLLMTemplateType.qwen_audio, template_cls=QwenAudioTemplate))


class Qwen2AudioTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        if not self.use_chat_template:
            return ['<|audio_bos|><|AUDIO|><|audio_eos|>\n']
        else:
            return [f'Audio {index + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if inputs.audios:
            sampling_rate = get_env_args('sampling_rate', int, self.processor.feature_extractor.sampling_rate)
            audios = load_batch(inputs.audios, load_func=partial(load_audio, sampling_rate=sampling_rate))
            audio_inputs = self.processor.feature_extractor(
                audios, sampling_rate=sampling_rate, return_attention_mask=True, return_tensors='pt')
            audio_inputs['feature_attention_mask'] = audio_inputs.pop('attention_mask')
            encoded.update(audio_inputs)
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_audio, template_cls=Qwen2AudioTemplate))


class Qwen2VLTemplate(Template):
    image_token_id = 151655
    video_token_id = 151656
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    version = 'v2'
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image, fetch_video
        # print(f"======= inputs in replace_tag: {inputs}")
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            if self.mode == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            else:
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            video = inputs.videos[index]
            if os.path.isdir(video):
                video = [os.path.join(video, fname) for fname in os.listdir(video)]
            video = fetch_video({'video': video})
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            inputs.videos[index] = video
            return ['<|vision_start|><|video_pad|><|vision_end|>']

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<|object_ref_start|>{ref}<|object_ref_end|>']

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        # print(f"replace_bbox: {bbox}")
        return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        # print(f"==== using Qwen2VLTemplate._encode")
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        images = inputs.images
        videos = inputs.videos
        for media_type in ['images', 'videos']:
            if locals()[media_type]:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(
                        images=images, videos=None, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    if hasattr(processor, 'video_processor'):
                        processor_func = processor.video_processor
                    else:
                        processor_func = processor.image_processor
                    media_inputs = processor_func(images=None, videos=videos, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                    if self.version == 'v2_5':
                        from qwen_vl_utils import vision_process
                        media_inputs['second_per_grid_ts'] = [
                            processor.image_processor.temporal_patch_size / vision_process.FPS
                        ] * len(media_grid_thw)
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    return [media_token] * token_len

                input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, _get_new_tokens)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded

    def forward_context(self, model, inputs):
        if 'real_position_ids' not in inputs:
            return super().forward_context(model, inputs)
        if self.version == 'v2':
            from transformers.models.qwen2_vl import modeling_qwen2_vl as modeling_module
        elif self.version == 'v2_5':
            from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as modeling_module
        elif self.version == 'omni':
            from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as modeling_module
        position_ids = inputs['position_ids']
        inputs['position_ids'] = inputs.pop('real_position_ids')
        return self._patch_flash_attention_forward(modeling_module, position_ids)

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # print(f"======= using Qwen2VLTemplate._post_encode.....")
        # print(f"======= inputs in Qwen2VL post_encode: {inputs.keys()}")
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        pixel_values = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        image_grid_thw = inputs.get('image_grid_thw')
        video_grid_thw = inputs.get('video_grid_thw')

        base_model = self.get_base_model(model)
        if hasattr(base_model.model, 'embed_tokens'):
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)

        dtype = model.visual.get_dtype() if self.version == 'v2' else model.visual.dtype
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            if is_deepspeed_enabled():
                from PIL import Image
                images = [Image.new('RGB', (32, 32), (0, 0, 0))]
                media_inputs = self.processor.image_processor(images=images, videos=None, return_tensors='pt')
                device = input_ids.device
                media_inputs = to_device(media_inputs, device)
                pixel_values = media_inputs['pixel_values'].type(dtype)
                image_embeds = model.visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
                inputs_embeds += image_embeds.mean() * 0.
        else:
            if pixel_values is not None:
                pixel_values = pixel_values.type(dtype)
                image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(dtype)
                video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == model.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        return {'inputs_embeds': inputs_embeds}

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        second_per_grid_ts = self.gather_list(batch, 'second_per_grid_ts')
        if second_per_grid_ts:
            res['second_per_grid_ts'] = second_per_grid_ts
        for media_type in ['image', 'video']:
            grid_thw = self.concat_tensor(batch, f'{media_type}_grid_thw', 0)
            if grid_thw is not None:
                res[f'{media_type}_grid_thw'] = grid_thw
        # print(f"DEBUG: res in _data_collator_mm_data before concat: {res.keys()}")
        return res

    def packing_row(self, row: List[Tuple[Dict[str, Any], int]]) -> Dict[str, Any]:
        # logger.info(f"===== packing_row in Qwen2VLTemplate: row_count={len(row)}, first_keys={list(row[0][0].keys()) if row else 'empty'}")
        position_ids = []
        for r in row:
            r = r[0].copy()
            r['input_ids'] = torch.tensor(r['input_ids'])[None]
            position_ids.append(self._get_position_ids(r))
        packed = super().packing_row(row)
        packed['real_position_ids'] = torch.concat(position_ids, dim=-1)
        return packed

    def _get_position_ids(self, inputs: Dict[str, Any]):
        # fix https://github.com/huggingface/transformers/pull/33487
        kwargs = {}
        if self.version == 'v2_5':
            kwargs = {'second_per_grid_ts': inputs.get('second_per_grid_ts')}
        base_model = self.get_base_model(self.model)
        if hasattr(base_model, 'get_rope_index'):
            get_rope_index = base_model.get_rope_index
        else:
            get_rope_index = base_model.model.get_rope_index
        position_ids, _ = get_rope_index(
            inputs['input_ids'],
            inputs.get('image_grid_thw'),
            inputs.get('video_grid_thw'),
            attention_mask=inputs.get('attention_mask'),
            **kwargs)
        return position_ids.contiguous()

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        # print(f"====== =getting data ====")
        res = super()._data_collator(batch, padding_to=padding_to)
        # print(f"====== res[0] in Qwen2VLTemplate._data_collator: {res}")
        if self._packing:
            res['real_position_ids'] = self.concat_tensor(batch, 'real_position_ids', -1)
            # print(f"===== after concat real_position_ids: {res['real_position_ids']}")
        elif self.is_training:
            res['position_ids'] = self._get_position_ids(res)

        # print(f"===== res in data_collator: {res.keys()}")

        return res


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_vl, template_cls=Qwen2VLTemplate))

register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qvq,
        default_system=('You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
                        'Answer in the language of the question. You should think step-by-step.'),
        template_cls=Qwen2VLTemplate,
    ))


class Qwen2_5VLTemplate(Qwen2VLTemplate):
    version = 'v2_5'
    norm_bbox = 'none'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加bbox格式控制标志
        self.use_json_bbox_format = True  # 默认使用JSON格式，可通过set_bbox_format修改
        # 存储图像缩放信息，用于调整bbox坐标
        self.image_scale_info = {}

    def set_bbox_format(self, use_json: bool = True):
        """
        设置bbox输出格式

        Args:
            use_json (bool): True为JSON格式 {"bbox_2d": [x1,y1,x2,y2], "label": "..."},
                           False为原格式 <|box_start|>(x1,y1),(x2,y2)<|box_end|>
        """
        self.use_json_bbox_format = use_json

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        # 如果有缩放信息，调整bbox坐标
        if hasattr(self, 'image_scale_info') and index in self.image_scale_info:
            scale_info = self.image_scale_info[index]
            scale_x, scale_y = scale_info['scale']

            # 调整bbox坐标
            adjusted_bbox = [
                int(bbox[0] * scale_x),  # x1
                int(bbox[1] * scale_y),  # y1
                int(bbox[2] * scale_x),  # x2
                int(bbox[3] * scale_y)  # y2
            ]

            # 确保坐标在有效范围内
            processed_width, processed_height = scale_info['processed_size']
            adjusted_bbox[0] = max(0, min(adjusted_bbox[0], processed_width))
            adjusted_bbox[1] = max(0, min(adjusted_bbox[1], processed_height))
            adjusted_bbox[2] = max(adjusted_bbox[0], min(adjusted_bbox[2], processed_width))
            adjusted_bbox[3] = max(adjusted_bbox[1], min(adjusted_bbox[3], processed_height))

            bbox = adjusted_bbox

        # 根据配置选择输出格式
        if self.use_json_bbox_format:
            return self._get_json_bbox_str(bbox, index, inputs)
        else:
            return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']

    def _get_json_bbox_str(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        """将bbox格式化为JSON字符串"""
        import json

        # 获取对应的ref对象作为label
        ref_objects = inputs.objects.get('ref', [])
        # 所有bboxes使用第一个ref对象的label
        first_label = ref_objects[0] if len(ref_objects) > 0 else 'object'

        # 创建JSON格式的bbox字典
        bbox_dict = {
            'bbox_2d': bbox,  # [x1, y1, x2, y2] 格式
            'label': first_label
        }

        # 将单个bbox也包装在数组中，并包裹在```json代码块中
        bbox_array = [bbox_dict]
        # 手动格式化JSON以获得所需的换行符
        json_objects = []
        for bbox_item in bbox_array:
            json_objects.append(json.dumps(bbox_item, ensure_ascii=False, separators=(',', ': ')))
        json_content = '[\n' + ',\n'.join(json_objects) + '\n]'
        formatted_json = f'```json\n{json_content}\n```'

        return [formatted_json]

    def _get_multiple_json_bbox_str(self, bbox_list: List[List[int]], inputs: StdTemplateInputs) -> str:
        """将多个bbox格式化为JSON数组字符串"""
        import json

        ref_objects = inputs.objects.get('ref', [])
        bbox_array = []

        # 获取第一个label，所有bbox都使用相同的label
        first_label = ref_objects[0] if len(ref_objects) > 0 else 'object'

        for i, bbox in enumerate(bbox_list):
            bbox_dict = {
                'bbox_2d': bbox,
                'label': first_label  # 所有bbox使用相同的label
            }
            bbox_array.append(bbox_dict)

        # 转换为JSON字符串，并包裹在```json代码块中
        # 手动格式化JSON以获得所需的换行符
        json_objects = []
        for bbox_item in bbox_array:
            json_objects.append(json.dumps(bbox_item, ensure_ascii=False, separators=(',', ': ')))
        json_content = '[\n' + ',\n'.join(json_objects) + '\n]'
        formatted_json = f'```json\n{json_content}\n```'

        return formatted_json

    def _pre_tokenize(self, context_list: List[Context], loss_scale_list: List[float],
                      inputs: StdTemplateInputs) -> Tuple[List[Context], List[float]]:
        """重写_pre_tokenize方法来处理多个bbox的合并"""

        # 如果不是JSON格式，使用父类的默认处理
        if not self.use_json_bbox_format:
            return super()._pre_tokenize(context_list, loss_scale_list, inputs)

        # 先调用父类的图像预处理
        context_list, loss_scale_list = self._pre_tokenize_images(context_list, loss_scale_list, inputs)
        if inputs.images and inputs.objects:
            self.normalize_bbox(inputs)

        res: List[Context] = []
        res_loss_scale: List[float] = []

        # reset
        for k in ['video', 'audio', 'object', 'box']:
            setattr(inputs, f'{k}_idx', 0)

        # 收集所有的bbox标签位置，为了合并处理
        bbox_positions = []
        for i, context in enumerate(context_list):
            if context == '<bbox>':
                bbox_positions.append(i)

        # 如果有多个bbox，我们需要特殊处理
        bbox_processed = False

        for i, (context, loss_scale) in enumerate(zip(context_list, loss_scale_list)):
            for k in ['video', 'audio']:
                if context == f'<{k}>' and inputs.is_multimodal and getattr(inputs, f'{k}_idx') < len(
                        getattr(inputs, f'{k}s')):
                    c_list = self.replace_tag(k, getattr(inputs, f'{k}_idx'), inputs)
                    setattr(inputs, f'{k}_idx', getattr(inputs, f'{k}_idx') + 1)
                    loss_scale = 0.
                    break
            else:
                ref = inputs.objects.get('ref') or []
                bbox = inputs.objects.get('bbox') or []
                if context == '<ref-object>' and inputs.ref_idx < len(ref):
                    idx = inputs.ref_idx
                    c_list = self.replace_ref(ref[idx], idx, inputs)
                    inputs.ref_idx += 1
                elif context == '<bbox>' and inputs.bbox_idx < len(bbox) and not bbox_processed:
                    # 如果这是第一个bbox标签，处理所有bbox
                    if len(bbox_positions) > 1:
                        # 多个bbox：生成合并的JSON数组
                        c_list = [self._get_multiple_json_bbox_str(bbox, inputs)]
                        # 跳过其他的bbox标签
                        bbox_processed = True
                        inputs.bbox_idx = len(bbox)  # 标记所有bbox都已处理
                    else:
                        # 单个bbox：使用原有逻辑（已经是数组格式）
                        idx = inputs.bbox_idx
                        c_list = self.replace_bbox(bbox[idx], idx, inputs)
                        inputs.bbox_idx += 1
                elif context == '<bbox>' and bbox_processed:
                    # 如果已经处理过多个bbox，跳过后续的bbox标签
                    c_list = []
                elif context == '<cot-process>' and self.mode == 'prm':
                    c_list = self.replace_cot_process(inputs)
                else:
                    c_list = [context]
            res += c_list
            res_loss_scale += [loss_scale] * len(c_list)

        return res, res_loss_scale

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        """重写replace_tag方法来收集图像缩放信息"""
        from qwen_vl_utils import fetch_image, fetch_video

        assert media_type in {'image', 'video'}
        if media_type == 'image':
            # 保存原始图像尺寸
            original_image = inputs.images[index]
            if isinstance(original_image, str):
                from PIL import Image
                original_image = Image.open(original_image)
            original_width, original_height = original_image.size

            # 获取处理后的图像
            processed_image = fetch_image({'image': inputs.images[index]})
            inputs.images[index] = processed_image

            # 计算缩放比例
            processed_width, processed_height = processed_image.size
            scale_x = processed_width / original_width
            scale_y = processed_height / original_height

            # 存储缩放信息
            if not hasattr(self, 'image_scale_info'):
                self.image_scale_info = {}
            self.image_scale_info[index] = {
                'original_size': (original_width, original_height),
                'processed_size': (processed_width, processed_height),
                'scale': (scale_x, scale_y)
            }

            if self.mode == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            else:
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            video = inputs.videos[index]
            if os.path.isdir(video):
                video = [os.path.join(video, fname) for fname in os.listdir(video)]
            video = fetch_video({'video': video})
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            inputs.videos[index] = video
            return ['<|vision_start|><|video_pad|><|vision_end|>']

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        """重写_preprocess_inputs方法来清理缩放信息"""
        # 先调用父类的预处理逻辑
        super()._preprocess_inputs(inputs)

        # 清理缩放信息，避免内存泄漏
        if hasattr(self, 'image_scale_info'):
            self.image_scale_info.clear()


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_5_vl, template_cls=Qwen2_5VLTemplate))

register_template(
    QwenTemplateMeta(
        MLLMTemplateType.mimo_vl,
        template_cls=Qwen2_5VLTemplate,
        default_system='You are MiMo, an AI assistant developed by Xiaomi.'))


class Qwen2_5OmniTemplate(Qwen2_5VLTemplate):
    version = 'omni'
    placeholder_tokens = ['<|IMAGE|>', '<|AUDIO|>', '<|VIDEO|>']

    def init_processor(self, processor) -> None:
        if processor is None:
            return
        super().init_processor(processor)
        from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessorKwargs
        default = Qwen2_5OmniProcessorKwargs._defaults
        self.seconds_per_chunk = default['videos_kwargs']['seconds_per_chunk']
        self.position_id_per_seconds = default['videos_kwargs']['position_id_per_seconds']
        self.use_audio_in_video = get_env_args('use_audio_in_video', bool, False)
        self.sampling_rate = get_env_args('sampling_rate', int, self.processor.feature_extractor.sampling_rate)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_omni_utils import fetch_image, fetch_video
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            return ['<|vision_bos|><|IMAGE|><|vision_eos|>']
        elif media_type == 'audio':
            if self.mode != 'vllm':
                inputs.audios[index] = load_audio(inputs.audios[index], self.sampling_rate)
            return ['<|audio_bos|><|AUDIO|><|audio_eos|>']
        elif media_type == 'video':
            video = inputs.videos[index]
            inputs.videos[index] = fetch_video({'video': video}).to(torch.uint8)
            if self.use_audio_in_video:
                import librosa
                if video.startswith('http://') or video.startswith('https://'):
                    import audioread
                    video = audioread.ffdec.FFmpegAudioFile(video)
                video = librosa.load(video, sr=self.sampling_rate)[0]
                inputs.audios.insert(inputs.audio_idx, (video, 'video'))
                inputs.audio_idx += 1
                return ['<|vision_bos|><|audio_bos|><|VIDEO|><|audio_eos|><|vision_eos|>']
            return ['<|vision_bos|><|VIDEO|><|vision_eos|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        processor = self.processor
        video_audios_mask = []
        for i, audio in enumerate(inputs.audios):
            if isinstance(audio, tuple) and audio[1] == 'video':
                inputs.audios[i] = audio[0]
                video_audios_mask.append(True)
            else:
                video_audios_mask.append(False)
        video_audios_mask = torch.tensor(video_audios_mask)
        media_inputs = processor(
            text='',
            audio=inputs.audios or None,
            images=inputs.images or None,
            videos=inputs.videos or None,
            do_resize=False,
            return_tensors='pt')
        media_inputs.pop('input_ids')
        media_inputs.pop('attention_mask')
        media_inputs = to_float_dtype(media_inputs, self.model_info.torch_dtype)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        # audio
        audio_token_id = self._tokenize('<|AUDIO|>')
        idx_list = findall(input_ids, audio_token_id)
        feature_attention_mask = media_inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            audio_lengths = (((audio_feature_lengths - 1) // 2 + 1 - 2) // 2 + 1)
        else:
            audio_lengths = None
        audio_lengths_origin = audio_lengths
        if idx_list:
            if self.use_audio_in_video:
                audio_lengths = audio_lengths[~video_audios_mask]

            def _get_new_audio_tokens(i):
                return audio_token_id * audio_lengths[i]

            input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, _get_new_audio_tokens)

        for media_type in ['image', 'video']:
            token = f'<|{media_type.upper()}|>'
            token_id = self._tokenize(token)
            idx_list = findall(input_ids, token_id)
            if idx_list:
                merge_size = processor.image_processor.merge_size
                media_grid_thw = media_inputs.get(f'{media_type}_grid_thw')
                if media_type == 'video' and self.use_audio_in_video:
                    audio_lengths = audio_lengths_origin[video_audios_mask]
                    video_second_per_grid = media_inputs['video_second_per_grid']

                    def _get_new_tokens_use_audio_in_video(i):
                        audio_token_indices = torch.arange(audio_lengths[i])
                        grid_thw = media_grid_thw[i]
                        height = grid_thw[1] // merge_size
                        width = grid_thw[2] // merge_size
                        video_token_indices = torch.arange(grid_thw[0]).reshape(-1, 1, 1)
                        video_token_indices = torch.broadcast_to(
                            video_token_indices, (video_token_indices.shape[0], height, width)).reshape(-1)
                        video_token_indices = (
                            video_token_indices * video_second_per_grid[i] * self.position_id_per_seconds)
                        tokens_per_chunk = int(self.position_id_per_seconds * self.seconds_per_chunk)
                        video_chunk_indexes = processor.get_chunked_index(video_token_indices, tokens_per_chunk)
                        audio_chunk_indexes = processor.get_chunked_index(audio_token_indices, tokens_per_chunk)

                        res = []
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            if j < len(video_chunk_indexes):
                                video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                                res += token_id * video_seq_length
                            if j < len(audio_chunk_indexes):
                                audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                                res += audio_token_id * audio_seq_length
                        return res

                    input_ids, labels = self._extend_tokens(input_ids, labels, idx_list,
                                                            _get_new_tokens_use_audio_in_video)

                else:

                    def _get_new_tokens(i):
                        token_len = (media_grid_thw[i].prod() // (merge_size**2))
                        return token_id * token_len

                    input_ids, labels = self._extend_tokens(input_ids, labels, idx_list, _get_new_tokens)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded.update(media_inputs)
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return Template._post_encode(self, model, inputs)

    def _get_position_ids(self, inputs: Dict[str, Any]):
        feature_attention_mask = inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None
        video_second_per_grid = inputs.pop('video_second_per_grid', None)
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids, _ = self.model.thinker.get_rope_index(
            input_ids,
            inputs.get('image_grid_thw'),
            inputs.get('video_grid_thw'),
            attention_mask,
            self.use_audio_in_video,
            audio_feature_lengths,
            video_second_per_grid,
        )
        return position_ids.contiguous()

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        video_second_per_grid = self.gather_list(batch, 'video_second_per_grid')
        if video_second_per_grid:
            res['video_second_per_grid'] = video_second_per_grid
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res

    def generate(self, model, *args, **kwargs):
        if kwargs.get('video_grid_thw') is not None:
            kwargs['use_audio_in_video'] = self.use_audio_in_video
        return super().generate(model, *args, **kwargs)


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_5_omni, template_cls=Qwen2_5OmniTemplate))


class Ovis1_6Template(Template):
    skip_prompt = False
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, [-200])
        added_tokens_len = 0
        pixel_values = []
        for i, idx in enumerate(idx_list):
            max_partition = get_env_args('max_partition', int, 9)
            raw_pixel_values, image_placeholders = self.model.visual_tokenizer.preprocess_image(
                images[i], max_partition=max_partition)
            input_ids = input_ids[:idx] + image_placeholders + input_ids[idx + 1:]
            if labels is not None:
                labels = labels[:idx] + [-100] * len(image_placeholders) + labels[idx + 1:]
            pixel_values.append(raw_pixel_values)
            added_tokens_len += len(image_placeholders) - 1
        dtype = self.model.visual_tokenizer.dtype
        if pixel_values:
            pixel_values = torch.cat(pixel_values, dim=0).to(dtype)
        else:
            pixel_values = torch.zeros((1, 3, 384, 384), dtype=dtype)  # dummpy
        encoded.update({'input_ids': input_ids, 'labels': labels})
        encoded['pixel_values'] = [pixel_values]
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        padding_side = self.padding_side if self.is_training else 'left'
        if self.max_length is not None:
            model.config.multimodal_max_length = self.max_length
        input_ids = inputs['input_ids']
        labels = inputs.get('labels')
        if labels is None:
            labels = input_ids.new_full(input_ids.shape, -100)
        _, inputs_embeds, labels, attention_mask = model.merge_multimodal(
            text_input_ids=input_ids,
            text_attention_masks=torch.ones_like(input_ids),  # not use, only compat
            text_labels=labels,
            pixel_values=inputs['pixel_values'],
            left_padding=padding_side == 'left')
        if inputs.get('labels') is None:
            labels = None
        return {'inputs_embeds': inputs_embeds, 'labels': labels, 'attention_mask': attention_mask}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        pixel_values = self.gather_list(batch, 'pixel_values')
        res = super()._data_collator(batch, padding_to=padding_to)
        res['pixel_values'] = pixel_values
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.ovis1_6,
        prefix=['<bos>'],
        prompt=['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'],
        chat_sep=['<end_of_turn>\n'],
        suffix=['<end_of_turn>'],
        system_prefix=['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'],
        template_cls=Ovis1_6Template,
    ))

register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.ovis1_6_llama3,
        default_system='You are a helpful and honest multimodal assistant.',
        template_cls=Ovis1_6Template,
    ))


class Ovis2Template(Ovis1_6Template):
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    nframes = 12

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return [[-200], '\n']
        elif media_type == 'video':
            nframes = get_env_args('nframes', int, self.nframes)
            inputs.images = load_video_ovis2(inputs.videos[index], nframes)
            return [[-200] * nframes, '\n']


register_template(QwenTemplateMeta(
    MLLMTemplateType.ovis2,
    template_cls=Ovis2Template,
))


@dataclass
class MarcoO1TemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = """
你是一个经过良好训练的AI助手，你的名字是Marco-o1.由阿里国际数字商业集团的AI Business创造.
        \n## 重要！！！！！
当你回答问题时，你的思考应该在<Thought>内完成，<Output>内输出你的结果。
<Thought>应该尽可能是英文，但是有2个特例，一个是对原文中的引用，另一个是是数学应该使用markdown格式，<Output>内的输出需要遵循用户输入的语言。
        """


register_template(MarcoO1TemplateMeta(LLMTemplateType.marco_o1))

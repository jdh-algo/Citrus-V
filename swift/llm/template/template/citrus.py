# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from swift.utils import get_env_args
from ..base import Template
from ..constant import MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context
from ..vision_utils import DirectResize, DirectResize_maks, load_audio, load_batch, rescale_image
from .qwen import Qwen2VLTemplate, QwenTemplateMeta


class CitrusVTemplate(Qwen2VLTemplate, Template):
    version = 'v2_5'
    norm_bbox = 'none'
    seg_token_idx = 151665
    image_size = 1024
    mask_size = 256
    grounding_image_processor = DirectResize(target_size=image_size)
    grounding_mask_processor = DirectResize_maks(target_size=mask_size)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_scale_info = {}  # store image scale info
        # add bbox format control flag
        self.use_json_bbox_format = True  # set to True use JSON format, False use original format

    def set_bbox_format(self, use_json: bool = True):
        """
        set bbox output format

        Args:
            use_json (bool): True use JSON format {"bbox_2d": [x1,y1,x2,y2], "label": "..."},
                           False use original format <|box_start|>(x1,y1),(x2,y2)<|box_end|>
        """
        self.use_json_bbox_format = use_json

    def _preprocess_inputs(
        self,
        inputs: StdTemplateInputs,
    ) -> None:
        self._preprocess_function_call(inputs)

        if self.model_meta.is_multimodal:
            self._replace_image_tags(inputs)
            self._replace_start_image_tags(inputs)
        images = inputs.images
        load_images = self.load_images or self.mode in {'vllm', 'lmdeploy'}
        load_images_origin = load_images

        if self.max_pixels is not None or inputs.objects:
            load_images = True
        if images:
            for i, image in enumerate(images):
                images[i] = self._load_image(images[i], load_images)

        if inputs.objects:
            self._get_height_width(inputs)

        if self.max_pixels is not None:
            # Scale the image proportionally without affecting the scaled objects.
            images = [rescale_image(img, self.max_pixels) for img in images]

        if images and not load_images_origin:  # fix pt & qwen-vl
            for i, image in enumerate(images):
                if isinstance(image, Image.Image):
                    images[i] = self._save_pil_image(image)
        inputs.images = images

        # add mask processing
        if inputs.masks:
            masks = inputs.masks
            for i, mask in enumerate(masks):
                masks[i] = self._load_image(masks[i], load_images)
            if self.max_pixels is not None:
                # Scale the image proportionally without affecting the scaled objects.
                masks = [rescale_image(mask, self.max_pixels) for mask in masks]
            inputs.masks = masks

        if self.mode == 'vllm' and inputs.audios:
            sampling_rate = get_env_args('sampling_rate', int, None)
            inputs.audios = load_batch(
                inputs.audios, load_func=partial(load_audio, sampling_rate=sampling_rate, return_sr=True))

        if inputs.is_multimodal:
            self._add_default_tags(inputs)

        if hasattr(self, 'image_scale_info'):
            self.image_scale_info.clear()

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image, fetch_video
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            # save original image size
            original_image = inputs.images[index]
            if isinstance(original_image, str):
                from PIL import Image
                original_image = Image.open(original_image)
            original_width, original_height = original_image.size

            # get processed image
            processed_image = fetch_image({'image': inputs.images[index]})
            inputs.images[index] = processed_image

            # calculate scale ratio
            processed_width, processed_height = processed_image.size
            scale_x = processed_width / original_width
            scale_y = processed_height / original_height

            # store scale info
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

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<|object_ref_start|>{ref}<|object_ref_end|>']

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        # if there is scale info, adjust bbox coordinates
        if index in self.image_scale_info:
            scale_info = self.image_scale_info[index]
            scale_x, scale_y = scale_info['scale']

            # adjust bbox coordinates
            adjusted_bbox = [
                int(bbox[0] * scale_x),  # x1
                int(bbox[1] * scale_y),  # y1
                int(bbox[2] * scale_x),  # x2
                int(bbox[3] * scale_y)  # y2
            ]

            # ensure coordinates are within valid range
            processed_width, processed_height = scale_info['processed_size']
            adjusted_bbox[0] = max(0, min(adjusted_bbox[0], processed_width))
            adjusted_bbox[1] = max(0, min(adjusted_bbox[1], processed_height))
            adjusted_bbox[2] = max(adjusted_bbox[0], min(adjusted_bbox[2], processed_width))
            adjusted_bbox[3] = max(adjusted_bbox[1], min(adjusted_bbox[3], processed_height))

            bbox = adjusted_bbox

        # according to configuration select output format
        if self.use_json_bbox_format:
            return self._get_json_bbox_str(bbox, index, inputs)
        else:
            return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']

    def _get_json_bbox_str(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        """format bbox to JSON string"""
        import json

        # get corresponding ref object as label
        ref_objects = inputs.objects.get('ref', [])
        label = ref_objects[index] if index < len(ref_objects) else 'object'

        # create JSON format bbox dictionary
        bbox_dict = {
            'bbox_2d': bbox,  # [x1, y1, x2, y2] format
            'label': label
        }

        # wrap single bbox in array and wrap in ```json code block
        bbox_array = [bbox_dict]
        # manually format JSON to get the required newline
        json_objects = []
        for bbox_item in bbox_array:
            json_objects.append(json.dumps(bbox_item, ensure_ascii=False, separators=(',', ': ')))
        json_content = '[\n' + ',\n'.join(json_objects) + '\n]'
        formatted_json = f'```json\n{json_content}\n```'

        return [formatted_json]

    def _get_multiple_json_bbox_str(self, bbox_list: List[List[int]], inputs: StdTemplateInputs) -> str:
        """format multiple bbox to JSON array string"""
        import json

        ref_objects = inputs.objects.get('ref', [])
        bbox_array = []

        # get first label, all bbox use the same label
        first_label = ref_objects[0] if len(ref_objects) > 0 else 'object'

        for i, bbox in enumerate(bbox_list):
            bbox_dict = {
                'bbox_2d': bbox,
                'label': first_label  # all bbox use the same label
            }
            bbox_array.append(bbox_dict)

        # convert to JSON string and wrap in ```json code block
        # manually format JSON to get the required newline
        json_objects = []
        for bbox_item in bbox_array:
            json_objects.append(json.dumps(bbox_item, ensure_ascii=False, separators=(',', ': ')))
        json_content = '[\n' + ',\n'.join(json_objects) + '\n]'
        formatted_json = f'```json\n{json_content}\n```'

        return formatted_json

    def _pre_tokenize(self, context_list: List[Context], loss_scale_list: List[float],
                      inputs: StdTemplateInputs) -> Tuple[List[Context], List[float]]:
        """rewrite _pre_tokenize method to handle multiple bbox merge"""
        from swift.llm.template.base import Template
        from swift.llm.template.utils import Context

        # if not JSON format, use parent class default processing
        if not self.use_json_bbox_format:
            return super()._pre_tokenize(context_list, loss_scale_list, inputs)

        context_list, loss_scale_list = self._pre_tokenize_images(context_list, loss_scale_list, inputs)
        if inputs.images and inputs.objects:
            self.normalize_bbox(inputs)

        res: List[Context] = []
        res_loss_scale: List[float] = []

        # reset
        for k in ['video', 'audio', 'object', 'box']:
            setattr(inputs, f'{k}_idx', 0)

        # collect all bbox label positions, to handle multiple bbox merge
        bbox_positions = []
        for i, context in enumerate(context_list):
            if context == '<bbox>':
                bbox_positions.append(i)

        # if there are multiple bbox, we need special processing
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
                    # if this is the first bbox label, process all bbox
                    if len(bbox_positions) > 1:
                        # multiple bbox: generate merged JSON array
                        c_list = [self._get_multiple_json_bbox_str(bbox, inputs)]
                        # skip other bbox labels
                        bbox_processed = True
                        inputs.bbox_idx = len(bbox)  # mark all bbox as processed
                    else:
                        # single bbox: use original logic (already in array format)
                        idx = inputs.bbox_idx
                        c_list = self.replace_bbox(bbox[idx], idx, inputs)
                        inputs.bbox_idx += 1
                elif context == '<bbox>' and bbox_processed:
                    # if already processed multiple bbox, skip subsequent bbox labels
                    c_list = []
                elif context == '<cot-process>' and self.mode == 'prm':
                    c_list = self.replace_cot_process(inputs)
                else:
                    c_list = [context]
            res += c_list
            res_loss_scale += [loss_scale] * len(c_list)

        return res, res_loss_scale

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images

        # masks
        new_masks = []
        if inputs.masks:
            # mask
            masks = inputs.masks
            for mask in masks:
                mask = self.grounding_mask_processor(mask)  # 1, 256, 256
                new_masks.append(mask[None])  # [1, 1, 256, 256]
            new_masks = np.concatenate(new_masks, axis=0)  # num, 1, 256, 256
            new_masks = torch.from_numpy(new_masks)
            encoded['masks'] = new_masks
            # g_pixel_value
            grounding_pixel_values = self.grounding_image_processor(images[0])
            encoded['g_pixel_values'] = grounding_pixel_values  # 1, 3, 1024, 1024
        else:  # no mask in data
            # mask
            new_masks = torch.from_numpy(np.zeros((0, 1, self.mask_size, self.mask_size)).astype(np.uint8))
            encoded['masks'] = new_masks
            # g_pixel_values
            grounding_pixel_values = torch.from_numpy(
                np.zeros((0, 3, self.image_size, self.image_size)).astype(np.float32))
            encoded['g_pixel_values'] = grounding_pixel_values  # 0, 3, 1024, 1024

        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]], is_batch: bool = True) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)

        # process SAM specific multi-modal fields
        second_per_grid_ts = self.gather_list(batch, 'second_per_grid_ts')
        if second_per_grid_ts:
            res['second_per_grid_ts'] = second_per_grid_ts
        for media_type in ['image', 'video']:
            grid_thw = self.concat_tensor(batch, f'{media_type}_grid_thw', 0)
            if grid_thw is not None:
                res[f'{media_type}_grid_thw'] = grid_thw

        # fix index mismatch: ensure each sample has corresponding tensor
        g_pixel_values = []
        masks = []

        for _, b in enumerate(batch):
            if is_batch:
                g_pixel_values.append(b['g_pixel_values'].unsqueeze(0))
                masks.append(b['masks'].unsqueeze(0))
            else:
                g_pixel_values.append(b['g_pixel_values'])
                masks.append(b['masks'])

        res['g_pixel_values'] = torch.concat(g_pixel_values, dim=0)
        res['masks'] = torch.concat(masks, dim=0)

        assert res['g_pixel_values'].shape[0] == res['masks'].shape[0], 'currently only support one mask per image.'

        return res

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:

        return inputs

    def packing_row(self, row: List[Tuple[Dict[str, Any], int]]) -> Dict[str, Any]:
        position_ids = []
        for r in row:
            r = r[0].copy()
            r['input_ids'] = torch.tensor(r['input_ids'])[None]
            position_ids.append(self._get_position_ids(r))

        packed = {}
        keys = set()
        for r in row:
            keys.update(r[0].keys())
        for key in keys:
            if key in {'input_ids', 'labels', 'loss_scale'}:
                packed[key] = sum((x[0][key] for x in row), start=[])
            elif key == 'length':
                packed[key] = sum((x[0][key] for x in row))
            elif key == 'channel':
                packed[key] = [x[0][key] for x in row]
        if 'position_ids' not in packed:
            packed['position_ids'] = sum((list(range(x[1])) for x in row), start=[])

        packed.update(self._data_collator_mm_data([r[0] for r in row], is_batch=False))
        packed['real_position_ids'] = torch.concat(position_ids, dim=-1)
        return packed

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)

        return res


class CitrusVInferTemplate(CitrusVTemplate, Template):

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        encoded.pop('masks', None)
        encoded.pop('g_pixel_values', None)

        if inputs.images:
            grounding_pixel_values = self.grounding_image_processor(inputs.images[0])
            encoded['g_pixel_values'] = grounding_pixel_values  # 1, 3, 1024, 1024
        else:
            grounding_pixel_values = torch.from_numpy(np.zeros((0, 3, self.image_size, self.image_size)).astype(np.float32))
            encoded['g_pixel_values'] = grounding_pixel_values  # 0, 3, 1024, 1024

        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]], is_batch: bool = True) -> Dict[str, Any]:
        res = Template._data_collator_mm_data(self, batch)
        second_per_grid_ts = self.gather_list(batch, 'second_per_grid_ts')
        if second_per_grid_ts:
            res['second_per_grid_ts'] = second_per_grid_ts
        for media_type in ['image', 'video']:
            grid_thw = self.concat_tensor(batch, f'{media_type}_grid_thw', 0)
            if grid_thw is not None:
                res[f'{media_type}_grid_thw'] = grid_thw

        g_pixel_values = [b['g_pixel_values'] for b in batch if b.get('g_pixel_values') is not None]
        if len(g_pixel_values) > 0:
            res['g_pixel_values'] = torch.concat(g_pixel_values)
        masks = [b['masks'] for b in batch if b.get('masks') is not None]
        if len(masks) > 0:
            res['masks'] = torch.concat(masks)

        return res


register_template(QwenTemplateMeta(
    MLLMTemplateType.citrus_v,
    template_cls=CitrusVTemplate,
))

register_template(QwenTemplateMeta(
    MLLMTemplateType.citrus_v_infer,
    template_cls=CitrusVInferTemplate,
))

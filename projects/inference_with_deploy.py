import io
import os
import re

import json
import matplotlib.pyplot as plt
import numpy as np
import requests
from openai import OpenAI
from PIL import Image
from pycocotools import mask as mask_utils


def rle_to_mask(rle):
    """
    rle: COCO RLE dict, counts为str
    return: numpy array, shape (H, W), dtype uint8
    """
    print(f'rle: {rle}')
    mask = mask_utils.decode(rle)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return mask


def postprocess_response(resp, img):
    """postprocess the response, extract the segmentation masks"""
    raw_response = getattr(resp, 'raw_response', None)
    if raw_response is None and hasattr(resp, 'to_dict'):
        raw_response = resp.to_dict().get('raw_response', None)
    if raw_response is None and isinstance(resp, dict):
        raw_response = resp

    masks = None
    if raw_response is not None and 'seg_masks_rle' in raw_response:
        seg_masks_rle = raw_response['seg_masks_rle']
        if seg_masks_rle and len(seg_masks_rle) > 0:
            masks = [rle_to_mask(rle) for rle in seg_masks_rle]
            return img, masks
        else:
            # no mask, return img
            return img, None
    else:
        # no mask, return img
        return img, None


def extract_bbox_from_response(response, bbox_format='standard', original_image_size=None, processed_image_size=None):
    """Extract bounding box from model response by trying multiple formats in order"""

    predict_bbox = None

    # method 1: standard format <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    box_pattern = r'<\|box_start\|>\(([\d]+),([\d]+)\),\(([\d]+),([\d]+)\)<\|box_end\|>'
    matches = re.findall(box_pattern, response)
    if matches:
        predict_bbox = []
        for match in matches:
            x1, y1, x2, y2 = int(match[0]), int(match[1]), int(match[2]), int(match[3])
            predict_bbox.append([float(x1), float(y1), float(x2), float(y2)])

    # method 2: JSON format, adapt to multiple results [{"bbox_2d": [...]}, ...] or {"bbox_2d": [...]}
    if predict_bbox is None:
        try:
            # only adapt to the following format: response is a string, content is a or multiple dictionaries, each dictionary has a "bbox_2d" field
            # for example: '{"bbox_2d": [x1, y1, x2, y2]}, {"bbox_2d": [x1, y1, x2, y2]}'
            # use regex to extract all dictionaries
            json_pattern = r'\{[^{}]*?"bbox_2d"\s*:\s*\[[^\[\]]+\][^{}]*?\}'
            json_matches = re.findall(json_pattern, response)
            bboxes = []
            for json_str in json_matches:
                try:
                    bbox_data = json.loads(json_str)
                    if 'bbox_2d' in bbox_data:
                        bbox_coords = bbox_data['bbox_2d']
                        if isinstance(bbox_coords, list) and len(bbox_coords) == 4:
                            bboxes.append([
                                float(bbox_coords[0]),
                                float(bbox_coords[1]),
                                float(bbox_coords[2]),
                                float(bbox_coords[3])
                            ])
                except Exception:
                    continue
            predict_bbox = bboxes
        except Exception as e:
            print(f'Error parsing JSON bbox: {e}')
            predict_bbox = None

    # if the image size information is provided, convert the coordinates
    if predict_bbox is not None and original_image_size is not None and processed_image_size is not None:
        orig_width, orig_height = original_image_size
        proc_width, proc_height = processed_image_size

        # calculate the scale
        scale_x = orig_width / proc_width
        scale_y = orig_height / proc_height

        # convert the coordinates
        converted_bbox = []
        for bbox in predict_bbox:
            x1, y1, x2, y2 = bbox
            # convert the processed coordinates back to the original image coordinates
            orig_x1 = int(x1 * scale_x)
            orig_y1 = int(y1 * scale_y)
            orig_x2 = int(x2 * scale_x)
            orig_y2 = int(y2 * scale_y)

            # ensure the coordinates are in the valid range
            orig_x1 = max(0, min(orig_x1, orig_width))
            orig_y1 = max(0, min(orig_y1, orig_height))
            orig_x2 = max(orig_x1, min(orig_x2, orig_width))
            orig_y2 = max(orig_y1, min(orig_y2, orig_height))

            converted_bbox.append([orig_x1, orig_y1, orig_x2, orig_y2])

        predict_bbox = converted_bbox

    return predict_bbox


def visualize_image(img, masks=None, predict_bbox=None):
    """visualize the segmentation result"""
    orig_W, orig_H = img.size

    # create a new image for display
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # ax.set_title('Segmentation/Detection Result')
    ax.imshow(img)

    if masks is not None and len(masks) > 0:
        # resize masks
        masks_resized = [Image.fromarray(mask).resize((orig_W, orig_H), resample=Image.NEAREST) for mask in masks]
        mask_sum = np.zeros((orig_H, orig_W), dtype=np.uint8)
        for mask in masks_resized:
            mask_array = np.array(mask)
            mask_sum = np.maximum(mask_sum, mask_array)
        if np.any(mask_sum > 0):
            colored_mask = np.zeros((orig_H, orig_W, 4), dtype=np.float32)
            cmap = plt.get_cmap('jet')
            mask_colors = cmap(np.ones(mask_sum.shape))
            colored_mask[mask_sum > 0] = mask_colors[mask_sum > 0]
            colored_mask[..., 3] = mask_sum * 0.35
            ax.imshow(colored_mask)

    # process the bbox
    if predict_bbox is not None:
        for bbox in predict_bbox:
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=4, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

    ax.axis('off')
    plt.tight_layout()

    # convert the matplotlib image to a PIL image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    result_img = Image.open(buf)
    plt.close()

    return result_img


def get_visualization_result(image, resp):
    image, masks = postprocess_response(resp, image)

    processed_image_size = None
    original_image_size = image.size
    if hasattr(resp, 'raw_response') and resp.raw_response:
        raw_response = resp.raw_response
        print(f'raw_response: {raw_response}')
        if isinstance(raw_response, dict) and 'image_grid_thw' in raw_response:
            image_grid_thw = raw_response['image_grid_thw']
            if image_grid_thw and len(image_grid_thw) > 0:
                processed_height = image_grid_thw[0][1] * 28  # 28 是 patch size
                processed_width = image_grid_thw[0][2] * 28
                processed_image_size = (processed_width, processed_height)
    if processed_image_size is None:
        from qwen_vl_utils.vision_process import smart_resize, MIN_PIXELS, MAX_PIXELS, IMAGE_FACTOR

        env_max_pixels = int(os.environ.get('MAX_PIXELS', MAX_PIXELS))
        env_min_pixels = int(os.environ.get('MIN_PIXELS', MIN_PIXELS))

        orig_width, orig_height = original_image_size
        processed_height, processed_width = smart_resize(
            orig_height, orig_width, factor=IMAGE_FACTOR, min_pixels=env_min_pixels, max_pixels=env_max_pixels)
        processed_image_size = (processed_width, processed_height)
    predict_bbox = extract_bbox_from_response(
        response, 'standard', original_image_size=original_image_size, processed_image_size=processed_image_size)

    if masks is not None or predict_bbox is not None:
        result_image = visualize_image(image, masks, predict_bbox)
    else:
        result_image = img

    return result_image


if __name__ == '__main__':
    client = OpenAI(
        api_key='EMPTY',
        base_url=f'http://127.0.0.1:8000/v1',
    )
    model = client.models.list().data[0].id

    img_url = 'asset/test_0001.png'
    if img_url.startswith('http'):
        image = Image.open(io.BytesIO(requests.get(img_url).content)).convert('RGB')
    else:
        image = Image.open(img_url).convert('RGB')

    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': img_url},
            {
                'type': 'text',
                'text': 'please help segment the nucleus in this scan.'
            }
        ]
    }]

    resp = client.chat.completions.create(model=model, messages=messages, max_tokens=512, temperature=0)
    query = messages[0]['content']
    response = resp.choices[0].message.content
    print(response)

    result_image = get_visualization_result(image, resp)
    result_image.save('test.png')
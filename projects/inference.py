import io
import os
import re
import time

import json
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from pycocotools import mask as mask_utils

from swift.llm import InferRequest, PtEngine, RequestConfig
from swift.llm import get_template
from swift.llm import get_model_tokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['MAX_PIXELS'] = '65535'
os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ['FPS_MAX_FRAMES'] = '12'

def init_engine(model_path, max_batch_size=1, template='citrus_v_infer'):

    _, processor = get_model_tokenizer(model_path, load_model=False)
    template = get_template(template, processor)
    template.max_length = 2048

    engine = PtEngine(model_path, max_batch_size=max_batch_size, template=template)
    return engine

def rle_to_mask(rle):
    """
    rle: COCO RLE dict, counts为str
    return: numpy array, shape (H, W), dtype uint8
    """
    mask = mask_utils.decode(rle)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return mask


def extract_bbox_from_response(response, bbox_format='standard'):
    """Extract bounding box from model response by trying multiple formats in order"""

    predict_bbox = None

    box_pattern = r'<\|box_start\|>\(([\d]+),([\d]+)\),\(([\d]+),([\d]+)\)<\|box_end\|>'
    matches = re.findall(box_pattern, response)
    if matches:
        predict_bbox = []
        for match in matches:
            x1, y1, x2, y2 = int(match[0]), int(match[1]), int(match[2]), int(match[3])
            predict_bbox.append([float(x1), float(y1), float(x2), float(y2)])

    if predict_bbox is None:
        try:
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

    return predict_bbox

def postprocess_response(resp, infer_request):
    raw_response = getattr(resp, 'raw_response', None)
    if raw_response is None and hasattr(resp, 'to_dict'):
        raw_response = resp.to_dict().get('raw_response', None)
    if raw_response is None and isinstance(resp, dict):
        raw_response = resp

    img = None
    masks = None
    if infer_request.images:
        img_url = infer_request.images[0]
        if img_url.startswith('http'):
            img = Image.open(io.BytesIO(requests.get(img_url).content)).convert('RGB')
        else:
            img = Image.open(img_url).convert('RGB')

    if raw_response is not None and 'seg_masks_rle' in raw_response:
        seg_masks_rle = raw_response['seg_masks_rle']
        if seg_masks_rle and len(seg_masks_rle) > 0:
            masks = [rle_to_mask(rle) for rle in seg_masks_rle]
            return img, masks
        else:
            return img, None
    else:
        return img, None


def visualize_segmentation(img, masks=None, predict_bbox=None, save_path='segmentation_result.png'):
    orig_W, orig_H = img.size
    plt.figure(figsize=(8, 8))
    plt.title('Segmentation/Detection Result')
    plt.imshow(img)

    if masks is not None and len(masks) > 0:
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
            colored_mask[..., 3] = mask_sum * 0.5 
            plt.imshow(colored_mask)

    if predict_bbox is not None:
        for bbox in predict_bbox:
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument('--model', type=str, default="/path/to/Citrus-V-1.0-8B", help='Path to the model')
    parser.add_argument('--image_path', type=str, default="../asset/ct_00--MSD_Liver--liver_123--y_0175.png", help='Path to the input image (optional)')
    parser.add_argument('--query', type=str, default='<image>\nWhat can you see in this scan. think step by step', help='Query string for inference')
    parser.add_argument('--template', type=str, default='citrus_v_infer', help='Template')
    args = parser.parse_args()

    model = args.model
    image_path = args.image_path
    query = args.query
    has_image = image_path is not None and os.path.isfile(image_path)

    engine = init_engine(model, max_batch_size=1, template=args.template)
    request_config = RequestConfig(
        max_tokens=1024,
        temperature=0,
    )

    if has_image:
        infer_requests = [
            InferRequest(messages=[{
                'role': 'user',
                'content': query
            }], images=[image_path]),
        ]
    else:
        infer_requests = [
            InferRequest(messages=[{
                'role': 'user',
                'content': query
            }], images=[]),
        ]

    start_time = time.time()
    resp_list = engine.infer(infer_requests, request_config)

    print(f'===========')
    print(f'answer: {resp_list[0].choices[0].message.content}')
    print(f'time: {time.time() - start_time}s')
    print(f'===========')

    img, masks = postprocess_response(resp_list[0], infer_requests[0])
    predict_bbox = extract_bbox_from_response(resp_list[0].choices[0].message.content, 'standard')
    if (masks is not None or predict_bbox is not None) and img is not None:
        visualize_segmentation(img, masks, predict_bbox)

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
CitrusV Segmentation/Detection Gradio Interface (Based on API Version)
Based on the functionality of app.py, but using the deployed interface for inference, improving inference speed
"""

import io
import os
import re
import time

import gradio as gr
import json
import matplotlib.pyplot as plt
import numpy as np
import requests
from openai import OpenAI
from PIL import Image
from pycocotools import mask as mask_utils

# environment variables
os.environ['MAX_PIXELS'] = '65535'
os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ['FPS_MAX_FRAMES'] = '12'

# global variable to store the client instance
global_client = None

# predefined model configurations
MODEL_CONFIGS = {
    'CitrusV_8B': {
        'model_id': 'CitrusV_8B',
        'description': 'CitrusV 8B 模型'
    },
}

# default model configuration
DEFAULT_MODEL_CONFIG = MODEL_CONFIGS['CitrusV_8B']

# deployment interface configuration
API_BASE_URL = 'http://127.0.0.1:8000/v1'
API_KEY = 'EMPTY'


def init_client():
    """Initializing OpenAI client"""
    global global_client

    if global_client is None:
        print(f'Initializing API client...')
        print(f'API address: {API_BASE_URL}')
        global_client = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE_URL,
        )
        print('API client initialized!')

    return global_client


def get_available_models():
    """Getting available model list"""
    client = init_client()
    try:
        models = client.models.list()
        available_models = [model.id for model in models.data]
        print(f'Available models: {available_models}')
        return available_models
    except Exception as e:
        print(f'Failed to get model list: {e}')
        return [DEFAULT_MODEL_CONFIG['model_id']]


def rle_to_mask(rle):
    """
    rle: COCO RLE dict, counts为str
    return: numpy array, shape (H, W), dtype uint8
    """
    mask = mask_utils.decode(rle)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return mask


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

    # method 2: JSON format, compatible with multiple results [{"bbox_2d": [...]}, ...] or {"bbox_2d": [...]}
    if predict_bbox is None:
        try:
            # only compatible with the following format: response is a string, containing one or more dictionaries, each dictionary has a "bbox_2d" field
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

    # if the image size information is provided, perform coordinate conversion
    if predict_bbox is not None and original_image_size is not None and processed_image_size is not None:
        orig_width, orig_height = original_image_size
        proc_width, proc_height = processed_image_size

        # calculate the scaling factor
        scale_x = orig_width / proc_width
        scale_y = orig_height / proc_height

        # convert coordinates
        converted_bbox = []
        for bbox in predict_bbox:
            x1, y1, x2, y2 = bbox
            # convert the processed coordinates back to the original image coordinates
            orig_x1 = int(x1 * scale_x)
            orig_y1 = int(y1 * scale_y)
            orig_x2 = int(x2 * scale_x)
            orig_y2 = int(y2 * scale_y)

            # ensure the coordinates are within the valid range
            orig_x1 = max(0, min(orig_x1, orig_width))
            orig_y1 = max(0, min(orig_y1, orig_height))
            orig_x2 = max(orig_x1, min(orig_x2, orig_width))
            orig_y2 = max(orig_y1, min(orig_y2, orig_height))

            converted_bbox.append([orig_x1, orig_y1, orig_x2, orig_y2])

        predict_bbox = converted_bbox

    return predict_bbox


def postprocess_response(resp, infer_request):
    """Postprocess the response, extract the segmentation masks"""
    raw_response = getattr(resp, 'raw_response', None)
    if raw_response is None and hasattr(resp, 'to_dict'):
        raw_response = resp.to_dict().get('raw_response', None)
    if raw_response is None and isinstance(resp, dict):
        raw_response = resp

    img = None
    masks = None

    # get the original image
    img_data = infer_request['images'][0]
    if isinstance(img_data, str):
        # if the image data is a file path or URL
        if img_data.startswith('http'):
            img = Image.open(io.BytesIO(requests.get(img_data).content)).convert('RGB')
        else:
            img = Image.open(img_data).convert('RGB')
    else:
        # if the image data is already a PIL image object
        img = img_data.convert('RGB')

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


def visualize_segmentation(img, masks=None, predict_bbox=None):
    """Visualize the segmentation results"""
    orig_W, orig_H = img.size

    # create a new image for displaying
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)

    # first process the mask
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
            colored_mask[..., 3] = mask_sum * 0.5
            ax.imshow(colored_mask)

    if predict_bbox is not None:
        for bbox in predict_bbox:
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
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


def process_inference(image, query, selected_model):
    """Process the inference request"""
    # check the input
    if image is None:
        return None, 'Please upload an image', ''

    if not query or query.strip() == '':
        return None, 'Please input the query text', ''

    # get the selected model configuration
    model_config = MODEL_CONFIGS.get(selected_model)
    if model_config is None:
        return None, f'Unknown model: {selected_model}', ''

    # initialize the client
    client = init_client()

    # save the original image size information
    original_image_size = image.size  # (width, height)

    # convert the image to base64 encoding
    import base64
    from io import BytesIO

    # convert the PIL image to base64 encoding
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # build the inference request
    messages = [{
        'role':
        'user',
        'content': [{
            'type': 'image',
            'image': f'data:image/png;base64,{img_base64}'
        }, {
            'type': 'text',
            'text': query
        }]
    }]

    # inference
    start_time = time.time()
    try:
        resp = client.chat.completions.create(
            model=model_config['model_id'], messages=messages, max_tokens=1024, temperature=0)
        inference_time = time.time() - start_time
    except Exception as e:
        return None, f'Inference failed: {str(e)}', ''

    # get the text response
    text_response = resp.choices[0].message.content

    # postprocess and visualize
    img, masks = postprocess_response(resp, {'images': [image]})

    # get the processed image size
    processed_image_size = None

    # try to extract the image processing information from the response
    if hasattr(resp, 'raw_response') and resp.raw_response:
        raw_response = resp.raw_response
        if isinstance(raw_response, dict) and 'image_grid_thw' in raw_response:
            image_grid_thw = raw_response['image_grid_thw']
            if image_grid_thw and len(image_grid_thw) > 0:
                processed_height = image_grid_thw[0][1] * 28  # 28 是 patch size
                processed_width = image_grid_thw[0][2] * 28
                processed_image_size = (processed_width, processed_height)

    # if cannot get the processed image size from the response, use the default value
    if processed_image_size is None:
        from qwen_vl_utils.vision_process import smart_resize, MIN_PIXELS, MAX_PIXELS, IMAGE_FACTOR

        env_max_pixels = int(os.environ.get('MAX_PIXELS', MAX_PIXELS))
        env_min_pixels = int(os.environ.get('MIN_PIXELS', MIN_PIXELS))

        orig_width, orig_height = original_image_size
        processed_height, processed_width = smart_resize(
            orig_height, orig_width, factor=IMAGE_FACTOR, min_pixels=env_min_pixels, max_pixels=env_max_pixels)
        processed_image_size = (processed_width, processed_height)

    # extract bbox and perform coordinate conversion
    predict_bbox = extract_bbox_from_response(
        text_response, 'standard', original_image_size=original_image_size, processed_image_size=processed_image_size)

    # visualize the results
    if masks is not None or predict_bbox is not None:
        result_image = visualize_segmentation(img, masks, predict_bbox)
    else:
        # if no mask or bbox is detected, return the original image
        result_image = img

    return result_image, text_response, f'推理时间: {inference_time:.2f}秒'


def test_connection():
    """Test the API connection"""
    try:
        client = init_client()
        models = client.models.list()
        return f'✅ Connection successful! Available models: {[model.id for model in models.data]}'
    except Exception as e:
        return f'❌ Connection failed: {str(e)}'


def create_interface():
    """Create the Gradio interface"""

    with gr.Blocks(title='CitrusV Segmentation/Detection Interface (API Version)', theme=gr.themes.Soft()) as demo:
        gr.Markdown('# 🎯 CitrusV Segmentation/Detection Interface (API Version)')
        gr.Markdown(
            'Upload an image and input the query text, then use the deployed API interface to perform segmentation or detection tasks'
        )

        # connection status display
        connection_status = gr.Markdown('🔄 Testing API connection...')

        # test the connection when the interface is created
        print('Testing API connection...')
        status_text = test_connection()
        print(f'Connection status: {status_text}')
        # update the connection status
        connection_status.value = status_text

        with gr.Row():
            with gr.Column(scale=1):
                # input area
                gr.Markdown('## 📤 Input')

                # model selection dropdown
                model_selector = gr.Dropdown(
                    label='Model',
                    choices=list(MODEL_CONFIGS.keys()),
                    value=list(MODEL_CONFIGS.keys())[0],
                    info='Select the model to use')

                # test connection button
                test_button = gr.Button('🔗 Test Connection', variant='secondary', size='sm')

                # connection status display
                test_status = gr.Textbox(label='Connection Status', value='Waiting for test...', interactive=False)

                # image upload
                image_input = gr.Image(label='Upload Image', type='pil', height=300)
                gr.Markdown('Support PNG, JPG, etc. common image formats')

                # query text input
                query_input = gr.Textbox(
                    label='Query Text',
                    placeholder='Please input the query text, for example: Please help segment the liver in this scan',
                    lines=3,
                    value='Please help segment the liver in this scan')
                gr.Markdown('It is recommended to include the <image> tag in the query')

                # inference button
                infer_button = gr.Button('🚀 Start Inference', variant='primary', size='lg')

                # inference time display
                time_output = gr.Textbox(label='Inference Time', interactive=False, value='Waiting for inference...')

            with gr.Column(scale=1):
                gr.Markdown('## 📊 Results')

                # 结果图像显示
                result_image = gr.Image(label='Results', height=300)
                gr.Markdown('Display the results')

                # text response display
                text_output = gr.Textbox(label='Model Response', lines=5, interactive=False)
                gr.Markdown('The text response from the model')

        # 示例区域
        with gr.Accordion('💡 Usage Examples', open=False):
            gr.Markdown("""
            ### Common Query Examples:
            - `Please help segment the liver in this scan`
            - `Please find the lung and return their locations in the form of coordinates.`
            - `Segment the heart in this image`
            - `Detect all lesions in this medical image`
            - `Please identify and segment the tumor in this CT scan`
            """)

        # bind the inference function
        infer_button.click(
            fn=process_inference,
            inputs=[image_input, query_input, model_selector],
            outputs=[result_image, text_output, time_output])

        # bind the test connection function
        def test_connection_wrapper():
            status = test_connection()
            return status, status

        test_button.click(fn=test_connection_wrapper, inputs=[], outputs=[test_status, connection_status])

        # add some examples
        gr.Examples(
            examples=[
                [
                    '../asset/ct_00--MSD_Liver--liver_123--y_0175.png', 'Please help segment the liver in this scan.',
                    list(MODEL_CONFIGS.keys())[0]
                ],
                [
                    '../asset/ct_00--MSD_Liver--liver_123--y_0175.png',
                    'Please help segment the left lung in the given image.',
                    list(MODEL_CONFIGS.keys())[0]
                ],
                [
                    '../asset/ct_00--MSD_Liver--liver_123--y_0175.png',
                    'Please locate and detect the right lung in this image.',
                    list(MODEL_CONFIGS.keys())[0]
                ],
                [
                    '../asset/ct_00--MSD_Liver--liver_123--y_0175.png',
                    'Please detect the kidney in this image, and return the cooresponging coordinates.',
                    list(MODEL_CONFIGS.keys())[0]
                ],
                [
                    '../asset/test_0007.png', 'Please provide a mask for lung infections in this image',
                    list(MODEL_CONFIGS.keys())[0]
                ],
                [
                    '../asset/mr_00--ACDC--patient018_frame01--x_0000.png',
                    'Please detect the heart in this image using json format.',
                    list(MODEL_CONFIGS.keys())[0]
                ],
                [
                    '../asset/mr_00--ACDC--patient018_frame01--x_0000.png',
                    'Please detect the heart in this image using json format.',
                    list(MODEL_CONFIGS.keys())[0]
                ],
                [
                    '../asset/test_0167.png', 'Please help segment the abnormal area in this scan.',
                    list(MODEL_CONFIGS.keys())[0]
                ],
                [
                    '../asset/test_0001.png', 'Please help segment the nucleus in this scan.',
                    list(MODEL_CONFIGS.keys())[0]
                ],
                [
                    '../asset/test_0075.png', 'Please help segment the abnormal regions in this scan.',
                    list(MODEL_CONFIGS.keys())[0]
                ],
            ],
            inputs=[image_input, query_input, model_selector],
            label='examples')

    return demo


if __name__ == '__main__':
    demo = create_interface()
    demo.launch(server_name='0.0.0.0', server_port=7863, share=True, debug=True, allowed_paths=['../asset'])

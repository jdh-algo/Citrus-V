import argparse
import os
import random
import re
import time

# 添加可视化相关的导入
import cv2
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops.boxes import box_area
from tqdm import tqdm

from swift.llm import InferRequest, PtEngine, RequestConfig

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
# os.environ['MAX_PIXELS'] = '401408'
os.environ['MAX_PIXELS'] = '1003520'
# os.environ['MIN_PIXELS'] = '200704'
os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ['FPS_MAX_FRAMES'] = '12'

# Dataset collections
ds_collections = {
    'med_sam2_eval':
    '/mnt/workspace/offline/caoxuyang5/code/ms-swift/data/evaluate_data/evaluation_dataset_with_id.jsonl',
    'ref_coco_testA': '/mnt/afs/xuyangcao/shared_datasets/RefCOCO/data/testA-00000-of-00001.jsonl',
    'med_sam2_eval_modified': '/mnt/afs/xuyangcao/code/ms-swift/data/MeCoVQA_Grounding_test_merged_modified_v1.jsonl'
}


def box_iou(boxes1, boxes2):
    """Calculate IoU between two sets of boxes"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def visualize_detection_results(image_path, gt_bbox, pred_bbox, iou, save_path, sentence):
    """
    可视化检测结果

    Args:
        image_path: 原始图像路径
        gt_bbox: 真实bbox [x1, y1, x2, y2]
        pred_bbox: 预测bbox [x1, y1, x2, y2]
        iou: IoU值
        save_path: 保存路径
        sentence: 检测目标的标签文本
    """
    try:
        # 读取图像
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # 获取图像尺寸
        img_width, img_height = image.size

        # 设置字体（如果可用）
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
        except:
            font = ImageFont.load_default()

        # 绘制真实bbox（绿色）
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
        draw.rectangle([gt_x1, gt_y1, gt_x2, gt_y2], outline=(0, 255, 0), width=3)

        # 添加 sentence 标签到 GT bbox
        if sentence:
            # 计算标签位置
            label_text = f'GT: {sentence}'
            bbox_text = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]

            # 尝试将标签放在 bbox 上方
            label_x = max(0, gt_x1)
            label_y = max(0, gt_y1 - text_height - 5)

            # 如果标签超出图像上边界，则放在 bbox 下方
            if label_y < 0:
                label_y = gt_y2 + 5
                # 如果还是超出下边界，则放在 bbox 内部
                if label_y + text_height > img_height:
                    label_y = gt_y1 + 5

            # 绘制标签背景
            draw.rectangle([label_x, label_y, label_x + text_width + 10, label_y + text_height + 5], fill=(0, 255, 0))
            # 绘制标签文本
            draw.text((label_x + 5, label_y + 2), label_text, fill=(255, 255, 255), font=font)
        else:
            draw.text((gt_x1, gt_y1 - 25), 'GT', fill=(0, 255, 0), font=font)

        # 绘制预测bbox（红色）
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
        draw.rectangle([pred_x1, pred_y1, pred_x2, pred_y2], outline=(255, 0, 0), width=3)
        draw.text((pred_x1, pred_y1 - 50), 'Pred', fill=(255, 0, 0), font=font)

        # 添加IoU信息
        iou_text = f'IoU: {iou:.3f}'
        draw.text((10, 10), iou_text, fill=(255, 255, 255), font=font)

        # 保存图像
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)

        print(f'可视化结果已保存到: {save_path}')

    except Exception as e:
        print(f'可视化失败: {e}')


def extract_bbox_from_response(response, bbox_format='standard', original_image_size=None, processed_image_size=None):
    """Extract bounding box from model response by trying multiple formats in order"""

    predict_bbox = None

    # 方法1: 标准格式 <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    box_pattern = r'<\|box_start\|>\(([\d]+),([\d]+)\),\(([\d]+),([\d]+)\)<\|box_end\|>'
    matches = re.findall(box_pattern, response)
    if matches:
        predict_bbox = []
        for match in matches:
            x1, y1, x2, y2 = int(match[0]), int(match[1]), int(match[2]), int(match[3])
            predict_bbox.append([float(x1), float(y1), float(x2), float(y2)])

    # 方法2: JSON格式，适配多个结果 [{"bbox_2d": [...]}, ...] 或 {"bbox_2d": [...]}
    if predict_bbox is None:
        try:
            # 只适配如下格式：response为一个字符串，内容为一个或多个字典，每个字典有"bbox_2d"字段
            # 例如: '{"bbox_2d": [x1, y1, x2, y2]}, {"bbox_2d": [x1, y1, x2, y2]}'
            # 用正则提取所有字典
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

    # 如果提供了图像尺寸信息，进行坐标转换
    if predict_bbox is not None and original_image_size is not None and processed_image_size is not None:
        orig_width, orig_height = original_image_size
        proc_width, proc_height = processed_image_size

        # 计算缩放比例
        scale_x = orig_width / proc_width
        scale_y = orig_height / proc_height

        print(f'坐标转换: 原始图像尺寸 {original_image_size}, 处理后图像尺寸 {processed_image_size}')
        print(f'缩放比例: scale_x={scale_x:.3f}, scale_y={scale_y:.3f}')

        # 转换坐标
        converted_bbox = []
        for bbox in predict_bbox:
            x1, y1, x2, y2 = bbox
            # 将处理后的坐标转换回原始图像坐标
            orig_x1 = int(x1 * scale_x)
            orig_y1 = int(y1 * scale_y)
            orig_x2 = int(x2 * scale_x)
            orig_y2 = int(y2 * scale_y)

            # 确保坐标在有效范围内
            orig_x1 = max(0, min(orig_x1, orig_width))
            orig_y1 = max(0, min(orig_y1, orig_height))
            orig_x2 = max(orig_x1, min(orig_x2, orig_width))
            orig_y2 = max(orig_y1, min(orig_y2, orig_height))

            converted_bbox.append([orig_x1, orig_y1, orig_x2, orig_y2])
            print(f'坐标转换: [{x1}, {y1}, {x2}, {y2}] -> [{orig_x1}, {orig_y1}, {orig_x2}, {orig_y2}]')

        predict_bbox = converted_bbox

    # 如果没有检测到 bbox，返回默认值
    if predict_bbox is None or len(predict_bbox) == 0:
        return [0., 0., 0., 0.]

    # 返回第一个检测到的 bbox
    return predict_bbox[0]


class GeneralDetectionDataset(torch.utils.data.Dataset):

    def __init__(self, test_file, prompt_template):
        self.datas = open(test_file).readlines()
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        image_path = data['image']
        text = data['sentence']
        bbox = data['bbox']
        w, h = data['width'], data['height']

        # print(f"==== data: {data}")

        return {
            'text': self.prompt_template.format(text),
            'image_path': image_path,
            'bbox': bbox,
            'hw': (h, w),
            'original_size': (w, h),  # 添加原始图像尺寸信息
            'sentence': text,  # 添加 sentence 信息
        }


def get_processed_image_size(original_size):
    """根据原始图像尺寸计算处理后的图像尺寸"""
    from qwen_vl_utils.vision_process import smart_resize, MIN_PIXELS, MAX_PIXELS, IMAGE_FACTOR

    # 使用环境变量中的 MAX_PIXELS 和 MIN_PIXELS
    env_max_pixels = int(os.environ.get('MAX_PIXELS', MAX_PIXELS))
    env_min_pixels = int(os.environ.get('MIN_PIXELS', MIN_PIXELS))

    orig_width, orig_height = original_size
    processed_height, processed_width = smart_resize(
        orig_height, orig_width, factor=IMAGE_FACTOR, min_pixels=env_min_pixels, max_pixels=env_max_pixels)

    return (processed_width, processed_height)


def evaluate_grounding_model(args):
    """Main evaluation function for Qwen models on grounding tasks"""
    print(f'Evaluating model: {args.model_path}')
    print(f'Using prompt template: {args.prompt_template}')
    print(f'Using bbox format: {args.bbox_format}')
    print(f'Visualization interval: {args.viz_interval}')

    # Set random seed
    random.seed(args.seed)

    # Track results across datasets
    summaries = []

    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)

    # Create visualization directory
    viz_dir = os.path.join(args.out_dir, 'visualization_results')
    os.makedirs(viz_dir, exist_ok=True)

    # Load Qwen model
    print('Loading Qwen model...')
    engine = PtEngine(args.model_path, max_batch_size=args.batch_size)
    request_config = RequestConfig(max_tokens=512, temperature=0)

    # Evaluate on each dataset
    for ds_name in args.datasets:
        print(f'Evaluating on dataset: {ds_name}')

        # Load dataset
        dataset = GeneralDetectionDataset(test_file=ds_collections[ds_name], prompt_template=args.prompt_template)

        # Process all samples in the dataset
        outputs = []
        for idx in tqdm(range(len(dataset)), desc=f'Processing {ds_name}'):
            sample = dataset[idx]

            # 获取原始图像尺寸和处理后图像尺寸
            original_size = sample['original_size']
            processed_size = get_processed_image_size(original_size)

            # print(f"Sample {idx}: 原始尺寸 {original_size}, 处理后尺寸 {processed_size}")

            # Prepare inference request
            infer_request = InferRequest(
                messages=[{
                    'role': 'user',
                    'content': sample['text']
                }], images=[sample['image_path']])

            # Run inference
            resp_list = engine.infer([infer_request], request_config)
            response = resp_list[0].choices[0].message.content
            print(f'===response: {response}')

            # 提取 bbox 并进行坐标转换
            predict_bbox = extract_bbox_from_response(
                response, args.bbox_format, original_image_size=original_size, processed_image_size=processed_size)

            # 计算IoU
            target_bbox = torch.tensor(sample['bbox'], dtype=torch.float32).view(-1, 4)
            pred_bbox_tensor = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)
            iou, _ = box_iou(pred_bbox_tensor, target_bbox)
            iou = iou.item()

            # 每隔N个样本进行可视化
            if idx % args.viz_interval == 0:
                # 获取原始图像文件名
                original_image_name = os.path.basename(sample['image_path'])
                viz_filename = f'viz_{idx:06d}_{original_image_name}'
                viz_path = os.path.join(viz_dir, viz_filename)

                # 进行可视化
                visualize_detection_results(
                    sample['image_path'],
                    sample['bbox'],
                    predict_bbox,
                    iou,
                    viz_path,
                    sample['sentence']  # 添加 sentence 参数
                )

            # Store results
            outputs.append({
                'answer': response,
                'gt_bbox': sample['bbox'],
                'hw': sample['hw'],
                'predict_bbox': predict_bbox,  # 添加预测的 bbox
                'original_size': original_size,
                'processed_size': processed_size,
                'iou': iou,  # 添加IoU信息
            })

            # Optional: print some examples
            if idx < 3 or idx % 100 == 0:
                print(f'\nSample {idx}:')
                print(f"Question: {sample['text']}")
                print(f'Answer: {response}')
                print(f"Ground truth bbox: {sample['bbox']}")
                print(f'Predicted bbox: {predict_bbox}')
                print(f'IoU: {iou:.3f}')

        # Save results
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{ds_name}_{time_prefix}.json'
        results_file = os.path.join(args.out_dir, results_file)
        json.dump(outputs, open(results_file, 'w'))

        # Evaluate results
        correct = 0
        total_cnt = 0

        for output in outputs:
            # 使用已经转换过的预测 bbox
            predict_bbox = output['predict_bbox']
            iou = output['iou']  # 使用已计算的IoU

            # Count correct predictions (IoU >= 0.5)
            total_cnt += 1
            if iou >= 0.5:
                correct += 1

        # Print and store results
        precision = correct / total_cnt
        print(f'Dataset: {ds_name}')
        print(f'Total samples: {total_cnt}')
        print(f'Correct predictions (IoU >= 0.5): {correct}')
        print(f'Precision @ 1: {precision:.4f}')

        # Store summary
        summaries.append([args.model_path, ds_name, f'Precision @ 1: {precision:.4f}'])

    # Write final summary
    model_name = os.path.basename(args.model_path)
    summary_file = os.path.join(args.out_dir, f'{model_name}_summary.txt')
    with open(summary_file, 'w') as writer:
        for summary in summaries:
            print(summary)
            writer.write(f'{summary[0]}, {summary[1]}, {summary[2]}\n')

    print(f'Results saved to {args.out_dir}')
    print(f'Visualization results saved to {viz_dir}')


def get_prompt_template(bbox_format):
    """根据边界框格式返回对应的提示模板"""
    if bbox_format == 'standard':
        return '<image>Please detect the {} in the given image using bounding box.'
    elif bbox_format == 'json':  # json
        return '<image>\nLocate the {}, output its bbox coordinates using JSON format.'
        # return  'Outline the position of {} and output all the coordinates in JSON format with bbox_2d key. do not output other contents'
    else:
        return '<image>Please find the bounding box coordinates for the area described by: <|object_ref_start|>{}<|object_ref_end|>.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Qwen models on grounding tasks')
    parser.add_argument(
        '--model-path',
        type=str,
        default=
        '/mnt/workspace/offline/caoxuyang5/code/ms-swift-370-main/output/samhook_ep3_OnlySegP_ep10_lr8e-5_noaugCT_bf16/v0-20250919-095431/checkpoint-6300'
    )
    parser.add_argument(
        '--datasets', type=str, default='med_sam2_eval')  # med_sam2_eval, ref_coco_testA, med_sam2_eval_modified
    parser.add_argument(
        '--bbox-format',
        type=str,
        choices=['standard', 'json'],
        default='json',
        help='Bounding box format: standard for <|box_start|>, json for JSON format with bbox_2d')
    parser.add_argument(
        '--prompt-template',
        type=str,
        default=None,
        help='Optional: custom prompt template. If not provided, a default one will be selected based on bbox format.')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument(
        '--out-dir',
        type=str,
        default='samhook_ep3_OnlySegP_ep10_lr8e-5_noaugCT_bf16/detection/ep10',
        help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--viz-interval', type=int, default=10, help='Visualization interval (every N samples)')
    args = parser.parse_args()

    # Convert datasets string to list
    args.datasets = args.datasets.split(',')

    # If no prompt template is provided, use the default one based on bbox_format
    if args.prompt_template is None:
        args.prompt_template = get_prompt_template(args.bbox_format)

    # Run evaluation
    evaluate_grounding_model(args)

#!/usr/bin/env python3
"""
多模态模型分割能力评测脚本
基于Sa2VA模型进行分割任务评测，计算mean Dice指标并按模态分组统计
"""
import argparse
import csv
import io
import os
import warnings

import cv2
import json
import numpy as np
import requests
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from swift.llm import InferRequest, PtEngine, RequestConfig

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['MAX_PIXELS'] = '1003520'
os.environ['MAX_PIXELS'] = '65535'
os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ['FPS_MAX_FRAMES'] = '12'

warnings.filterwarnings('ignore')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Sa2VA分割能力评测')
    parser.add_argument(
        '--model_path',
        type=str,
        default=
        '/mnt/workspace/offline/caoxuyang5/code/ms-swift-370-main/output/samhook_ep3_OnlySegP_ep10_lr8e-5_noaugCT_bf16/v0-20250919-095431/checkpoint-6300',
        help='模型路径')
    parser.add_argument(
        '--test_json',
        type=str,
        # default="/mnt/workspace/offline/caoxuyang5/code/ms-swift/data/evaluate_data/MeCoVQA_Grounding_test_merged_modified_v3-pet.json",
        default=
        '/mnt/workspace/offline/caoxuyang5/code/ms-swift/data/evaluate_data/MeCoVQA_Grounding_test_merged_modified_v3.json',
        # default="/mnt/workspace/offline/caoxuyang5/code/ms-swift/data/evaluate_data/MeCoVQA_Grounding_test_merged_modified_v2.json"
        # default="/mnt/workspace/offline/caoxuyang5/code/ms-swift/data/evaluate_data/MeCoVQA_Grounding_test_merged_modified_v3-ct.json",
        help='测试数据json文件路径')
    parser.add_argument(
        '--root_dir',
        type=str,
        default='/mnt/workspace/offline/shared_data/SA-Med2D-20M/SAMed2Dv1',
        help='图像和mask的根目录路径')
    parser.add_argument(
        '--output_dir', type=str, default='samhook_ep3_OnlySegP_ep10_lr8e-5_noaugCT_bf16/medsam/ep10', help='评测结果输出目录')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    return parser.parse_args()


def init_engine(model_path, max_batch_size=1):
    engine = PtEngine(model_path, max_batch_size=max_batch_size)
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


def load_test_data(test_json_path):
    """加载测试数据"""
    print(f'正在加载测试数据: {test_json_path}')

    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print(f'测试数据加载完成，共{len(test_data)}个样本')
    return test_data


def extract_modality_and_dataset(sample_id):
    """从样本ID中提取模态和数据集信息

    Args:
        sample_id: 样本ID，如 "ct_00--AMOS2022--amos_0001--x_0020"

    Returns:
        tuple: (modality, dataset) 模态和数据集信息
    """
    # 按下划线分割，取第0部分作为模态
    modality = sample_id.split('_')[0].upper()

    # 提取数据集信息（在第一个--之后的部分）
    if '--' in sample_id:
        dataset_part = sample_id.split('--')[1]
        dataset = dataset_part
    else:
        dataset = 'unknown'

    return modality, dataset


def parse_mask_path(gpt_value):
    """解析GPT回复中的mask路径

    Args:
        gpt_value: GPT的回复，包含<SEG><mask>路径</mask>格式

    Returns:
        mask_path: 清理后的mask路径
    """
    # 移除特殊token
    mask_path = gpt_value.replace('<SEG>', '').replace('<mask>', '').replace('</mask>', '').strip()
    return mask_path


def load_mask(mask_path):
    """加载mask图像并转换为二值掩码

    Args:
        mask_path: mask图像路径

    Returns:
        mask: 二值掩码数组，值为0或1
    """
    if not os.path.exists(mask_path):
        print(f'警告: mask文件不存在: {mask_path}')
        return None

    try:
        # 加载mask图像
        # print(f"==== mask_path: {mask_path}")
        mask_img = Image.open(mask_path).convert('L')  # 转为灰度图
        mask = np.array(mask_img)

        # 转换为二值掩码 (0 或 1)
        mask = (mask > 0).astype(np.uint8)

        return mask
    except Exception as e:
        print(f'错误: 无法加载mask {mask_path}: {e}')
        return None


def calculate_dice(pred_mask, gt_mask):
    """计算Dice系数

    Args:
        pred_mask: 预测mask (H, W)
        gt_mask: 真实mask (H, W)

    Returns:
        dice_score: Dice分数
    """
    if pred_mask is None or gt_mask is None:
        return 0.0

    # 确保尺寸一致
    if pred_mask.shape != gt_mask.shape:
        # 将pred_mask调整到gt_mask的尺寸
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 转换为二值
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    # 计算交集和并集
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)

    # 计算Dice系数
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    else:
        dice_score = 2.0 * intersection / union
        return dice_score


def postprocess_response(resp, infer_request):
    # 解析分割掩码
    raw_response = getattr(resp, 'raw_response', None)
    if raw_response is None and hasattr(resp, 'to_dict'):
        raw_response = resp.to_dict().get('raw_response', None)
    if raw_response is None and isinstance(resp, dict):
        raw_response = resp

    img = None
    masks = None

    # 读取原图
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
            # 没有mask，返回img
            masks = [np.zeros((img.height, img.width))]
            return img, masks
    else:
        # 没有mask，返回img
        masks = [np.zeros((img.height, img.width))]
        return img, masks


def masks_to_single_mask(prediction_masks):
    """将多个预测mask合并为单个mask

    Args:
        prediction_masks: 预测mask列表

    Returns:
        combined_mask: 合并后的单个mask
    """
    if not prediction_masks or len(prediction_masks) == 0:
        return None

    try:
        # 取第一个mask的第一个通道
        combined_mask = prediction_masks[0][0]  # (H, W)

        # 如果有多个mask，可以选择合并策略
        for mask_group in prediction_masks[1:]:
            combined_mask = np.logical_or(combined_mask, mask_group[0])

        return combined_mask.astype(np.uint8)

    except Exception as e:
        print(f'错误: mask合并失败: {e}')
        return None


def evaluate_segmentation(args):
    """主评测函数"""

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化模型
    engine = init_engine(args.model_path)
    request_config = RequestConfig(max_tokens=1024, temperature=0.2)

    # 加载测试数据
    test_data = load_test_data(args.test_json)

    # 初始化结果统计
    results = {'all': {'dice_scores': [], 'count': 0}, 'by_modality': {}}

    # 处理每个测试样本
    print('开始评测...')
    for sample in tqdm(test_data, desc='评测进度'):

        sample_id = sample['id']
        image_path = os.path.join(args.root_dir, sample['image'])
        # print(f"======== image_path: {image_path}")

        # 提取模态信息
        modality, dataset = extract_modality_and_dataset(sample_id)

        # 获取查询和真实mask路径
        conversation = sample['conversations']
        query = conversation[0]['value']  # human的输入
        gt_response = conversation[1]['value']  # gpt的输出

        # 解析真实mask路径
        gt_mask_path = parse_mask_path(gt_response)
        gt_mask_full_path = os.path.join(args.root_dir, gt_mask_path)
        # print(f"======== gt_mask_full_path: {gt_mask_full_path}")

        # 加载真实mask
        gt_mask = load_mask(gt_mask_full_path)
        if gt_mask is None:
            print(f'跳过样本 {sample_id}: 无法加载真实mask')
            continue

        # 2. 构建推理请求
        infer_requests = [
            InferRequest(messages=[{
                'role': 'user',
                'content': query
            }], images=[image_path]),
        ]

        # 模型推理
        resp_list = engine.infer(infer_requests, request_config)
        # print(f'response: {resp_list[0].choices[0].message.content}')

        # INSERT_YOUR_CODE
        import matplotlib.pyplot as plt
        import cv2

        # 4. 后处理与可视化
        img, pred_masks = postprocess_response(resp_list[0], infer_requests[0])
        # pred_mask = masks_to_single_mask(pred_masks)
        pred_mask = pred_masks[0].astype(np.uint8)
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        # print(f"pred_mask: {pred_mask.shape}, gt_mask: {gt_mask.shape}")
        # 计算Dice分数
        dice_score = calculate_dice(pred_mask, gt_mask)

        # 创建结果目录
        result_dir = os.path.join(args.output_dir, 'results')
        os.makedirs(result_dir, exist_ok=True)

        # 获取原始图像名（不带路径）
        img_name = os.path.basename(image_path)
        save_path = os.path.join(result_dir, img_name)

        # 可视化三列：原图、原图+预测mask、原图+真实mask
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

        # 确保mask尺寸与原始图像一致
        if pred_mask.shape != img.shape[:2]:
            pred_mask_resized = cv2.resize(pred_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            pred_mask_resized = pred_mask

        if gt_mask.shape != img.shape[:2]:
            gt_mask_resized = cv2.resize(gt_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            gt_mask_resized = gt_mask

        # 创建叠加图像：原图 + 预测mask（带透明度）
        pred_overlay = img_rgb.copy()

        # 红色mask叠加，透明度为0.6
        alpha = 0.6
        # 创建红色mask图像
        red_mask = np.zeros_like(img_rgb)
        red_mask[pred_mask_resized > 0] = [255, 0, 0]  # 红色

        # 透明度混合
        pred_overlay = cv2.addWeighted(pred_overlay, 1 - alpha, red_mask, alpha, 0)

        # 创建叠加图像：原图 + 真实mask（带透明度）
        gt_overlay = img_rgb.copy()

        # 绿色mask叠加，透明度为0.6
        green_mask = np.zeros_like(img_rgb)
        green_mask[gt_mask_resized > 0] = [0, 255, 0]  # 绿色

        # 透明度混合
        gt_overlay = cv2.addWeighted(gt_overlay, 1 - alpha, green_mask, alpha, 0)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # 第一列：原图
        axs[0].imshow(img_rgb)
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        # 第二列：原图 + 预测mask叠加（带透明度）
        axs[1].imshow(pred_overlay)
        axs[1].set_title('Image + Pred Mask (Red, α=0.6)')
        axs[1].axis('off')

        # 第三列：原图 + 真实mask叠加（带透明度）
        axs[2].imshow(gt_overlay)
        axs[2].set_title('Image + GT Mask (Green, α=0.6)')
        axs[2].axis('off')

        # 提取并清理query文本（移除<image>标记）
        clean_query = query.replace('<image>', '').replace('\n', ' ').strip()

        # 设置主标题，包含Dice分数和查询文本
        plt.suptitle(f'Dice Score: {dice_score:.4f}\nQuery: {clean_query}', fontsize=10, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.85])
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        # 记录结果
        results['all']['dice_scores'].append(dice_score)
        results['all']['count'] += 1

        # 按模态记录
        if modality not in results['by_modality']:
            results['by_modality'][modality] = {'dice_scores': [], 'count': 0}

        results['by_modality'][modality]['dice_scores'].append(dice_score)
        results['by_modality'][modality]['count'] += 1

        # 收集每个样本的详细信息用于CSV保存
        if 'sample_details' not in results:
            results['sample_details'] = []

        sample_detail = {
            'sample_id': sample_id,
            'modality': modality,
            'dataset': dataset,
            'image_path': image_path,
            'mask_path': gt_mask_full_path,
            'dice_score': dice_score,
            'query': query.replace('<image>', '').replace('\n', ' ').strip()
        }
        results['sample_details'].append(sample_detail)

        # 特殊处理CT模态的Totalsegmentator_dataset统计
        if modality == 'CT':
            # 初始化CT模态的特殊统计结构
            if 'ct_special_stats' not in results:
                results['ct_special_stats'] = {
                    'totalsegmentator_only': {
                        'dice_scores': [],
                        'count': 0
                    },
                    'ct_except_totalsegmentator': {
                        'dice_scores': [],
                        'count': 0
                    },
                    'ct_all': {
                        'dice_scores': [],
                        'count': 0
                    }
                }

            # 记录到CT所有数据的统计
            results['ct_special_stats']['ct_all']['dice_scores'].append(dice_score)
            results['ct_special_stats']['ct_all']['count'] += 1

            # 根据数据集分别统计
            if dataset == 'Totalsegmentator_dataset':
                results['ct_special_stats']['totalsegmentator_only']['dice_scores'].append(dice_score)
                results['ct_special_stats']['totalsegmentator_only']['count'] += 1
            else:
                results['ct_special_stats']['ct_except_totalsegmentator']['dice_scores'].append(dice_score)
                results['ct_special_stats']['ct_except_totalsegmentator']['count'] += 1

        # 打印当前样本结果
        print(f'样本 {sample_id} ({modality}): Dice = {dice_score:.4f}')

    # 计算平均结果
    print('\n' + '=' * 60)
    print('评测结果汇总:')
    print('=' * 60)

    # 总体结果
    overall_mean_dice = np.mean(results['all']['dice_scores'])
    overall_std_dice = np.std(results['all']['dice_scores'])
    print(f"总体 Mean Dice: {overall_mean_dice:.4f} ± {overall_std_dice:.4f} (样本数: {results['all']['count']})")

    # 按模态统计
    print('\n按模态统计:')
    print('-' * 40)
    modality_results = {}
    for modality, data in results['by_modality'].items():
        mean_dice = np.mean(data['dice_scores'])
        std_dice = np.std(data['dice_scores'])
        modality_results[modality] = {'mean_dice': mean_dice, 'std_dice': std_dice, 'count': data['count']}
        print(f"{modality}: Mean Dice = {mean_dice:.4f} ± {std_dice:.4f} (样本数: {data['count']})")

    # CT模态特殊统计
    if 'ct_special_stats' in results:
        print('\nCT模态详细统计:')
        print('-' * 40)

        # 1. Totalsegmentator_dataset单独统计
        if results['ct_special_stats']['totalsegmentator_only']['count'] > 0:
            totalseg_mean = np.mean(results['ct_special_stats']['totalsegmentator_only']['dice_scores'])
            totalseg_std = np.std(results['ct_special_stats']['totalsegmentator_only']['dice_scores'])
            print(
                f"1. Totalsegmentator_dataset: Mean Dice = {totalseg_mean:.4f} ± {totalseg_std:.4f} (样本数: {results['ct_special_stats']['totalsegmentator_only']['count']})"
            )

        # 2. 除去Totalsegmentator_dataset的其余CT模态统计
        if results['ct_special_stats']['ct_except_totalsegmentator']['count'] > 0:
            ct_except_mean = np.mean(results['ct_special_stats']['ct_except_totalsegmentator']['dice_scores'])
            ct_except_std = np.std(results['ct_special_stats']['ct_except_totalsegmentator']['dice_scores'])
            print(
                f"2. CT模态(除Totalsegmentator): Mean Dice = {ct_except_mean:.4f} ± {ct_except_std:.4f} (样本数: {results['ct_special_stats']['ct_except_totalsegmentator']['count']})"
            )

        # 3. 包括Totalsegmentator_dataset的所有CT模态统计
        if results['ct_special_stats']['ct_all']['count'] > 0:
            ct_all_mean = np.mean(results['ct_special_stats']['ct_all']['dice_scores'])
            ct_all_std = np.std(results['ct_special_stats']['ct_all']['dice_scores'])
            print(
                f"3. CT模态(全部): Mean Dice = {ct_all_mean:.4f} ± {ct_all_std:.4f} (样本数: {results['ct_special_stats']['ct_all']['count']})"
            )

    # 保存详细结果
    detailed_results = {
        'overall': {
            'mean_dice': float(overall_mean_dice),
            'std_dice': float(overall_std_dice),
            'count': results['all']['count'],
            'all_scores': [float(x) for x in results['all']['dice_scores']]
        },
        'by_modality': {}
    }

    for modality, data in modality_results.items():
        detailed_results['by_modality'][modality] = {
            'mean_dice': float(data['mean_dice']),
            'std_dice': float(data['std_dice']),
            'count': data['count'],
            'all_scores': [float(x) for x in results['by_modality'][modality]['dice_scores']]
        }

    # 添加CT模态特殊统计到详细结果
    if 'ct_special_stats' in results:
        detailed_results['ct_special_stats'] = {}

        # 1. Totalsegmentator_dataset单独统计
        if results['ct_special_stats']['totalsegmentator_only']['count'] > 0:
            totalseg_mean = np.mean(results['ct_special_stats']['totalsegmentator_only']['dice_scores'])
            totalseg_std = np.std(results['ct_special_stats']['totalsegmentator_only']['dice_scores'])
            detailed_results['ct_special_stats']['totalsegmentator_only'] = {
                'mean_dice': float(totalseg_mean),
                'std_dice': float(totalseg_std),
                'count': results['ct_special_stats']['totalsegmentator_only']['count'],
                'all_scores': [float(x) for x in results['ct_special_stats']['totalsegmentator_only']['dice_scores']]
            }

        # 2. 除去Totalsegmentator_dataset的其余CT模态统计
        if results['ct_special_stats']['ct_except_totalsegmentator']['count'] > 0:
            ct_except_mean = np.mean(results['ct_special_stats']['ct_except_totalsegmentator']['dice_scores'])
            ct_except_std = np.std(results['ct_special_stats']['ct_except_totalsegmentator']['dice_scores'])
            detailed_results['ct_special_stats']['ct_except_totalsegmentator'] = {
                'mean_dice': float(ct_except_mean),
                'std_dice': float(ct_except_std),
                'count': results['ct_special_stats']['ct_except_totalsegmentator']['count'],
                'all_scores':
                [float(x) for x in results['ct_special_stats']['ct_except_totalsegmentator']['dice_scores']]
            }

        # 3. 包括Totalsegmentator_dataset的所有CT模态统计
        if results['ct_special_stats']['ct_all']['count'] > 0:
            ct_all_mean = np.mean(results['ct_special_stats']['ct_all']['dice_scores'])
            ct_all_std = np.std(results['ct_special_stats']['ct_all']['dice_scores'])
            detailed_results['ct_special_stats']['ct_all'] = {
                'mean_dice': float(ct_all_mean),
                'std_dice': float(ct_all_std),
                'count': results['ct_special_stats']['ct_all']['count'],
                'all_scores': [float(x) for x in results['ct_special_stats']['ct_all']['dice_scores']]
            }

    # 保存结果到文件
    results_file = os.path.join(args.output_dir, 'segmentation_evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    print(f'\n详细结果已保存到: {results_file}')

    # 保存CSV格式的详细结果
    if 'sample_details' in results:
        csv_file = os.path.join(args.output_dir, 'segmentation_evaluation_details.csv')

        # 按模态和数据集排序
        sorted_samples = sorted(results['sample_details'], key=lambda x: (x['modality'], x['dataset']))

        # 写入CSV文件
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['sample_id', 'modality', 'dataset', 'image_path', 'mask_path', 'dice_score', 'query']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # 写入表头
            writer.writeheader()

            # 写入数据
            for sample in sorted_samples:
                writer.writerow(sample)

        print(f'CSV详细结果已保存到: {csv_file}')
        print(f'共保存 {len(sorted_samples)} 个样本的详细信息')

    print('=' * 60)


if __name__ == '__main__':
    args = parse_args()
    evaluate_segmentation(args)

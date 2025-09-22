#!/usr/bin/env python3
"""
多模态模型分割能力评测脚本 - 新数据格式专用版本
基于Sa2VA模型进行分割任务评测，计算mean Dice指标并按数据集分组统计
适用于新数据格式，包含images、masks字段和metadata信息
"""
import argparse
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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['MAX_PIXELS'] = '65535'
os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ['FPS_MAX_FRAMES'] = '12'

warnings.filterwarnings('ignore')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Sa2VA分割能力评测 - 新数据格式专用')
    parser.add_argument(
        '--model_path',
        type=str,
        default=
        '/mnt/workspace/offline/caoxuyang5/code/ms-swift-370-main/output/samhook_ep3_OnlySegP_ep10_lr8e-5_noaugCT_bf16/v0-20250919-095431/checkpoint-6300',
        help='模型路径')
    parser.add_argument(
        '--test_json',
        type=str,
        default='/mnt/workspace/offline/caoxuyang5/code/ms-swift/data/MedSegBench/medseg_bench_test_2571_ENG.json',
        help='测试数据json文件路径')
    parser.add_argument(
        '--root_dir', type=str, default='/mnt/workspace/offline/shared_data/MedSegBench', help='图像和mask的根目录路径')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='samhook_ep3_OnlySegP_ep10_lr8e-5_noaugCT_bf16/medsegbench/ep10-eng',
        help='评测结果输出目录')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    parser.add_argument('--max_samples', type=int, default=None, help='最大评测样本数，None表示评测全部样本')
    return parser.parse_args()


def init_engine(model_path, max_batch_size=1):
    """初始化推理引擎"""
    engine = PtEngine(model_path, max_batch_size=max_batch_size)
    return engine


def rle_to_mask(rle):
    """将RLE格式转换为mask

    Args:
        rle: COCO RLE dict, counts为str

    Returns:
        mask: numpy array, shape (H, W), dtype uint8
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


def validate_data_format(sample):
    """验证数据格式是否为新格式

    Args:
        sample: 数据样本

    Returns:
        bool: 是否为新格式
    """
    required_fields = ['id', 'images', 'masks', 'conversations', 'metadata']
    for field in required_fields:
        if field not in sample:
            return False

    # 检查metadata中是否包含必要字段
    metadata = sample['metadata']
    required_metadata = ['dataset', 'region']
    for field in required_metadata:
        if field not in metadata:
            return False

    return True


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
    """后处理模型响应，提取分割mask"""
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
            # 没有mask，返回空mask
            masks = [np.zeros((img.height, img.width), dtype=np.uint8)]
            return img, masks
    else:
        # 没有mask，返回空mask
        masks = [np.zeros((img.height, img.width), dtype=np.uint8)]
        return img, masks


def save_visualization(image_path, pred_mask, gt_mask, dice_score, sample_info, output_dir):
    """保存可视化结果"""
    import matplotlib.pyplot as plt

    # 创建结果目录
    result_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(result_dir, exist_ok=True)

    # 构建文件名
    dataset = sample_info['dataset']
    region = sample_info['region'].replace(' ', '_')
    sample_id = sample_info['id'].replace('/', '_').replace('\\', '_')
    img_name = f'{dataset}_{region}_{sample_id}_dice_{dice_score:.4f}.png'
    save_path = os.path.join(result_dir, img_name)

    # 读取原始图像
    img = cv2.imread(image_path)
    if img is None:
        print(f'警告: 无法读取图像 {image_path}')
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 预测mask
    axes[1].imshow(pred_mask, cmap='Reds', alpha=0.8)
    axes[1].set_title('Predicted Mask', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # 真实mask
    axes[2].imshow(gt_mask, cmap='Greens', alpha=0.8)
    axes[2].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # 添加总标题
    plt.suptitle(f'Dataset: {dataset} | Region: {region} | Dice: {dice_score:.4f}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def evaluate_segmentation(args):
    """主评测函数"""

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化模型
    print('初始化模型...')
    engine = init_engine(args.model_path)
    request_config = RequestConfig(max_tokens=1024, temperature=0.2)

    # 加载测试数据
    test_data = load_test_data(args.test_json)

    # 验证数据格式
    if test_data:
        if not validate_data_format(test_data[0]):
            print('错误: 数据格式不符合新格式要求')
            return
        print('数据格式验证通过')
    else:
        print('错误: 测试数据为空')
        return

    # 限制样本数量
    if args.max_samples and args.max_samples < len(test_data):
        test_data = test_data[:args.max_samples]
        print(f'限制评测样本数量为: {args.max_samples}')

    # 初始化结果统计
    results = {'all': {'dice_scores': [], 'count': 0}, 'by_dataset': {}, 'by_region': {}}

    # 处理每个测试样本
    print('开始评测...')
    failed_samples = []

    for sample in tqdm(test_data, desc='评测进度'):

        sample_id = sample['id']

        # 获取路径和信息
        image_rel_path = sample['images'][0]
        gt_mask_rel_path = sample['masks'][0]
        query = sample['conversations'][0]['value']  # human的输入

        # 获取元数据
        metadata = sample['metadata']
        dataset = metadata['dataset']
        region = metadata['region']

        # 构建完整路径
        image_path = os.path.join(args.root_dir, image_rel_path)
        gt_mask_full_path = os.path.join(args.root_dir, gt_mask_rel_path)

        print(f'\n处理样本: {sample_id}')
        print(f'数据集: {dataset}, 区域: {region}')
        print(f'图像路径: {image_path}')
        print(f'掩模路径: {gt_mask_full_path}')

        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f'错误: 图像文件不存在: {image_path}')
            failed_samples.append({'id': sample_id, 'reason': 'image_not_found'})
            continue

        # 加载真实mask
        gt_mask = load_mask(gt_mask_full_path)
        if gt_mask is None:
            print(f'错误: 无法加载真实mask: {gt_mask_full_path}')
            failed_samples.append({'id': sample_id, 'reason': 'mask_load_failed'})
            continue

        try:
            # 构建推理请求
            infer_requests = [InferRequest(messages=[{'role': 'user', 'content': query}], images=[image_path])]

            # 模型推理
            resp_list = engine.infer(infer_requests, request_config)
            response_text = resp_list[0].choices[0].message.content
            print(f'模型回复: {response_text}')

            # 后处理得到预测mask
            img, pred_masks = postprocess_response(resp_list[0], infer_requests[0])
            pred_mask = pred_masks[0].astype(np.uint8)

            # 调整尺寸一致性
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            print(f'预测mask形状: {pred_mask.shape}, 真实mask形状: {gt_mask.shape}')

            # 计算Dice分数
            dice_score = calculate_dice(pred_mask, gt_mask)
            print(f'Dice分数: {dice_score:.4f}')

            # 保存可视化结果
            sample_info = {'id': sample_id, 'dataset': dataset, 'region': region}

            save_path = save_visualization(image_path, pred_mask, gt_mask, dice_score, sample_info, args.output_dir)
            if save_path:
                print(f'可视化结果已保存: {save_path}')

            # 记录结果
            results['all']['dice_scores'].append(dice_score)
            results['all']['count'] += 1

            # 按数据集记录
            if dataset not in results['by_dataset']:
                results['by_dataset'][dataset] = {'dice_scores': [], 'count': 0}
            results['by_dataset'][dataset]['dice_scores'].append(dice_score)
            results['by_dataset'][dataset]['count'] += 1

            # 按区域记录
            if region not in results['by_region']:
                results['by_region'][region] = {'dice_scores': [], 'count': 0}
            results['by_region'][region]['dice_scores'].append(dice_score)
            results['by_region'][region]['count'] += 1

        except Exception as e:
            print(f'错误: 处理样本 {sample_id} 时发生异常: {e}')
            failed_samples.append({'id': sample_id, 'reason': f'processing_error: {str(e)}'})
            continue

    # 计算和打印结果
    print('\n' + '=' * 80)
    print('评测结果汇总')
    print('=' * 80)

    if results['all']['count'] == 0:
        print('警告: 没有成功处理的样本')
        return

    # 总体结果
    overall_mean_dice = np.mean(results['all']['dice_scores'])
    overall_std_dice = np.std(results['all']['dice_scores'])
    print(f'\n【总体结果】')
    print(f'Mean Dice: {overall_mean_dice:.4f} ± {overall_std_dice:.4f}')
    print(f"成功样本数: {results['all']['count']}")
    print(f'失败样本数: {len(failed_samples)}')

    # 按数据集统计
    print(f'\n【按数据集统计】')
    print('-' * 60)
    dataset_results = {}
    for dataset, data in results['by_dataset'].items():
        mean_dice = np.mean(data['dice_scores'])
        std_dice = np.std(data['dice_scores'])
        dataset_results[dataset] = {'mean_dice': mean_dice, 'std_dice': std_dice, 'count': data['count']}
        print(f"{dataset:20} | Mean Dice: {mean_dice:.4f} ± {std_dice:.4f} | 样本数: {data['count']:4d}")

    # 按区域统计
    print(f'\n【按区域统计】')
    print('-' * 60)
    region_results = {}
    for region, data in results['by_region'].items():
        mean_dice = np.mean(data['dice_scores'])
        std_dice = np.std(data['dice_scores'])
        region_results[region] = {'mean_dice': mean_dice, 'std_dice': std_dice, 'count': data['count']}
        print(f"{region:20} | Mean Dice: {mean_dice:.4f} ± {std_dice:.4f} | 样本数: {data['count']:4d}")

    # 保存详细结果
    detailed_results = {
        'summary': {
            'total_samples_attempted': len(test_data),
            'successful_samples': results['all']['count'],
            'failed_samples': len(failed_samples),
            'overall_mean_dice': float(overall_mean_dice),
            'overall_std_dice': float(overall_std_dice)
        },
        'overall': {
            'mean_dice': float(overall_mean_dice),
            'std_dice': float(overall_std_dice),
            'count': results['all']['count'],
            'all_scores': [float(x) for x in results['all']['dice_scores']]
        },
        'by_dataset': {},
        'by_region': {},
        'failed_samples': failed_samples
    }

    # 保存数据集结果
    for dataset, data in dataset_results.items():
        detailed_results['by_dataset'][dataset] = {
            'mean_dice': float(data['mean_dice']),
            'std_dice': float(data['std_dice']),
            'count': data['count'],
            'all_scores': [float(x) for x in results['by_dataset'][dataset]['dice_scores']]
        }

    # 保存区域结果
    for region, data in region_results.items():
        detailed_results['by_region'][region] = {
            'mean_dice': float(data['mean_dice']),
            'std_dice': float(data['std_dice']),
            'count': data['count'],
            'all_scores': [float(x) for x in results['by_region'][region]['dice_scores']]
        }

    # 保存结果到文件
    results_file = os.path.join(args.output_dir, 'segmentation_evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    print(f'\n详细结果已保存到: {results_file}')

    # 如果有失败样本，保存失败列表
    if failed_samples:
        failed_file = os.path.join(args.output_dir, 'failed_samples.json')
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_samples, f, indent=2, ensure_ascii=False)
        print(f'失败样本列表已保存到: {failed_file}')

    print('=' * 80)


if __name__ == '__main__':
    args = parse_args()
    evaluate_segmentation(args)

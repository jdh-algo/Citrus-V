import argparse
import os
import re

import json
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops.boxes import box_area
from tqdm import tqdm


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


def extract_all_bboxes_from_response(response):
    """Extract all bounding boxes from model response"""
    all_bboxes = []

    try:
        # 方法1: 提取标准JSON格式中的bbox_2d [x1, y1, x2, y2]
        json_pattern = r'\"bbox_2d\":\s*\[([\d\s,\.]+)\]'
        json_matches = re.findall(json_pattern, response)

        for match in json_matches:
            coords_str = match
            coords = [float(x.strip()) for x in coords_str.split(',')]
            if len(coords) == 4:
                all_bboxes.append(coords)

        # 方法2: 提取混合格式 {"bbox_2d": [x1, y1),(x2, y2)}
        mixed_pattern = r'\"bbox_2d\":\s*\[(\d+),\s*(\d+)\),\((\d+),\s*(\d+)\)'
        mixed_matches = re.findall(mixed_pattern, response)

        for match in mixed_matches:
            x1, y1, x2, y2 = int(match[0]), int(match[1]), int(match[2]), int(match[3])
            all_bboxes.append([float(x1), float(y1), float(x2), float(y2)])

        # 如果通过JSON格式找到了bbox，直接返回
        if all_bboxes:
            return all_bboxes

        # 方法3: 标准格式 <|box_start|>(x1,y1),(x2,y2)<|box_end|>
        box_pattern = r'<\|box_start\|>\(([\d]+),([\d]+)\),\(([\d]+),([\d]+)\)<\|box_end\|>'
        box_matches = re.findall(box_pattern, response)

        for match in box_matches:
            x1, y1, x2, y2 = int(match[0]), int(match[1]), int(match[2]), int(match[3])
            all_bboxes.append([float(x1), float(y1), float(x2), float(y2)])

        if all_bboxes:
            return all_bboxes

        # 方法4: 不完整的标准格式，如 [148, 156),(190, 203)<|box_end|
        incomplete_pattern = r'\[(\d+),\s*(\d+)\),\((\d+),\s*(\d+)\)<\|box_end\|'
        incomplete_matches = re.findall(incomplete_pattern, response)

        for match in incomplete_matches:
            x1, y1, x2, y2 = int(match[0]), int(match[1]), int(match[2]), int(match[3])
            all_bboxes.append([float(x1), float(y1), float(x2), float(y2)])

        if all_bboxes:
            return all_bboxes

        # 方法5: 寻找纯数组格式 [x1, y1, x2, y2]
        array_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        array_matches = re.findall(array_pattern, response)

        for match in array_matches:
            x1, y1, x2, y2 = int(match[0]), int(match[1]), int(match[2]), int(match[3])
            all_bboxes.append([float(x1), float(y1), float(x2), float(y2)])

        if all_bboxes:
            return all_bboxes

        # 方法6: 寻找任何四个连续的数字作为坐标
        numbers_pattern = r'(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)'
        numbers_match = re.search(numbers_pattern, response)
        if numbers_match:
            x1, y1, x2, y2 = int(numbers_match.group(1)), int(numbers_match.group(2)), int(numbers_match.group(3)), int(
                numbers_match.group(4))
            all_bboxes.append([float(x1), float(y1), float(x2), float(y2)])

    except Exception as e:
        print(f'Error parsing bbox from response: {e}')
        print(f'Response: {response}')

    return all_bboxes


def extract_bbox_from_response(response, selection_strategy='first'):
    """Extract single bounding box from model response based on selection strategy

    Args:
        response: Model response string
        selection_strategy: Strategy for multiple bboxes - "first", "largest", "smallest"
    """
    all_bboxes = extract_all_bboxes_from_response(response)
    return select_bbox(all_bboxes, selection_strategy)


def select_bbox(bboxes, strategy='first'):
    """Select one bbox from multiple bboxes based on strategy"""
    if not bboxes:
        return [0., 0., 0., 0.]

    if strategy == 'first':
        return bboxes[0]
    elif strategy == 'largest':
        # 选择面积最大的bbox
        areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
        max_idx = areas.index(max(areas))
        return bboxes[max_idx]
    elif strategy == 'smallest':
        # 选择面积最小的bbox
        areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
        min_idx = areas.index(min(areas))
        return bboxes[min_idx]
    else:
        return bboxes[0]


def visualize_sample(index, results, dataset_jsonl_path, output_dir='visualizations', show_all_preds=True):
    """可视化特定索引的预测结果和真实标签

    Args:
        index: 样本索引
        results: 结果列表
        dataset_jsonl_path: 数据集JSONL文件路径
        output_dir: 输出目录
        show_all_preds: 是否显示所有预测框
    """
    # 加载数据集
    with open(dataset_jsonl_path, 'r') as f:
        dataset = [json.loads(line) for line in f]

    if index < 0 or index >= len(results):
        print(f'Error: Index {index} out of range (0-{len(results)-1})')
        return False

    # 获取结果和数据
    result = results[index]
    data = dataset[index]

    # 获取图像路径、边界框和类别名称
    image_path = data['image']
    gt_bbox = data['bbox']
    category = data['sentence']

    # 提取所有预测边界框
    all_pred_bboxes = extract_all_bboxes_from_response(result['answer'])

    # 计算每个预测框与真实框的IoU
    ious = []
    for pred_bbox in all_pred_bboxes:
        gt_tensor = torch.tensor(gt_bbox, dtype=torch.float32).view(-1, 4)
        pred_tensor = torch.tensor(pred_bbox, dtype=torch.float32).view(-1, 4)
        iou, _ = box_iou(pred_tensor, gt_tensor)
        ious.append(iou.item())

    # 找到最佳IoU
    best_iou = max(ious) if ious else 0.0

    print(f'Sample {index}:')
    print(f'Image: {image_path}')
    print(f'Category: {category}')
    print(f'GT Bbox: {gt_bbox}')
    print(f'Predicted bboxes: {len(all_pred_bboxes)}')
    for i, (pred_bbox, iou) in enumerate(zip(all_pred_bboxes, ious)):
        print(f'  Pred {i+1}: {pred_bbox}, IoU: {iou:.4f}')
    print(f'Best IoU: {best_iou:.4f}')

    # 加载图像
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f'Error loading image {image_path}: {e}')
        return False

    # 创建绘图对象
    draw = ImageDraw.Draw(image)

    # 尝试加载字体
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 15)
    except IOError:
        font = ImageFont.load_default()

    # 绘制真实边界框（绿色）
    draw.rectangle([(gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3])], outline='green', width=3)

    # 为真实边界框添加标签
    gt_label = f'GT: {category}'
    _, _, text_w, text_h = draw.textbbox((0, 0), text=gt_label, font=font)
    draw.rectangle([(gt_bbox[0], gt_bbox[1] - text_h - 2), (gt_bbox[0] + text_w + 2, gt_bbox[1])], fill='green')
    draw.text((gt_bbox[0] + 1, gt_bbox[1] - text_h - 1), gt_label, fill='white', font=font)

    # 定义预测框的颜色
    pred_colors = ['red', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']

    # 绘制预测边界框
    for i, (pred_bbox, iou) in enumerate(zip(all_pred_bboxes, ious)):
        color = pred_colors[i % len(pred_colors)]

        # 绘制预测框
        draw.rectangle([(pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3])], outline=color, width=2)

        # 添加预测框标签
        pred_label = f'Pred{i+1}: IoU={iou:.2f}'
        _, _, text_w, text_h = draw.textbbox((0, 0), text=pred_label, font=font)

        # 标签位置：框的右下角
        label_x = pred_bbox[2] - text_w - 2
        label_y = pred_bbox[3] + 2

        # 确保标签不超出图像边界
        label_x = max(0, min(label_x, image.width - text_w - 2))
        label_y = max(0, min(label_y, image.height - text_h - 4))

        draw.rectangle([(label_x, label_y), (label_x + text_w + 2, label_y + text_h + 2)], fill=color)
        draw.text((label_x + 1, label_y + 1), pred_label, fill='white', font=font)

    # 在图像顶部添加总体信息
    info_label = f'Best IoU: {best_iou:.3f} | Preds: {len(all_pred_bboxes)}'
    _, _, text_w, text_h = draw.textbbox((0, 0), text=info_label, font=font)
    pos_x = (image.width - text_w) // 2
    draw.rectangle([(pos_x - 5, 10), (pos_x + text_w + 5, 10 + text_h + 5)], fill='black')
    draw.text((pos_x, 12), info_label, fill='white', font=font)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存结果
    output_path = os.path.join(output_dir, f'sample_{index}_precision.png')
    image.save(output_path)

    print(f'Visualization saved to {output_path}')

    return True


def calculate_precision(json_file_path,
                        iou_threshold=0.5,
                        verbose=False,
                        selection_strategy='first',
                        visualize=False,
                        dataset_jsonl_path=None,
                        output_dir='visualizations',
                        visualize_samples=None):
    """Calculate precision for grounding task results"""

    print(f'Loading results from: {json_file_path}')
    print(f'Selection strategy for multiple bboxes: {selection_strategy}')

    # 加载JSON文件
    with open(json_file_path, 'r') as f:
        results = json.load(f)

    print(f'Total samples: {len(results)}')

    # 统计变量
    correct = 0
    total_cnt = 0
    total_iou = 0.0
    failed_extractions = 0
    multiple_bbox_count = 0

    # 存储每个样本的IoU用于分析
    ious = []

    # 可视化设置
    if visualize and dataset_jsonl_path is None:
        print('Warning: Visualization enabled but no dataset JSONL path provided. Skipping visualization.')
        visualize = False

    if visualize:
        os.makedirs(output_dir, exist_ok=True)
        print(f'Visualization output directory: {output_dir}')

        # 确定要可视化的样本
        if visualize_samples is None:
            # 默认可视化前10个样本
            visualize_samples = list(range(min(10, len(results))))
        elif isinstance(visualize_samples, str):
            # 解析范围字符串，如 "0-5" 或 "0,2,4"
            if '-' in visualize_samples:
                start, end = map(int, visualize_samples.split('-'))
                visualize_samples = list(range(start, min(end + 1, len(results))))
            else:
                visualize_samples = [int(x) for x in visualize_samples.split(',') if x.strip()]

    # 处理每个样本
    for idx, sample in enumerate(tqdm(results, desc='Calculating IoU')):
        # 提取预测bbox
        predict_bbox = extract_bbox_from_response(sample['answer'], selection_strategy)
        gt_bbox = sample['gt_bbox']

        # 检查是否有多个bbox（通过检查answer中bbox_2d出现的次数）
        bbox_count = sample['answer'].count('bbox_2d')
        if bbox_count > 1:
            multiple_bbox_count += 1

        # 检查是否成功提取bbox
        if predict_bbox == [0., 0., 0., 0.]:
            failed_extractions += 1
            if verbose:
                print(f'Sample {idx}: Failed to extract bbox from response')
                print(f"Response: {sample['answer']}")

        # 转换为tensor
        target_bbox = torch.tensor(gt_bbox, dtype=torch.float32).view(-1, 4)
        predict_bbox_tensor = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)

        # 计算IoU
        iou, _ = box_iou(predict_bbox_tensor, target_bbox)
        iou_value = iou.item()

        # 累加统计
        total_cnt += 1
        total_iou += iou_value
        ious.append(iou_value)

        if iou_value >= iou_threshold:
            correct += 1

        # 详细输出前几个样本或低IoU样本
        if verbose and (idx < 5 or iou_value < 0.1):
            print(f'\nSample {idx}:')
            print(f'GT bbox: {gt_bbox}')
            print(f'Pred bbox: {predict_bbox}')
            print(f'IoU: {iou_value:.4f}')
            if idx < 5:
                print(f"Response snippet: {sample['answer'][:200]}...")

        # 可视化指定样本
        if visualize and visualize_samples is not None and idx in visualize_samples:
            try:
                visualize_sample(idx, results, dataset_jsonl_path, output_dir)
            except Exception as e:
                print(f'Error visualizing sample {idx}: {e}')

    # 计算precision和平均IoU
    precision = correct / total_cnt if total_cnt > 0 else 0
    avg_iou = total_iou / total_cnt if total_cnt > 0 else 0

    # 输出结果
    print(f"\n{'='*50}")
    print(f'EVALUATION RESULTS')
    print(f"{'='*50}")
    print(f'Total samples: {total_cnt}')
    print(f'Samples with multiple bboxes: {multiple_bbox_count} ({multiple_bbox_count/total_cnt*100:.1f}%)')
    print(f'Failed bbox extractions: {failed_extractions} ({failed_extractions/total_cnt*100:.1f}%)')
    print(f'Correct predictions (IoU >= {iou_threshold}): {correct}')
    print(f'Precision @ {iou_threshold}: {precision:.4f} ({precision*100:.2f}%)')
    print(f'Average IoU: {avg_iou:.4f}')

    # IoU分布统计
    ious_tensor = torch.tensor(ious)
    print(f'\nIoU Distribution:')
    print(f'Min IoU: {ious_tensor.min().item():.4f}')
    print(f'Max IoU: {ious_tensor.max().item():.4f}')
    print(f'Median IoU: {ious_tensor.median().item():.4f}')
    print(f'IoU >= 0.1: {(ious_tensor >= 0.1).sum().item()} ({(ious_tensor >= 0.1).float().mean().item()*100:.1f}%)')
    print(f'IoU >= 0.3: {(ious_tensor >= 0.3).sum().item()} ({(ious_tensor >= 0.3).float().mean().item()*100:.1f}%)')
    print(f'IoU >= 0.5: {(ious_tensor >= 0.5).sum().item()} ({(ious_tensor >= 0.5).float().mean().item()*100:.1f}%)')
    print(f'IoU >= 0.7: {(ious_tensor >= 0.7).sum().item()} ({(ious_tensor >= 0.7).float().mean().item()*100:.1f}%)')
    print(f'IoU >= 0.9: {(ious_tensor >= 0.9).sum().item()} ({(ious_tensor >= 0.9).float().mean().item()*100:.1f}%)')

    return {
        'precision': precision,
        'avg_iou': avg_iou,
        'total_samples': total_cnt,
        'correct_predictions': correct,
        'failed_extractions': failed_extractions,
        'multiple_bbox_count': multiple_bbox_count,
        'ious': ious
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate grounding precision from JSON results')
    parser.add_argument(
        '--json-file',
        type=str,
        default='/mnt/afs/xuyangcao/code/ms-swift/eval/results/med_sam2_eval_250710070744_med.json',
        help='Path to the JSON results file')
    parser.add_argument(
        '--iou-threshold', type=float, default=0.5, help='IoU threshold for counting correct predictions')
    parser.add_argument(
        '--selection-strategy',
        type=str,
        choices=['first', 'largest', 'smallest'],
        default='first',
        help='Strategy for selecting bbox when multiple bboxes are found')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information for debugging')

    # 可视化相关参数
    parser.add_argument('--visualize', action='store_true', help='Enable visualization of results')
    parser.add_argument(
        '--dataset-jsonl',
        type=str,
        default='/mnt/afs/xuyangcao/shared_datasets/SA-Med2D-20M/raw/SAMed2Dv1/evaluation_dataset.jsonl',
        help='Path to the dataset JSONL file (required for visualization)')
    parser.add_argument('--output-dir', type=str, default='visualizations', help='Directory to save visualizations')
    parser.add_argument(
        '--visualize-samples',
        type=str,
        default='all',
        help="Samples to visualize: 'all', '0-10', or '0,5,10' (default: first 10)")

    args = parser.parse_args()

    # 处理可视化样本参数
    vis_samples = None
    if args.visualize:
        if args.visualize_samples == 'all':
            # 加载结果文件以获取样本总数
            with open(args.json_file, 'r') as f:
                results_temp = json.load(f)
            vis_samples = list(range(len(results_temp)))
        elif args.visualize_samples is not None:
            vis_samples = args.visualize_samples

    # 计算precision
    results = calculate_precision(
        json_file_path=args.json_file,
        iou_threshold=args.iou_threshold,
        verbose=args.verbose,
        selection_strategy=args.selection_strategy,
        visualize=args.visualize,
        dataset_jsonl_path=args.dataset_jsonl if args.visualize else None,
        output_dir=args.output_dir,
        visualize_samples=vis_samples)

    print(f"\nFinal Precision @ {args.iou_threshold}: {results['precision']:.4f}")
    print(f"Total samples with multiple bboxes: {results['multiple_bbox_count']}")

    if args.visualize:
        print(f'Visualizations saved to: {args.output_dir}')

import argparse
import os
import re

import json
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def extract_bbox_from_response(response, bbox_format='json'):
    """从模型响应中提取边界框坐标"""
    if bbox_format == 'standard':
        # 标准格式: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
        box_pattern = r'<\|box_start\|>\(([\d]+),([\d]+)\),\(([\d]+),([\d]+)\)<\|box_end\|>'
        match = re.search(box_pattern, response)
        if match:
            x1, y1, x2, y2 = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
            return [x1, y1, x2, y2]
    else:  # json format
        # JSON格式: {"bbox_2d": [x1, y1, x2, y2], "label": "..."}
        try:
            json_pattern = r'({.*?"bbox_2d":\s*\[[\d\s,\.]+\].*?})'
            json_match = re.search(json_pattern, response, re.DOTALL)

            if json_match:
                json_str = json_match.group(1)
                # 处理可能的注释或额外文本
                json_str = re.sub(r'```json|```', '', json_str).strip()
                bbox_data = json.loads(json_str)

                if 'bbox_2d' in bbox_data:
                    # 提取坐标
                    bbox_coords = bbox_data['bbox_2d']
                    return [int(float(x)) for x in bbox_coords]
        except Exception as e:
            print(f'Error parsing JSON bbox: {e}')

    # 如果提取失败，返回空边界框
    return [0, 0, 0, 0]


def load_jsonl(file_path):
    """加载JSONL文件内容"""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def load_json(file_path):
    """加载JSON文件内容"""
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_iou(box1, box2):
    """计算两个边界框的IoU值"""
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)

    # 计算各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union = area1 + area2 - intersection

    # 计算IoU
    return intersection / union if union > 0 else 0.0


def visualize_result(index, results, dataset, bbox_format='json', output_dir='visualizations'):
    """可视化特定索引的预测结果和真实标签"""
    # 确保索引有效
    if index < 0 or index >= len(results):
        print(f'Error: Index {index} out of range (0-{len(results)-1})')
        return False

    # 获取指定索引的结果和数据
    result = results[index]
    data = dataset[index]

    # 获取图像路径、边界框和类别名称
    image_path = data['image']
    gt_bbox = data['bbox']
    category = data['sentence']

    # 从响应中提取预测边界框
    pred_bbox = extract_bbox_from_response(result['answer'], bbox_format)

    # 计算IoU
    iou = calculate_iou(gt_bbox, pred_bbox)

    # 如果是批量处理，则只打印索引和IoU
    if args.all:
        print(f'Processing {index}: IoU = {iou:.2f}')
    else:
        print(f'Image: {image_path}')
        print(f'Category: {category}')
        print(f'GT Bbox: {gt_bbox}')
        print(f'Pred Bbox: {pred_bbox}')
        print(f'IoU: {iou:.4f}')

    # 加载图像
    try:
        image = Image.open(image_path).convert('RGB')  # Convert to RGB mode
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
    draw.rectangle([(gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3])], outline='green', width=2)

    # 绘制预测边界框（红色）
    draw.rectangle([(pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3])], outline='red', width=2)

    # 为真实边界框添加标签（GT: 类别名）
    gt_label = f'GT: {category}'
    _, _, text_w, text_h = draw.textbbox((0, 0), text=gt_label, font=font)
    draw.rectangle([(gt_bbox[0], gt_bbox[1] - text_h - 2), (gt_bbox[0] + text_w + 2, gt_bbox[1])], fill='green')
    draw.text((gt_bbox[0] + 1, gt_bbox[1] - text_h - 1), gt_label, fill='white', font=font)

    # 为预测边界框添加标签（Pred: 类别名）
    pred_label = f'Pred: {category}'
    _, _, text_w, text_h = draw.textbbox((0, 0), text=pred_label, font=font)
    draw.rectangle([(pred_bbox[0], pred_bbox[3] + 2), (pred_bbox[0] + text_w + 2, pred_bbox[3] + text_h + 4)],
                   fill='red')
    draw.text((pred_bbox[0] + 1, pred_bbox[3] + 3), pred_label, fill='white', font=font)

    # 在图像顶部添加IoU信息
    iou_label = f'IoU: {iou:.2f}'
    _, _, text_w, text_h = draw.textbbox((0, 0), text=iou_label, font=font)
    # 在图像顶部居中位置绘制IoU值
    pos_x = (image.width - text_w) // 2
    draw.rectangle([(pos_x - 5, 10), (pos_x + text_w + 5, 10 + text_h + 5)], fill='blue')
    draw.text((pos_x, 12), iou_label, fill='white', font=font)

    # 保存结果
    output_path = os.path.join(output_dir, f'result_{index}.png')
    image.save(output_path)

    if not args.all:
        print(f'Visualization saved to {output_path}')

    return True


def batch_visualize_results(results_file, dataset_file, bbox_format='json', output_dir='visualizations'):
    """批量可视化所有结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载结果和数据集
    results = load_json(results_file)
    dataset = load_jsonl(dataset_file)

    print(f'Processing {len(results)} results...')

    # 用于存储成功/失败的计数
    success_count = 0
    failure_count = 0
    total_iou = 0.0

    # 使用tqdm显示进度条
    for idx in tqdm(range(len(results)), desc='Visualizing results'):
        if visualize_result(idx, results, dataset, bbox_format, output_dir):
            success_count += 1

            # 计算并累加IoU
            gt_bbox = dataset[idx]['bbox']
            pred_bbox = extract_bbox_from_response(results[idx]['answer'], bbox_format)
            iou = calculate_iou(gt_bbox, pred_bbox)
            total_iou += iou
        else:
            failure_count += 1

    # 输出统计信息
    print('\nVisualization completed!')
    print(f'Total processed: {len(results)}')
    print(f'Successful: {success_count}')
    print(f'Failed: {failure_count}')

    # 计算平均IoU
    if success_count > 0:
        avg_iou = total_iou / success_count
        print(f'Average IoU: {avg_iou:.4f}')

    print(f'Results saved to {output_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize detection results')
    parser.add_argument('--index', type=int, default=100, help='Index of the result to visualize')
    parser.add_argument(
        '--results',
        type=str,
        default='/mnt/afs/xuyangcao/code/ms-swift/eval/results/med_sam2_eval_250805100303.json',
        help='Path to the results JSON file')
    parser.add_argument(
        '--dataset',
        type=str,
        default='/mnt/afs/xuyangcao/shared_datasets/SA-Med2D-20M/raw/SAMed2Dv1/evaluation_dataset.jsonl',
        help='Path to the dataset JSONL file')
    parser.add_argument(
        '--bbox-format', type=str, choices=['standard', 'json'], default='json', help='Bounding box format in results')
    parser.add_argument('--output-dir', type=str, default='visualizations', help='Directory to save visualizations')
    parser.add_argument('--all', action='store_true', help='Process all samples instead of just one')
    args = parser.parse_args()

    if args.all:
        batch_visualize_results(args.results, args.dataset, args.bbox_format, args.output_dir)
    else:
        # 加载结果和数据集
        results = load_json(args.results)
        dataset = load_jsonl(args.dataset)

        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)

        # 可视化单个结果
        visualize_result(args.index, results, dataset, args.bbox_format, args.output_dir)

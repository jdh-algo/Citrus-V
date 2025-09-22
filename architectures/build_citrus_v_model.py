import argparse
import os
from typing import Any, Tuple, cast

import torch
from architectures.citrus_v_transformers import CitrusV, CitrusVConfig
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer


def build_model(ori_config, mllm_path, sam2_path, save_model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = CitrusVConfig.from_json_file(ori_config)

    print('\n=== load tokenizer ===')
    tokenizer = AutoTokenizer.from_pretrained(mllm_path)

    print('\n=== load processor ===')
    processor = AutoProcessor.from_pretrained(mllm_path)

    # save tokenizer and processor
    os.makedirs(save_model_path, exist_ok=True)
    tokenizer.save_pretrained(save_model_path)
    processor.save_pretrained(save_model_path)
    print(f'Tokenizer and Processor has been saved to {save_model_path}')

    # create model
    print('\n=== create model ===')
    model = CitrusV(config)

    print('\n=== load original model config ===')
    model.load_ori_state_dict(mllm_path, sam2_path)

    # save model and config
    print('\n=== save model and config ===')
    model.save_pretrained(save_model_path)
    print(f'Model has been saved to {save_model_path}')

    return save_model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_config', type=str, default='architectures/config_citrus_8B.json')
    parser.add_argument('--mllm_path', type=str, default='/mnt/workspace/offline/shared_models/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--sam2_path', type=str, default='architectures/third_parts/sam2')
    parser.add_argument('--save_model_path', type=str, default='architectures/saved_models/citrus_v_8B')
    args = parser.parse_args()

    model_path = build_model(args.ori_config, args.mllm_path, args.sam2_path, args.save_model_path)

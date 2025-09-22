import logging

import torch
from huggingface_hub import hf_hub_download

HF_HUB_PREFIX = 'hf-hub:'


def load_checkpoint_with_prefix(filename, prefix=None, map_location='cpu', logger='current'):
    """Load partial pretrained model with specific prefix.

    Args:
        prefix (str): The prefix of sub-module.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`.
            Defaults to None.
        logger: logger

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    if filename.startswith('hf-hub:'):
        model_id = filename[len(HF_HUB_PREFIX):]
        filename = hf_hub_download(model_id, 'pytorch_model.bin')

    checkpoint = torch.load(filename, map_location=map_location)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if not prefix:
        return state_dict
    if not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    state_dict = {k[prefix_len:]: v for k, v in state_dict.items() if k.startswith(prefix)}

    assert state_dict, f'{prefix} is not in the pretrained model'
    return state_dict


def load_state_dict_to_model(model, state_dict, logger='current', assign=True):
    # print(f'===== load state dict to model ...... in grounder')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, assign=True)
    # print(f"====== missing_keys: {missing_keys}")
    # print(f"====== unexpected_keys: {unexpected_keys}")

    if missing_keys:
        logging.error(f'Missing keys: {missing_keys}')
        raise RuntimeError('Missing keys found when loading state dict')
    if unexpected_keys:
        logging.error(f'Unexpected keys: {unexpected_keys}')
        raise RuntimeError('Unexpected keys found when loading state dict')
    logging.info('Loaded checkpoint successfully')
    print(f'成功加载模型权重')

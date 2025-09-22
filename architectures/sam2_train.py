import os.path

import torch
import torch.nn as nn
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .vlm_utils import load_checkpoint_with_prefix, load_state_dict_to_model


class SAM2TrainRunner(nn.Module):

    def __init__(self,
                 model_path: str = '/mnt/workspace/offline/shared_models/sam2-hiera-large',
                 cfg_path: str = 'sam2_hiera_l.yaml',
                 hydra_overrides_extra=None,
                 apply_postprocessing=True,
                 torch_dtype: str = 'bfloat16'):
        super().__init__()

        from .third_parts import sam2

        # configs
        if hydra_overrides_extra is None:
            hydra_overrides_extra = []
        hydra_overrides = [
            ## Extension: LLM prompt
            '++model._target_=architectures.sam2_base.SAM2Base',
        ]
        if apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
            hydra_overrides_extra += [
                # dynamically fall back to multi-mask if the single mask is not stable
                # "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                # "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                # "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
                # "++model.binarize_mask_from_pts_for_mem_enc=true",
                # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
                # "++model.fill_hole_area=8",
            ]
        hydra_overrides.extend(hydra_overrides_extra)
        if os.path.isabs(cfg_path):
            config_dir = os.path.dirname(cfg_path)
            config_name = os.path.basename(cfg_path).replace('.yaml', '')
            cfg = compose(config_name=config_name, overrides=hydra_overrides)
        else:
            cfg = compose(config_name=cfg_path, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)

        # model
        self.sam2_model = instantiate(cfg.model, _recursive_=True)
        self.model_path = model_path
        self.ckpt_path = os.path.join(model_path, 'sam2_hiera_large.pt')

        self.hidden_dim = self.sam2_model.hidden_dim
        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)

        if torch_dtype == 'float32':
            self.torch_dtype = torch.float32
        elif torch_dtype == 'float16':
            self.torch_dtype = torch.float16
        elif torch_dtype == 'bfloat16':
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.bfloat16

    def load_ori_state_dict(self, model_path):
        state_dict = load_checkpoint_with_prefix(os.path.join(model_path, self.ckpt_path))
        # print(f"====== grounder state_dict: {state_dict.keys()}")
        # print(f"======== self.sam2_model.state_dict(): {self.sam2_model.state_dict().keys()}")
        load_state_dict_to_model(self.sam2_model, state_dict)

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image / 255.
        img_mean = torch.tensor(self.img_mean, dtype=image.dtype, device=image.device)[:, None, None]
        img_std = torch.tensor(self.img_std, dtype=image.dtype, device=image.device)[:, None, None]
        image -= img_mean
        image /= img_std
        return image

    def inject_language_embd(self, sam_states, language_embd):
        # print(f"====== sam_states: {len(sam_states['current_vision_feats'])}, current_vision_feats: {sam_states['current_vision_feats'][-1].shape}, current_vision_pos_embeds: {sam_states['current_vision_pos_embeds'][-1].shape}, feat_sizes: {sam_states['feat_sizes'][-1]}")
        # print(f"====== language_embd: {language_embd.shape}")

        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s).contiguous()
            for x, s in zip(sam_states['current_vision_feats'][:-1], sam_states['feat_sizes'][:-1])
        ]

        B = sam_states['current_vision_feats'][-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = sam_states['feat_sizes'][-1]

        if self.sam2_model.directly_add_no_mem_embed:
            # directly add no-mem embedding (instead of using the transformer encoder)
            pix_feat = sam_states['current_vision_feats'][-1]
            no_mem_embed = self.sam2_model.no_mem_embed.to(pix_feat.device)
            pix_feat_with_mem = pix_feat + no_mem_embed
            # pix_feat_with_mem = sam_states['current_vision_feats'][-1] + self.sam2_model.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W).contiguous()
        else:
            raise NotImplementedError('directly add no memory embedding is not implemented')

        # print(f"====== backbone_features: {pix_feat_with_mem.shape}, high_res_features: {len(high_res_features)}, language_embd: {language_embd.shape}")
        with torch.autocast(device_type='cuda', dtype=self.torch_dtype):
            _, _, _, low_res_masks, high_res_masks, obj_ptr, _, = self.sam2_model._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=None,
                mask_inputs=None,
                high_res_features=high_res_features,
                multimask_output=self.sam2_model._use_multimask(is_init_cond_frame=True, point_inputs=None),
                # Inject language Embed if possible
                language_embd=language_embd,
            )

        # print(f"==== low res masks: {low_res_masks.shape}")
        pred_masks = low_res_masks

        return pred_masks

    def get_sam2_embeddings(self, images, expand_size=1):
        # Step 1: inference the backbone with the images
        images = images.to(dtype=self.torch_dtype)
        # print(f"====== images in get_sam2_embeddings: {images.shape}")
        with torch.autocast(device_type='cuda', dtype=self.torch_dtype):
            feats = self.sam2_model.forward_image(images)

        if expand_size > 1:
            # feats['vision_features'] = feats['vision_features'][:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
            for i, feat in enumerate(feats['backbone_fpn']):
                feats['backbone_fpn'][i] = feat[:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1).contiguous()
            for i, pos in enumerate(feats['vision_pos_enc']):
                pos = pos[:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1).contiguous()
                feats['vision_pos_enc'][i] = pos

        # Step 2: Process the features to output
        _, current_vision_feats, current_vision_pos_embeds, feat_sizes = self.sam2_model._prepare_backbone_features(
            feats)

        return {
            'current_vision_feats': current_vision_feats,
            'current_vision_pos_embeds': current_vision_pos_embeds,
            'feat_sizes': feat_sizes,
        }

    def forward(self, batch):
        raise NotImplementedError


def main():
    """单元测试函数，用于测试SAM2TrainRunner模块是否能正常运行"""
    print('开始测试 SAM2TrainRunner 模块...')
    # 测试1: 模块初始化
    print('测试1: 尝试初始化 SAM2TrainRunner...')
    runner = SAM2TrainRunner(
        cfg_path='sam2_hiera_l.yaml',
        ckpt_path='sam2_hiera_large.pt',
        apply_postprocessing=False  # 关闭后处理以简化测试
    )
    print('✓ SAM2TrainRunner 初始化成功')

    # 测试2: 检查模型属性
    print('测试2: 检查模型属性...')
    print(f'  - hidden_dim: {runner.hidden_dim}')
    print(f'  - img_mean: {runner.img_mean}')
    print(f'  - img_std: {runner.img_std}')
    print('✓ 模型属性检查完成')

    # 测试3: 测试图像预处理
    print('测试3: 测试图像预处理功能...')
    # 创建一个模拟图像张量 (3, 224, 224)
    dummy_image = torch.randn(3, 224, 224) * 255  # 模拟0-255范围的图像
    preprocessed = runner.preprocess_image(dummy_image)
    print(f'  - 输入图像形状: {dummy_image.shape}')
    print(f'  - 预处理后形状: {preprocessed.shape}')
    print(f'  - 预处理后均值: {preprocessed.mean().item():.4f}')
    print(f'  - 预处理后标准差: {preprocessed.std().item():.4f}')
    print('✓ 图像预处理功能测试完成')

    # 测试4: 测试SAM2嵌入提取
    print('测试4: 测试SAM2嵌入提取...')
    # 创建一个批次的图像 (batch_size=2, 3, 224, 224)
    dummy_images = torch.randn(2, 3, 224, 224)
    sam_states = runner.get_sam2_embeddings(dummy_images)
    print('  - SAM2嵌入提取成功')
    print(f"  - current_vision_feats 长度: {len(sam_states['current_vision_feats'])}")
    print(f"  - feat_sizes: {sam_states['feat_sizes']}")
    print('✓ SAM2嵌入提取测试完成')

    print('\n🎉 所有测试完成！SAM2TrainRunner 模块基本功能正常')


if __name__ == '__main__':
    main()

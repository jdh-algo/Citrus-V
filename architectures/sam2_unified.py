"""
统一的SAM2模型，兼容transformers格式
支持训练和推理两种模式，自动根据配置选择相应的实现
"""

import os.path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from architectures.vlm_utils import load_checkpoint_with_prefix, load_state_dict_to_model
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


class SAM2UnifiedConfig(PretrainedConfig):
    """统一的SAM2模型配置类"""

    model_type = 'sam2_unified'

    def __init__(
            self,
            model_path: str = '/mnt/workspace/offline/shared_models/sam2-hiera-large',
            cfg_path: str = 'sam2_hiera_l.yaml',
            mode: str = 'auto',  # "auto", "train", "inference"
            apply_postprocessing: bool = True,
            torch_dtype: str = 'bfloat16',
            hydra_overrides_extra: Optional[List[str]] = None,
            **kwargs):
        super().__init__(**kwargs)

        self.model_path = model_path
        self.cfg_path = cfg_path
        self.mode = mode
        self.apply_postprocessing = apply_postprocessing
        self.torch_dtype = torch_dtype
        self.hydra_overrides_extra = hydra_overrides_extra or []

        # 自动检测模式
        if self.mode == 'auto':
            # 根据环境变量或配置自动选择模式
            if os.environ.get('SAM2_MODE') == 'train':
                self.mode = 'train'
            elif os.environ.get('SAM2_MODE') == 'inference':
                self.mode = 'inference'
            else:
                # 默认使用训练模式（更安全）
                self.mode = 'train'


class SAM2Unified(PreTrainedModel):
    """
    统一的SAM2模型，兼容transformers格式
    自动根据配置选择训练或推理模式
    """

    config_class = SAM2UnifiedConfig
    base_model_prefix = 'sam2_unified'
    supports_gradient_checkpointing = True

    def __init__(self, config: SAM2UnifiedConfig):
        super().__init__(config)

        # 解析数据类型
        if config.torch_dtype == 'float32':
            self.torch_dtype = torch.float32
        elif config.torch_dtype == 'float16':
            self.torch_dtype = torch.float16
        elif config.torch_dtype == 'bfloat16':
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.bfloat16

        # 初始化SAM2模型
        self._init_sam2_model()

        # 设置图像预处理参数
        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)

        # 获取隐藏维度
        self.hidden_dim = self.sam2_model.hidden_dim

        # 设置模型路径
        self.model_path = config.model_path
        self.ckpt_path = os.path.join(config.model_path, 'sam2_hiera_large.pt')

        # 初始化权重
        self.post_init()

    def _init_sam2_model(self):
        """初始化SAM2模型，根据模式选择不同的实现"""
        from .third_parts import sam2  # noqa: F401

        # 根据模式选择不同的目标类
        if self.config.mode == 'train':
            target_class = 'architectures.sam2_base.SAM2Base'
        elif self.config.mode == 'inference':
            target_class = 'architectures.sam2_predictor.SAM2VideoPredictor'
        else:
            raise ValueError(f'Unknown mode: {self.config.mode}')

        # 构建hydra配置
        hydra_overrides = [
            f'++model._target_={target_class}',
        ]

        if self.config.apply_postprocessing:
            postprocessing_overrides = [
                # 动态多掩码回退
                '++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true',
                '++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05',
                '++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98',
                # 其他后处理选项可以根据需要添加
            ]
            hydra_overrides.extend(postprocessing_overrides)

        # 添加额外的覆盖
        if self.config.hydra_overrides_extra:
            hydra_overrides.extend(self.config.hydra_overrides_extra)

        # 解析配置
        if os.path.isabs(self.config.cfg_path):
            config_dir = os.path.dirname(self.config.cfg_path)
            config_name = os.path.basename(self.config.cfg_path).replace('.yaml', '')
            cfg = compose(config_name=config_name, overrides=hydra_overrides, config_dir=config_dir)
        else:
            cfg = compose(config_name=self.config.cfg_path, overrides=hydra_overrides)

        OmegaConf.resolve(cfg)

        # 实例化模型
        self.sam2_model = instantiate(cfg.model, _recursive_=True)

        print(f'✅ SAM2模型初始化成功，模式: {self.config.mode}')

    def post_init(self):
        """后初始化，加载预训练权重"""
        # 这里可以添加权重加载逻辑
        pass

    def load_ori_state_dict(self):
        """加载原始状态字典"""
        state_dict = load_checkpoint_with_prefix(self.ckpt_path)
        load_state_dict_to_model(self.sam2_model, state_dict)
        print(f'✅ 原始权重加载成功: {self.ckpt_path}')

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """图像预处理"""
        image = image / 255.
        img_mean = torch.tensor(self.img_mean, dtype=image.dtype, device=image.device)[:, None, None]
        img_std = torch.tensor(self.img_std, dtype=image.dtype, device=image.device)[:, None, None]
        image -= img_mean
        image /= img_std
        return image

    def get_sam2_embeddings(self, images: torch.Tensor, expand_size: int = 1) -> Dict[str, Any]:
        """获取SAM2嵌入特征"""
        images = images.to(dtype=self.torch_dtype)

        if self.config.mode == 'train':
            # 训练模式：使用forward_image
            with torch.autocast(device_type='cuda', dtype=self.torch_dtype):
                feats = self.sam2_model.forward_image(images)

            if expand_size > 1:
                # 扩展特征维度
                for i, feat in enumerate(feats['backbone_fpn']):
                    feats['backbone_fpn'][i] = feat[:, None].expand(-1, expand_size, -1, -1,
                                                                    -1).flatten(0, 1).contiguous()
                for i, pos in enumerate(feats['vision_pos_enc']):
                    pos = pos[:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1).contiguous()
                    feats['vision_pos_enc'][i] = pos

            # 准备特征
            _, current_vision_feats, current_vision_pos_embeds, feat_sizes = self.sam2_model._prepare_backbone_features(
                feats)

            return {
                'current_vision_feats': current_vision_feats,
                'current_vision_pos_embeds': current_vision_pos_embeds,
                'feat_sizes': feat_sizes,
            }

        else:
            # 推理模式：使用init_state
            return self.sam2_model.init_state(images)

    def inject_language_embd(self, sam_states: Dict[str, Any], language_embd: torch.Tensor) -> torch.Tensor:
        """注入语言嵌入，生成分割掩码"""
        if self.config.mode == 'train':
            # 训练模式：使用_forward_sam_heads
            return self._inject_language_embd_train(sam_states, language_embd)
        else:
            # 推理模式：使用add_language_embd
            return self._inject_language_embd_inference(sam_states, language_embd)

    def _inject_language_embd_train(self, sam_states: Dict[str, Any], language_embd: torch.Tensor) -> torch.Tensor:
        """训练模式的语言嵌入注入"""
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s).contiguous()
            for x, s in zip(sam_states['current_vision_feats'][:-1], sam_states['feat_sizes'][:-1])
        ]

        B = sam_states['current_vision_feats'][-1].size(1)
        C = self.hidden_dim
        H, W = sam_states['feat_sizes'][-1]

        if self.sam2_model.directly_add_no_mem_embed:
            # 直接添加无记忆嵌入
            pix_feat = sam_states['current_vision_feats'][-1]
            no_mem_embed = self.sam2_model.no_mem_embed.to(pix_feat.device)
            pix_feat_with_mem = pix_feat + no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W).contiguous()
        else:
            raise NotImplementedError('directly add no memory embedding is not implemented')

        with torch.autocast(device_type='cuda', dtype=self.torch_dtype):
            _, _, _, low_res_masks, high_res_masks, obj_ptr, _, = self.sam2_model._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=None,
                mask_inputs=None,
                high_res_features=high_res_features,
                multimask_output=self.sam2_model._use_multimask(is_init_cond_frame=True, point_inputs=None),
                language_embd=language_embd,
            )

        return low_res_masks

    def _inject_language_embd_inference(self, inference_state: Any,
                                        language_embd: List[List[torch.Tensor]]) -> torch.Tensor:
        """推理模式的语言嵌入注入"""
        num_frame = len(language_embd)
        num_obj = len(language_embd[0])
        mask_out = []

        for frame_idx in range(num_frame):
            frame_mask_out = []
            for obj_idx in range(num_obj):
                _language_embd = language_embd[frame_idx][obj_idx][None][None]
                _, _, out_mask_logits = self.sam2_model.add_language_embd(inference_state, frame_idx, obj_idx + 100,
                                                                          _language_embd)
                frame_mask_out.append(out_mask_logits)
            frame_mask_out = torch.cat(frame_mask_out, dim=1)
            mask_out.append(frame_mask_out)

        mask_out = torch.cat(mask_out, dim=0)
        return mask_out

    def language_embd_inference(self, inference_state: Any, language_embd: List[List[torch.Tensor]]) -> torch.Tensor:
        """推理模式的语言嵌入推理（视频传播）"""
        if self.config.mode != 'inference':
            raise ValueError('language_embd_inference only available in inference mode')

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            mask_out = []
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model.propagate_in_video(inference_state):
                mask_out.append(out_mask_logits)
            mask_out = torch.cat(mask_out, dim=0)

        return mask_out

    def forward(self, batch: Any) -> Any:
        """前向传播（需要根据具体任务实现）"""
        raise NotImplementedError('forward method needs to be implemented for specific tasks')

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """获取可训练参数"""
        return list(self.parameters())

    def get_frozen_parameters(self) -> List[nn.Parameter]:
        """获取冻结参数"""
        return []

    def set_mode(self, mode: str):
        """动态设置模式"""
        if mode not in ['train', 'inference']:
            raise ValueError(f'Invalid mode: {mode}')

        if mode != self.config.mode:
            print(f'⚠️ 警告：动态切换模式从 {self.config.mode} 到 {mode} 可能不安全')
            # 这里可以实现动态模式切换的逻辑
            pass

    def save_pretrained(self, save_directory: str, **kwargs):
        """保存预训练模型"""
        super().save_pretrained(save_directory, **kwargs)

        # 保存SAM2特定配置
        sam2_config = {
            'model_path': self.model_path,
            'cfg_path': self.config.cfg_path,
            'mode': self.config.mode,
            'apply_postprocessing': self.config.apply_postprocessing,
            'torch_dtype': self.config.torch_dtype,
        }

        config_path = os.path.join(save_directory, 'sam2_config.json')
        import json
        with open(config_path, 'w') as f:
            json.dump(sam2_config, f, indent=2)

        print(f'✅ SAM2配置已保存到: {config_path}')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        """从预训练模型加载"""
        # 首先调用父类方法
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # 尝试加载SAM2特定配置
        sam2_config_path = os.path.join(pretrained_model_name_or_path, 'sam2_config.json')
        if os.path.exists(sam2_config_path):
            import json
            with open(sam2_config_path, 'r') as f:
                sam2_config = json.load(f)

            # 更新配置
            for key, value in sam2_config.items():
                if hasattr(model.config, key):
                    setattr(model.config, key, value)

            print(f'✅ SAM2配置已从 {sam2_config_path} 加载')

        return model


# 兼容性别名
SAM2TrainRunner = SAM2Unified
SAM2 = SAM2Unified


def create_sam2_model(mode: str = 'auto',
                      model_path: str = '/mnt/workspace/offline/shared_models/sam2-hiera-large',
                      cfg_path: str = 'sam2_hiera_l.yaml',
                      apply_postprocessing: bool = True,
                      torch_dtype: str = 'bfloat16',
                      **kwargs) -> SAM2Unified:
    """创建SAM2模型的便捷函数"""

    config = SAM2UnifiedConfig(
        model_path=model_path,
        cfg_path=cfg_path,
        mode=mode,
        apply_postprocessing=apply_postprocessing,
        torch_dtype=torch_dtype,
        **kwargs)

    return SAM2Unified(config)


# 测试函数
def test_sam2_unified():
    """测试统一的SAM2模型"""
    print('=== 测试统一的SAM2模型 ===')

    # 测试1：训练模式
    print('\n--- 测试训练模式 ---')
    try:
        train_model = create_sam2_model(mode='train', cfg_path='sam2_hiera_l.yaml', apply_postprocessing=False)
        print('✅ 训练模式模型创建成功')
        print(f'  模式: {train_model.config.mode}')
        print(f'  隐藏维度: {train_model.hidden_dim}')
        print(f'  数据类型: {train_model.torch_dtype}')
    except Exception as e:
        print(f'❌ 训练模式模型创建失败: {e}')
        import traceback
        traceback.print_exc()

    # 测试2：推理模式
    print('\n--- 测试推理模式 ---')
    try:
        inference_model = create_sam2_model(mode='inference', cfg_path='sam2_hiera_l.yaml', apply_postprocessing=True)
        print('✅ 推理模式模型创建成功')
        print(f'  模式: {inference_model.config.mode}')
        print(f'  隐藏维度: {inference_model.hidden_dim}')
        print(f'  数据类型: {inference_model.torch_dtype}')
    except Exception as e:
        print(f'❌ 推理模式模型创建失败: {e}')
        import traceback
        traceback.print_exc()

    # 测试3：自动模式
    print('\n--- 测试自动模式 ---')
    try:
        auto_model = create_sam2_model(mode='auto', cfg_path='sam2_hiera_l.yaml', apply_postprocessing=False)
        print('✅ 自动模式模型创建成功')
        print(f'  模式: {auto_model.config.mode}')
        print(f'  隐藏维度: {auto_model.hidden_dim}')
        print(f'  数据类型: {auto_model.torch_dtype}')
    except Exception as e:
        print(f'❌ 自动模式模型创建失败: {e}')
        import traceback
        traceback.print_exc()

    print('\n🎉 测试完成！')


if __name__ == '__main__':
    test_sam2_unified()

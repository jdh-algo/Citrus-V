import os.path

import torch
import torch.nn as nn
from architectures.vlm_utils import load_checkpoint_with_prefix, load_state_dict_to_model
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


class SAM2(nn.Module):

    def __init__(self,
                 model_path: str = '/mnt/workspace/offline/shared_models/sam2-hiera-large',
                 cfg_path: str = 'sam2_hiera_l.yaml',
                 hydra_overrides_extra=None,
                 apply_postprocessing=True,
                 torch_dtype: str = 'bfloat16'):
        super().__init__()

        from .third_parts import sam2  # noqa: F401

        if hydra_overrides_extra is None:
            hydra_overrides_extra = []
        hydra_overrides = [
            ## Extension: LLM prompt
            '++model._target_=architectures.sam2_predictor.SAM2VideoPredictor',
        ]

        if apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
            hydra_overrides_extra += [
                # dynamically fall back to multi-mask if the single mask is not stable
                '++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true',
                '++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05',
                '++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98',
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
        self.ckpt_path = os.path.join(self.model_path, 'sam2_hiera_large.pt')
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

    def load_ori_state_dict(self, ):
        state_dict = load_checkpoint_with_prefix(os.path.join(self.model_path, self.ckpt_path))
        # print(f"======== self.sam2_model.state_dict(): {self.sam2_model.state_dict().keys()}")
        load_state_dict_to_model(self.sam2_model, state_dict)

    def inject_language_embd(self, inference_state, language_embd):
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

    def language_embd_inference(self, inference_state, language_embd):
        # print(f"===  using sam2.language_embd_inference....")
        num_frame = len(language_embd)
        num_obj = len(language_embd[0])
        mask_out = []
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for frame_idx in range(num_frame):
                frame_mask_out = []

                for obj_idx in range(num_obj):
                    _language_embd = language_embd[frame_idx][obj_idx][None][None]
                    _, _, out_mask_logits = self.sam2_model.add_language_embd(
                        inference_state,
                        frame_idx,
                        obj_idx + 100,
                        _language_embd,
                        inference=True,
                    )
                    frame_mask_out.append(out_mask_logits)
                frame_mask_out = torch.cat(frame_mask_out, dim=1)
                mask_out.append(frame_mask_out)

            mask_out = []
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model.propagate_in_video(inference_state):
                mask_out.append(out_mask_logits)
            mask_out = torch.cat(mask_out, dim=0)
        return mask_out

    def get_sam2_embeddings(self, images):
        # print(f"===== using sam2 to get embeddings")
        return self.sam2_model.init_state(images)

    def forward(self, batch):
        raise NotImplementedError

    def preprocess_image(self, image: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
        image = image / 255.

        img_mean = torch.tensor(self.img_mean, dtype=dtype, device=image.device)[:, None, None]
        img_std = torch.tensor(self.img_std, dtype=dtype, device=image.device)[:, None, None]
        image -= img_mean
        image /= img_std

        return image

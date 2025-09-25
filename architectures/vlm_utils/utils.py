import torch

def deepcopy_generate_kwargs(input_ids, pixel_values, image_grid_thw, attention_mask, device, useargs, **kwargs):
        def _clone_tensor(t):
            if isinstance(t, torch.Tensor):
                return t.detach().clone().to(device)
            return t
        if not useargs:
            generate_kwargs = dict(
                input_ids=_clone_tensor(input_ids),
                pixel_values=_clone_tensor(pixel_values),
                image_grid_thw=_clone_tensor(image_grid_thw),
                attention_mask=_clone_tensor(attention_mask),
            )
        else:
            generate_kwargs = dict(
                input_ids=_clone_tensor(input_ids),
                pixel_values=_clone_tensor(pixel_values),
                image_grid_thw=_clone_tensor(image_grid_thw),
                attention_mask=_clone_tensor(attention_mask),
                **{k: _clone_tensor(v) for k, v in kwargs.items()}
            )
        return generate_kwargs
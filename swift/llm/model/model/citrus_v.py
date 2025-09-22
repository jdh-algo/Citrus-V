# register the custom config file and class to Transformers
from architectures.citrus_v_transformers import CitrusV, CitrusVConfig
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from swift.llm import TemplateType
from ..model_arch import ModelArch
from ..patcher import patch_get_input_embeddings, patch_output_clone, patch_output_to_input_device
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal, register_model
from .qwen import patch_qwen_vl_utils

AutoConfig.register('citrus_v', CitrusVConfig)
AutoModel.register(CitrusVConfig, CitrusV)
AutoModelForCausalLM.register(CitrusVConfig, CitrusV)


def get_model_tokenizer_citrus_v(*args, **kwargs):
    from architectures.citrus_v_transformers import CitrusV
    kwargs['automodel_class'] = kwargs['automodel_class'] or CitrusV

    model, tokenizer = get_model_tokenizer_multimodal(*args, **kwargs)

    if model is not None:
        base_model = model.model if 'AWQ' in model.__class__.__name__ else model
        if hasattr(base_model.model, 'embed_tokens'):
            embed_tokens = base_model.mllm.model.embed_tokens
        else:
            embed_tokens = base_model.mllm.model.language_model.embed_tokens
        patch_output_clone(embed_tokens)
        patch_output_to_input_device(embed_tokens)
        patch_get_input_embeddings(base_model.visual, 'patch_embed')

    # assign the result of environment variables to vision_process
    from qwen_vl_utils import vision_process
    patch_qwen_vl_utils(vision_process)

    return model, tokenizer


# register the model to swift
register_model(
    ModelMeta(
        model_type='citrus_v',
        model_groups=[
            ModelGroup([Model(model_path='/mnt/afs/xuyangcao/code/ms-swift/architectures/saved_models/citrus_v_8B')]),
        ],
        template=TemplateType.citrus_v,
        get_function=get_model_tokenizer_citrus_v,
        model_arch=ModelArch.citrus_v,
        is_multimodal=True,
        architectures=['CitrusV'],
    ))

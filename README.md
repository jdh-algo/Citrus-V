# Citrus-V: Advancing Medical Foundation Models with Unified Medical Image Grounding for Clinical Reasoning

<p align="center">
    <br>
    <img src="asset/Citrus-V-Architecture.png"/>
    <em>Citrus-V Model Structure</em>
    <br>
</p>

## 📖 Table of Contents
- [Introduction](#-introduction)
- [Key Features](#-key-features)
- [Installation](#%EF%B8%8F-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Datasets](#-datasets)
- [Evaluation](#-evaluation)
- [License](#-license)
- [Citation](#-citation)

## 📝 Introduction
In clinical practice, physicians routinely operate in highly multimodal environments, where medical imaging plays a central role in diagnosis, treatment planning, and surgical decision-making. Accurate interpretation of imaging data is indispensable, as it provides critical evidence that complements textual reports, laboratory results, and patient history. Consequently, any artificial intelligence system intended for clinical deployment must be capable of integrating visual and textual information at a fine-grained, pixel-level resolution while supporting structured reasoning and clinically grounded decision-making.

Existing medical imaging models are largely designed as expert systems specialized for narrow tasks such as lesion detection, segmentation, classification, or report generation. These models often require multiple specialized networks to cover different organs, disease types, or diagnostic tasks, and they rarely generalize effectively across diverse clinical scenarios. While large-scale language and multimodal models have demonstrated remarkable progress, including strong reasoning capabilities and multi-task generalization, applying them to real-world clinical settings remains challenging.

Clinical tasks demand not only multimodal understanding but also precise visual grounding and integrated chain-of-thought reasoning to interpret complex medical data, support decision-making workflows, and provide reliable second opinions with explainability and clinical fidelity. Existing multimodal medical approaches often fail to provide pixel-level, fine-grained visual insights or to integrate heterogeneous data modalities effectively, which limits their utility in comprehensive diagnostic reasoning.

Building upon our prior work, Citrus: Leveraging Expert Cognitive Pathways in a Medical Language Model for Advanced Medical Decision Support, which introduced a language-based medical foundation model incorporating expert-inspired reasoning pathways, we now present Citrus-V. This upgraded multimodal medical foundation model addresses the critical need for integrating medical images into clinical decision support systems.

## ✨ Key Contributions
Citrus-V makes the following key contributions to the field of medical AI:

1. **Unified Integration of Visual and Reasoning Capabilities**: We construct a unified model that integrates detection, segmentation, and multimodal chain-of-thought reasoning, enabling pixel-level lesion localization, structured report generation, and physician-like diagnostic inference within a single model.

2. **Comprehensive Open-Source Data Suite**: To facilitate reproducibility and support the research community, we release Citrus-V along with a curated open-source data suite, including:
   - A multimodal chain-of-thought reasoning dataset for report generation
   - A refined detection and segmentation benchmark with corrected labels
   - A medical document understanding benchmark with graded difficulty levels

3. **Novel Multimodal Training Paradigm**: We design a novel multimodal training paradigm to accelerate convergence and enhance generalization across diverse imaging and reasoning tasks.

Extensive experiments demonstrate that Citrus-V surpasses existing open-source medical foundation models and expert-level imaging systems across multiple benchmarks, establishing new state-of-the-art performance in both visual and multimodal tasks. By providing a complete pipeline from visual grounding to clinical reasoning, Citrus-V offers critical support for precise lesion quantification, automated radiology reporting, and reliable second opinions, marking a significant step toward general-purpose medical foundation models and the broader adoption of AI in clinical practice.

## 🔍 Key Features
- **Unified Medical Image Grounding**: Advanced techniques for precise localization and understanding of medical images at the pixel level
- **Comprehensive Clinical Reasoning**: Integration of medical knowledge graphs and clinical guidelines with multimodal chain-of-thought reasoning
- **Multi-modal Medical Understanding**: Seamlessly process images, text, and structured data from electronic health records
- **Medical Image Analysis**: Support for various medical imaging modalities (CT, MRI, X-ray, ultrasound, etc.) with detection and segmentation capabilities
- **Medical OCR**: Specialized optical character recognition for medical documents and reports
- **Fine-grained Control**: Adjustable parameters for different medical specialties and use cases
- **Efficient Training Pipeline**: Optimized for medical datasets with packing and streaming capabilities

## 🛠️ Installation


To install Citrus-V:

1. Create base environment.
    ```shell
    conda create -n citrus_v python=3.10 -y
    conda activate citrus_v
    ```

2. Install requirements.
    ```bash
    git clone https://github.com/jdh-algo/Citrus-V.git
    cd Citrus-V
    pip install -r requirements_citrus.txt
    ```

3. Install [flash-attention](https://github.com/Dao-AILab/flash-attention) according to your environment. Here we used `flash-attn==2.7.3`.

4. Install Citrus-V training environment. (Based on [ms-swift](https://github.com/modelscope/ms-swift)).
    ```shell
    pip install -e .
    ```

<p align="center">
    <br>
    <img src="asset/fig_train_stages.png"/>
    <br>
    <em>Four Training Stages of the Citrus-V</em>
    <br>
</p>


## 🚀 Quick Start
### Training Section
HHere’s a quick example to get started with Citrus-V: this repo provides a pretrained checkpoint that has completed Stage 1 and Stage 2. The repo is designed for Stage 3 and Stage 4 training.
The key difference is that Stage 3 performs full-network tuning and includes the HookGrad module. Stage 4 is the SAM-adaptation phase: every component is frozen except the SegProjector and SAM modules, which are jointly updated to align segmentation prompts with the Segment-Anything paradigm.

#### training stage 3
```shell
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=8 \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model {pretrained ckpt address} \
    --dataset {your dataset address} \
    --template citrus_v \
    --train_type full \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --max_length 12288 \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --warmup_ratio 0 \
    --warmup_steps 100 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --dataloader_num_workers 64 \
    --dataset_num_proc 1 \
    --freeze_vit false \
    --freeze_aligner false \
    --freeze_llm false \
    --save_strategy epoch \
    --save_total_limit 8 \
    --logging_steps 5 \
    --output_dir {your model save path}\
    --save_only_model \
    --gradient_checkpointing true \
    --ddp_find_unused_parameters true
```

#### training stage 4
```shell
YTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=8 \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model {pretrained ckpt address} \
    --dataset {your dataset address} \
    --template citrus_v \
    --train_type full \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --max_length 12288 \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --warmup_ratio 0 \
    --warmup_steps 100 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --dataloader_num_workers 64 \
    --dataset_num_proc 1 \
    --freeze_vit true \
    --freeze_aligner true \
    --freeze_llm true \
    --save_strategy epoch \
    --save_total_limit 8 \
    --logging_steps 5 \
    --output_dir {your model save path}\
    --save_only_model \
    --gradient_checkpointing true \
    --ddp_find_unused_parameters true
```

### Testing Section


### Gradio Demo

1. Deploy the model
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=65535 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift deploy \
    --model /path/to/model \
    --served_model_name CitrusV_8B \
    --infer_backend pt \
    --torch_dtype bfloat16 \
    --port 8000
```

1. Start the gradio app

```bash 
cd projects
python app.py
```


## 🏛 License
This project is licensed under the Apache License (Version 2.0). For models and datasets, please refer to the original resource page and follow the corresponding License.

## 📎 Citation
If you use Citrus-V in your research, please cite our work:

```bibtex
@article{citrusv2024,
  title={Citrus-V: Advancing Medical Foundation Models with Unified Medical Image Grounding for Clinical Reasoning},
  author={Your Name and Contributors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

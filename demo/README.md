# Citrus-V Demo

## 🚀 Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/jdh-algo/Citrus-V.git
   cd demo
   ```
2. Install dependencies: [install-guide](../README.md##🛠️-Installation)
3. Setup model service:

   ```bash
   CUDA_VISIBLE_DEVICES=0 \
   MAX_PIXELS=MAX_PIXELS=1003520 \
   VIDEO_MAX_PIXELS=50176 \
   FPS_MAX_FRAMES=12 \
   swift deploy \
      --model path_to_model_ckpt \
      --served_model_name CitrusV_8B \
      --template citrus_v_infer \
      --infer_backend pt \
      --torch_dtype bfloat16 \
      --port 8000
   ```
4. Running the Demo

   1. Option1: Run compact gradio demo focused on segmentation

   ```bash
   python citrusv_gradio_app.py
   ```

   - The demo will be available at `http://localhost:7863`
   - Configure the model parameters in `citrusv_gradio_app.py` when needed.

   2. Option2: Run full-featured streamlit demo

   ```bash
   streamlit run citrusv_gradio_app.py
   ```

   - The demo will be available at `http://localhost:8501`
   - Configure the model parameters in the sidebar when needed.
   - Select application tab for better result display.
   - Click "Load example images" to view example images. Select existing examples to run, or use your own input.
   - Enable/disable streaming mode as needed. Segementation app doesn't support streaming mode.

## 📁 Project Structure

```
CITRUS_V/demo/
├── citrusv_gradio_app.py         # Compact gradio demo focused on segmentation
├── citrusv_streamlit_app.py      # Full-featured streamlit demo
├── utils.py                      # Utility functions for model communication
├── display_utils.py              # Display and visualization utilities
├── demo_images/                  # Example images for each task type
│   ├── QA/                       # Question answering examples
│   ├── MDU/                      # Document understanding examples
│   ├── Report/                   # Report generation examples
│   └── Segmentation/             # Segmentation examples
└── README.md                     # This file
```

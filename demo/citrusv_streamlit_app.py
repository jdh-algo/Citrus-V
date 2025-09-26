import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import requests
import json
import pandas as pd
import os
from typing import Optional, Dict, List, Tuple
import random
import time
import qwen_vl_utils
from utils import call_vllm_model, load_image, call_vllm_model_stream
from display_utils import (
    display_answer_with_images,
    extract_last_bbox_from_text,
    draw_bbox_on_image,
    extract_bbox_from_response,
    postprocess_segmentation_response,
    visualize_segmentation,
)

# Page configuration
st.set_page_config(page_title="CitrusV demo", page_icon="⛨", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better display
st.markdown(
    """
<style>
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .example-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }
    .example-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    .task-header {
        font-size: 1.2em;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 10px;
    }
    .config-section {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
    }
    /* Ensure consistent height for alignment */
    .stImage > img {
        height: 200px;
        object-fit: cover;
        width: 100%;
    }
    /* Align empty containers */
    .stEmpty {
        min-height: 200px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Task types
TASK_TYPES = {
    "QA": "Medical Question Answering",
    "MDU": "Medical Document Understanding",
    "Report": "Medical Image Report Generation",
    "Segmentation": "Medical Image Detection and Segmentation",
}

# Example data for different tasks
EXAMPLE_DATA = {
    "QA": [
        {
            "question": "图中体积最大的器官是什么? 请简要回答该问题。",
            "image": "demo_images/QA/001.jpg",
        },
        {
            "question": "这张图片中黑色的器官可以被用来做什么? ",
            "image": "demo_images/QA/002.jpg",
        },
        {
            "question": "Which side of lung is abnormal in this image,left or right? Answer the question using a single word or phrase.",
            "image": "demo_images/QA/003.jpg",
        },
        {
            "question": "影像左侧是患者的左肺还是右肺？如何判断",
            "image": "demo_images/QA/003.jpg",
        },
    ],
    "MDU": [
        {
            "question": "请将图中检验单表格解析为Markdown格式，表头包括检验项目、结果、单位、参考范围",
            "image": "demo_images/MDU/001.png",
        },
        {
            "question": "请识别检验单中的异常项目，输出JSON格式，包含检验项目名称、结果、参考范围和异常状态。异常状态包括：偏高、偏低、阳性、阴性、异常。只输出异常项目，正常项目不要输出。",
            "image": "demo_images/MDU/002.jpg",
        },
        {
            "question": "右侧乳腺低回声团的具体大小？",
            "image": "demo_images/MDU/003.jpg",
        },
        {
            "question": "请列出图中处方的诊断、药品名称、规格、单位、数量及用法用量。",
            "image": "demo_images/MDU/004.jpg",
        },
    ],
    "Report": [
        {
            "question": "Evaluate the provided chest radiograph and compile a report. Let's think step by step.",
            "image": "demo_images/Report/001.png",
        },
    ],
    "Segmentation": [
        {
            "question": "Please help segment the kidney in this scan.",
            "image": "demo_images/Segmentation/001.png",
        },
        {
            "question": "Please help segment the spleen in this scan.",
            "image": "demo_images/Segmentation/001.png",
        },
        {
            "question": "Please help segment the lung in this scan.",
            "image": "demo_images/Segmentation/001.png",
        },
        {
            "question": "Please help segment the liver in this scan.",
            "image": "demo_images/Segmentation/001.png",
        },
        {
            "question": "Please locate and detect the right lung in this image.",
            "image": "demo_images/Segmentation/001.png",
        },
        {
            "question": "Please detect the left lung in this image, and return the cooresponging coordinates.",
            "image": "demo_images/Segmentation/001.png",
        },
        {
            "question": "Please detect the liver in this image using json format.",
            "image": "demo_images/Segmentation/001.png",
        },
        {
            "question": "Please detect the kidney in this image using json format.",
            "image": "demo_images/Segmentation/001.png",
        },
    ],
}

# Set environment variables
os.environ["MAX_PIXELS"] = "1003520"
os.environ["VIDEO_MAX_PIXELS"] = "50176"
os.environ["FPS_MAX_FRAMES"] = "12"


# Initialize session state
def init_session_state():
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "current_answer" not in st.session_state:
        st.session_state.current_answer = ""
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "selected_task" not in st.session_state:
        st.session_state.selected_task = "QA"
    if "stream_mode" not in st.session_state:
        st.session_state.stream_mode = True
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "general"
    if "current_response_obj" not in st.session_state:
        st.session_state.current_response_obj = None
    if "segmentation_result_image" not in st.session_state:
        st.session_state.segmentation_result_image = None
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None
    if "example_images_loaded" not in st.session_state:
        st.session_state.example_images_loaded = False
    if "example_image_cache" not in st.session_state:
        st.session_state.example_image_cache = {}

    # Initialize model configurations
    if "model_config" not in st.session_state:
        st.session_state.model_config = {
            "enabled": False,
            "server_url": "http://localhost:8000/v1/chat/completions",
            "model_name": "CitrusV_8B",
            "api_key": "",
            "max_tokens": 8192,
            "temperature": 0.0,
        }


def count_unique_images(examples: List[Dict]) -> int:
    """Count unique image URLs in examples"""
    unique_image_urls = set()
    for example in examples:
        if example.get("image") and example["image"] and not example["image"].startswith("path/"):
            unique_image_urls.add(example["image"])
    return len(unique_image_urls)


def preload_example_images(examples: List[Dict]) -> None:
    """Preload all example images in the background"""
    # Get unique image URLs to avoid counting duplicates
    unique_image_urls = set()
    for example in examples:
        if example.get("image") and example["image"] and not example["image"].startswith("path/"):
            unique_image_urls.add(example["image"])

    for image_url in unique_image_urls:
        if image_url not in st.session_state.example_image_cache:
            img = load_image(image_url)
            if img:
                st.session_state.example_image_cache[image_url] = img


def display_example_card(example: Dict, task_type: str, idx: int):
    """Display a single example card with lazy loading support"""
    with st.container():
        # Check if example has an actual image
        has_image = example.get("image") and example["image"] and not example["image"].startswith("path/")
        image_url = example.get("image") if has_image else None

        if has_image:
            # Check if images are loaded and cached
            if st.session_state.example_images_loaded and image_url in st.session_state.example_image_cache:
                # Display cached image
                img = st.session_state.example_image_cache[image_url]
                st.image(img, use_container_width=True)
            elif st.session_state.example_images_loaded:
                # Try to load and cache the image
                img = load_image(image_url)
                if img:
                    st.session_state.example_image_cache[image_url] = img
                    st.image(img, use_container_width=True)
                else:
                    st.empty()  # Placeholder to maintain alignment
            else:
                # Show placeholder when images are not loaded
                st.info("📷")
        else:
            # Add empty space to maintain alignment
            st.empty()

        # Display question with appropriate label
        question_label = "Text-only Question" if not has_image else "Question"
        edited_question = st.text_area(
            question_label,
            value=example["question"],
            height=120,
            key=f"example_{task_type}_{idx}",
            disabled=False,  # Make editable
        )

        if st.button(f"Use this example", key=f"btn_{task_type}_{idx}", type="secondary"):
            st.session_state.current_question = edited_question  # Use edited version
            if has_image:
                # Load image when using the example
                img = load_image(image_url)
                if img:
                    st.session_state.current_image = img
            else:
                st.session_state.current_image = None
            st.session_state.selected_task = task_type
            st.rerun()


def main():
    init_session_state()

    # Title and description
    st.title("⛨ Citrus-V")

    # Sidebar for model configuration
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        st.session_state.model_config["server_url"] = st.text_input(
            "Server URL", value=st.session_state.model_config["server_url"], key="server_url"
        )

        st.session_state.model_config["model_name"] = st.text_input(
            "Model Name", value=st.session_state.model_config["model_name"], key="model_name"
        )

        st.session_state.model_config["api_key"] = st.text_input(
            "API Key", value=st.session_state.model_config["api_key"], type="password", key="api_key"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.model_config["max_tokens"] = st.number_input(
                "Max Tokens",
                min_value=1,
                max_value=32768,
                value=st.session_state.model_config["max_tokens"],
                key="general_max_tokens",
            )

        with col2:
            st.session_state.model_config["temperature"] = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.model_config["temperature"],
                step=0.1,
                key="general_temp",
            )

    # Main content area
    main_container = st.container()

    with main_container:
        # Create task selection options
        task_options = list(TASK_TYPES.keys())
        task_labels = [TASK_TYPES[key] for key in task_options]

        # Find current index
        current_idx = 0
        if st.session_state.selected_task in task_options:
            current_idx = task_options.index(st.session_state.selected_task)

        # Task selection radio buttons
        selected_task_idx = st.radio(
            "**Choose a task type for better display:**",
            options=range(len(task_options)),
            format_func=lambda x: task_labels[x],
            index=current_idx,
            horizontal=True,
            key="task_selector",
        )

        # Update selected task based on radio button selection
        previous_task = st.session_state.selected_task
        st.session_state.selected_task = task_options[selected_task_idx]

        # Clear all input and results when switching tasks
        if previous_task != st.session_state.selected_task:
            # Clear current inputs and results
            st.session_state.current_image = None
            st.session_state.current_question = ""
            st.session_state.current_answer = ""
            st.session_state.current_response_obj = None
            st.session_state.last_uploaded_file = None
            st.session_state.processing = False

            # Reset image loading state
            st.session_state.example_images_loaded = False
            st.session_state.example_image_cache = {}

        # Handle task-specific logic
        if st.session_state.selected_task == "Segmentation":
            st.session_state.stream_mode = False
        else:
            st.session_state.stream_mode = True

        # Examples section (collapsible)
        with st.expander(f"📚 {TASK_TYPES[st.session_state.selected_task]} Examples", expanded=False):
            if st.session_state.selected_task in EXAMPLE_DATA:
                examples = EXAMPLE_DATA[st.session_state.selected_task]

                if st.button("🖼️ Load Example Images", type="primary", help="Load example images (may take a moment)"):
                    st.session_state.example_images_loaded = True
                    st.session_state.example_image_cache = {}  # Clear cache to force reload
                    # Preload images with progress bar
                    preload_example_images(examples)
                    st.rerun()

                # Display all examples in 4 columns
                cols = st.columns(4)
                for i, example in enumerate(examples):
                    with cols[i % 4]:
                        display_example_card(example, st.session_state.selected_task, i)
            else:
                st.info("Examples coming soon...")

        st.divider()

        # Main interaction area
        col1, col2 = st.columns([1, 1.2])

        # Left column - Input
        with col1:
            st.header("📥 Input")

            # Task type indicator
            st.info(f"Current Task: **{TASK_TYPES[st.session_state.selected_task]}**")

            # Image upload (for tasks that need images - optional for QA)
            if st.session_state.selected_task in ["QA", "MDU", "Report", "Segmentation"]:
                uploaded_file = st.file_uploader(
                    "Upload Image (Optional for QA)" if st.session_state.selected_task == "QA" else "Upload Image",
                    type=["png", "jpg", "jpeg", "gif", "bmp", "webp", "dcm"],
                    help="Upload medical image for analysis",
                )

                if uploaded_file is not None:
                    st.session_state.current_image = Image.open(uploaded_file)
                    st.session_state.last_uploaded_file = uploaded_file.name
                elif st.session_state.last_uploaded_file is not None and uploaded_file is None:
                    # Only clear if we previously had an uploaded file and now it's None
                    st.session_state.current_image = None
                    st.session_state.last_uploaded_file = None

                # Display current image
                if st.session_state.current_image:
                    st.image(
                        st.session_state.current_image,
                        caption="Current Image",
                        use_container_width=True,
                    )

                    # Image info
                    with st.expander("📊 Image Information"):
                        img = st.session_state.current_image
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.write(f"**Dimensions:** {img.size[0]} × {img.size[1]} px")
                            st.write(f"**Mode:** {img.mode}")
                        with col_info2:
                            buffered = BytesIO()
                            img.save(buffered, format="PNG")
                            size_kb = len(buffered.getvalue()) / 1024
                            st.write(f"**Size:** {size_kb:.1f} KB")
                            if hasattr(img, "format"):
                                st.write(f"**Format:** {img.format}")
                else:
                    st.info("Please upload an image or select an example")

        # Right column - Question and Answer
        with col2:
            st.header("💬 Analysis")
            if st.session_state.selected_task == "Segmentation":
                st.session_state.stream_mode = False
            else:
                st.session_state.stream_mode = st.checkbox("streaming mode", value=st.session_state.stream_mode)

            def update_question():
                """Callback function to update question in session state"""
                st.session_state.current_question = st.session_state.question_input

            question = st.text_area(
                "Question/Instruction",
                value=st.session_state.current_question,
                height=120,
                placeholder="Enter your question...",
                key="question_input",
                on_change=update_question,
            )

            # Action buttons
            col_analyze, col_clear = st.columns([9, 1])

            with col_analyze:
                # Determine if analyze should be disabled
                analyze_disabled = st.session_state.processing
                if st.session_state.selected_task == "QA":
                    # For QA, need at least a question (image is optional)
                    analyze_disabled = analyze_disabled or not st.session_state.current_question
                else:
                    # For other tasks, follow original logic
                    analyze_disabled = analyze_disabled or (
                        not question and st.session_state.selected_task not in ["Report", "Segmentation"]
                    )

                if st.button("🔍 Analyze", type="primary", disabled=analyze_disabled, use_container_width=True):
                    st.session_state.processing = True
                    if st.session_state.stream_mode and st.session_state.selected_task != "Segmentation":
                        # Streaming output (not available for segmentation yet)
                        if st.session_state.selected_task == "Report":
                            # Special handling for Report with bbox support
                            stream_text_placeholder = st.empty()
                            stream_image_placeholder = st.empty()
                            accumulated_text = ""
                            last_drawn_bbox = None

                            # Immediately show the image before any bbox is detected
                            if st.session_state.current_image is not None:
                                resized_image = qwen_vl_utils.fetch_image({"image": st.session_state.current_image})
                                stream_image_placeholder.image(
                                    resized_image, caption="Current Region of Interest", width=400
                                )

                            for chunk in call_vllm_model_stream(
                                st.session_state.current_image,
                                st.session_state.current_question,
                                st.session_state.model_config,
                            ):
                                if chunk is None:
                                    continue
                                accumulated_text += chunk
                                # Update streamed text
                                stream_text_placeholder.markdown(accumulated_text)

                                # Try to extract the most recent bbox and draw it
                                try:
                                    latest_bbox = extract_last_bbox_from_text(accumulated_text)
                                    if latest_bbox and st.session_state.current_image is not None:
                                        # Avoid redrawing if unchanged
                                        if last_drawn_bbox != latest_bbox:
                                            resized_image = qwen_vl_utils.fetch_image(
                                                {"image": st.session_state.current_image}
                                            )
                                            img_with_last = draw_bbox_on_image(
                                                resized_image.copy(), latest_bbox, (255, 0, 0)
                                            )
                                            stream_image_placeholder.image(
                                                img_with_last, caption="Current Region of Interest", width=400
                                            )
                                            last_drawn_bbox = latest_bbox
                                except Exception:
                                    pass

                            st.session_state.current_answer = accumulated_text
                            st.session_state.processing = False
                        else:
                            # Standard streaming for other tasks
                            stream_placeholder = st.empty()
                            accumulated_text = ""

                            for chunk in call_vllm_model_stream(
                                st.session_state.current_image,
                                st.session_state.current_question,
                                st.session_state.model_config,
                            ):
                                if chunk is None:
                                    continue
                                accumulated_text += chunk
                                stream_placeholder.markdown(accumulated_text)

                            st.session_state.current_answer = accumulated_text
                            st.session_state.processing = False
                    else:
                        with st.spinner("⏳ Analyzing... This may take a few moments..."):
                            if st.session_state.selected_task == "Segmentation":
                                # For segmentation, we need the full response to extract mask data
                                response_obj, answer = call_vllm_model(
                                    st.session_state.current_image,
                                    st.session_state.current_question,
                                    st.session_state.model_config,
                                    return_seg_det=True,
                                )
                                st.session_state.current_answer = answer
                                st.session_state.current_response_obj = response_obj

                                # Process segmentation results
                                if st.session_state.current_image is not None and response_obj is not None:
                                    # Extract masks from response
                                    img, masks = postprocess_segmentation_response(
                                        response_obj, st.session_state.current_image
                                    )

                                    # Get original and processed image sizes
                                    original_image_size = st.session_state.current_image.size
                                    processed_image_size = None

                                    # Try to get processed size from response
                                    if hasattr(response_obj, "raw_response") and response_obj.raw_response:
                                        raw_response = response_obj.raw_response
                                        if isinstance(raw_response, dict) and "image_grid_thw" in raw_response:
                                            image_grid_thw = raw_response["image_grid_thw"]
                                            if image_grid_thw and len(image_grid_thw) > 0:
                                                processed_height = image_grid_thw[0][1] * 28
                                                processed_width = image_grid_thw[0][2] * 28
                                                processed_image_size = (processed_width, processed_height)

                                    # If processed_image_size is not found, use smart_resize to estimate it
                                    if processed_image_size is None:
                                        from qwen_vl_utils.vision_process import (
                                            smart_resize,
                                            MIN_PIXELS,
                                            MAX_PIXELS,
                                            IMAGE_FACTOR,
                                        )

                                        env_max_pixels = int(os.environ.get("MAX_PIXELS", MAX_PIXELS))
                                        env_min_pixels = int(os.environ.get("MIN_PIXELS", MIN_PIXELS))

                                        orig_width, orig_height = original_image_size
                                        processed_height, processed_width = smart_resize(
                                            orig_height,
                                            orig_width,
                                            factor=IMAGE_FACTOR,
                                            min_pixels=env_min_pixels,
                                            max_pixels=env_max_pixels,
                                        )
                                        processed_image_size = (processed_width, processed_height)
                                        # print(f"Image size after smart resize: {processed_image_size}")
                                        # print(f"MAX_PIXELS={env_max_pixels}, MIN_PIXELS={env_min_pixels}")

                                    # Extract bbox
                                    predict_bbox = extract_bbox_from_response(
                                        answer,
                                        original_image_size=original_image_size,
                                        processed_image_size=processed_image_size,
                                    )

                                    # Visualize results
                                    if masks is not None or predict_bbox is not None:
                                        result_image = visualize_segmentation(img, masks, predict_bbox)
                                        st.session_state.segmentation_result_image = result_image
                                    else:
                                        st.session_state.segmentation_result_image = None
                            else:
                                answer = call_vllm_model(
                                    st.session_state.current_image,
                                    st.session_state.current_question,
                                    st.session_state.model_config,
                                )
                                st.session_state.current_answer = answer
                            st.session_state.processing = False
                            st.rerun()

            with col_clear:
                if st.button("🗑️", use_container_width=True):
                    st.session_state.current_image = None
                    st.session_state.current_question = ""
                    st.session_state.current_answer = ""
                    st.session_state.current_response_obj = None
                    st.session_state.segmentation_result_image = None
                    st.session_state.processing = False
                    st.session_state.last_uploaded_file = None
                    st.rerun()

            # Display results
            if st.session_state.current_answer:
                st.divider()
                st.subheader("📋 Results")

                # Special handling for different task types
                if st.session_state.selected_task == "Segmentation":
                    # Display segmentation results
                    if st.session_state.segmentation_result_image is not None:
                        st.image(
                            st.session_state.segmentation_result_image,
                            use_container_width=True,
                        )

                        # Download button for result image
                        buf = BytesIO()
                        st.session_state.segmentation_result_image.save(buf, format="PNG")
                        buf.seek(0)
                        st.download_button(
                            label="📥 Download Result",
                            data=buf,
                            file_name="segmentation_result.png",
                            mime="image/png",
                        )
                    else:
                        st.info("No segmentation masks or bounding boxes detected")

                    st.markdown(st.session_state.current_answer)
                    with st.expander("📋 View raw text"):
                        st.code(st.session_state.current_answer, language="text", wrap_lines=True)
                        st.info("Select and copy the text above")

                elif st.session_state.selected_task == "Report":
                    # Format as a medical report with bbox support
                    with st.container():
                        # Use display utilities for report with bbox support
                        display_answer_with_images(st.session_state.current_answer, st.session_state.current_image, st)
                        with st.expander("📋 View raw text"):
                            st.code(st.session_state.current_answer, language="text", wrap_lines=True)
                            st.info("Select and copy the text above")
                else:
                    # Standard display for QA, MDU
                    if st.session_state.current_answer.startswith("Error:"):
                        st.error(st.session_state.current_answer)
                    else:
                        try:
                            # optimize json answer
                            json_answer = json.loads(st.session_state.current_answer)
                            st.session_state.current_answer = (
                                f"```json\n{json.dumps(json_answer, ensure_ascii=False, indent=4)}\n```"
                            )
                        except Exception:
                            pass
                        st.markdown(st.session_state.current_answer)
                        with st.expander("📋 View raw text"):
                            st.code(st.session_state.current_answer, language="text", wrap_lines=True)
                            st.info("Select and copy the text above")


if __name__ == "__main__":
    main()

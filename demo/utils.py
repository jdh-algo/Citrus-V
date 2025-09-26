import random
import os
import tempfile
from PIL import Image
import requests
from io import BytesIO
import base64
from typing import Optional, Tuple, Dict, Any
import streamlit as st
import json
import re

def load_image(url: str) -> Optional[Image.Image]:
    """Load an example image from URL or local file path"""
    if url is None or url == "":
        return None
    try:
        if url.startswith("http"):
            # Handle URL
            response = requests.get(url, timeout=10)
            return Image.open(BytesIO(response.content))
        else:
            # Handle local file path
            if not os.path.isabs(url):
                app_dir = os.path.dirname(os.path.abspath(__file__))
                full_path = os.path.join(app_dir, url)
                return Image.open(full_path)
            else:
                return Image.open(url)
    except Exception as e:
        st.error(f"Failed to load image: {str(e)}")
        return None


def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for API calls"""
    buffered = BytesIO()
    if image.mode in ("RGBA", "LA", "P"):
        rgb_image = Image.new("RGB", image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
        image = rgb_image
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def call_vllm_model(
    image: Image.Image,
    question: str,
    cfg: dict,
    return_seg_det: bool = False,
    timeout_sec: int = 300,
) -> Any:
    """
    Call VLLM backend with image and question.

    Args:
        image: PIL Image object
        question: Query text
        cfg: Model configuration dictionary
        return_seg_det: If True, return tuple of (response_dict, text_content) for segmentation
        timeout_sec: Request timeout in seconds

    Returns:
        If return_seg_det=True: Tuple of (response_dict, text_content)
        Otherwise: text_content string
    """
    try:
        # Initialize variables
        oss_url = None
        image_base64 = None

        if image is None:
            payload = {
                "model": cfg["model_name"],
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": question}],
                    }
                ],
            }
        else:
            # Use base64 encoding with PNG format
            image_base64 = encode_image_to_base64(image)
            payload = {
                "model": cfg["model_name"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                            {"type": "text", "text": question},
                        ],
                    }
                ],
            }

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg['api_key']}"}

        if "max_tokens" in cfg:
            payload["max_tokens"] = cfg["max_tokens"]
        if "temperature" in cfg:
            payload["temperature"] = cfg["temperature"]

        response = requests.post(
            cfg["server_url"],
            json=payload,
            headers=headers,
            timeout=(10, timeout_sec),
        )

        if response.status_code == 200:
            result = response.json()

            # For segmentation detection, return the full result dict and text content
            if return_seg_det:
                raw_response = result.get("raw_response", None)
                
                text_content = ""
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    message = choice.get("message", {})
                    text_content = message.get("content", "")
                # Create a response wrapper object that mimics the OpenAI response structure
                class ResponseWrapper:
                    def __init__(self, raw_response, text_content):
                        self.raw_response = raw_response
                        self.text_content = text_content
                    def to_dict(self):
                        return {"raw_response": self.raw_response}

                response_obj = ResponseWrapper(raw_response, text_content)
                return response_obj, text_content

            # For normal calls, just return the text content
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "Error: Unexpected response format from server"
        else:
            error_msg = f"Error: Server returned status code {response.status_code}: {response.text}"
            if return_seg_det:
                return None, error_msg
            return error_msg

    except requests.exceptions.Timeout:
        error_msg = "Error: Request timed out. The server might be busy or unreachable."
        if return_seg_det:
            return None, error_msg
        return error_msg
    except requests.exceptions.ConnectionError:
        error_msg = "Error: Could not connect to the VLLM server. Please check if the server is running."
        if return_seg_det:
            return None, error_msg
        return error_msg
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if return_seg_det:
            return None, error_msg
        return error_msg


def call_vllm_model_stream(
    image: Image.Image,
    question: str,
    cfg: dict,
    timeout_sec: int = 300,
):
    """
    Stream responses from VLLM backend as a generator that yields text chunks.
    """
    oss_url = None
    image_base64 = None
    try:
        if image is None:
            payload = {
                "model": cfg["model_name"],
                "messages": [{"role": "user", "content": [{"type": "text", "text": question}]}],
                "stream": True,
            }
        else:
            image_base64 = encode_image_to_base64(image)
            payload = {
                "model": cfg["model_name"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ],
                "stream": True,
            }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['api_key']}",
        }
        if "max_tokens" in cfg:
            payload["max_tokens"] = cfg["max_tokens"]
        if "temperature" in cfg:
            payload["temperature"] = cfg["temperature"]
        response = requests.post(
            cfg["server_url"],
            json=payload,
            headers=headers,
            timeout=(10, timeout_sec),
            stream=True,
        )
        # Ensure UTF-8 decoding for SSE to avoid mojibake
        try:
            response.encoding = "utf-8"
        except Exception:
            pass

        accumulated = []

        if response.status_code != 200:
            error_msg = f"Error: Server returned status code {response.status_code}: {response.text}"
            # Yield once so UI can display the error immediately
            yield error_msg
            return

        content_type = (response.headers.get("Content-Type") or "").lower()
        # Treat ndjson as streaming too
        is_event_stream = "text/event-stream" in content_type or "ndjson" in content_type

        if is_event_stream:
            chunk_count = 0
            content_chunks = 0

            # Use simplified streaming approach that works
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                # Skip empty lines and comments
                if not line.strip() or line.startswith(":"):
                    continue

                # Extract data from SSE format
                if line.startswith("data:"):
                    data = line[5:].strip()  # Remove "data:" prefix
                else:
                    data = line.strip()

                # Skip empty data
                if not data:
                    continue

                try:
                    # Parse JSON
                    obj = json.loads(data)
                    chunk_count += 1

                    # Extract content - try multiple formats
                    delta_text = ""

                    # Standard OpenAI streaming format
                    if "choices" in obj and obj["choices"]:
                        choice = obj["choices"][0]
                        if "delta" in choice:
                            delta = choice["delta"]
                            # Concatenate normal content and reasoning content if provided
                            content_piece = delta.get("content", "")
                            reasoning_piece = delta.get("reasoning_content", "")
                            delta_text = f"{content_piece}{reasoning_piece}"
                        elif "message" in choice:
                            message = choice["message"]
                            delta_text = message.get("content", "")

                    # Alternative formats
                    if not delta_text and "text" in obj:
                        delta_text = obj["text"]
                    if not delta_text and "content" in obj:
                        delta_text = obj["content"]

                    if delta_text:
                        content_chunks += 1
                        accumulated.append(delta_text)
                        yield delta_text
                except Exception as e:
                    print(f"Exception in streaming: {e}")
        else:
            # Non-SSE fallback: return full JSON/text once
            try:
                result = response.json()
                final_text_once = ""
                if "choices" in result and result["choices"]:
                    choice0 = result["choices"][0]
                    if isinstance(choice0, dict) and "message" in choice0:
                        msg = choice0.get("message", {}) or {}
                        # Prefer standard content; if empty, fall back to reasoning content
                        final_text_once = msg.get("content") or ""
                        if not final_text_once:
                            final_text_once = msg.get("reasoning_content") or ""
                    else:
                        final_text_once = choice0.get("text", "") or ""
                else:
                    final_text_once = json.dumps(result, ensure_ascii=False)
                if final_text_once:
                    accumulated.append(final_text_once)
                    yield final_text_once
                else:
                    print(f"No text extracted from non-SSE response")
            except Exception as e:
                print(f"Exception in non-SSE fallback: {e}")
                raw = response.text or ""
                if raw:
                    accumulated.append(raw)
                    yield raw

    except requests.exceptions.Timeout:
        yield "Error: Request timed out. The server might be busy or unreachable."
    except requests.exceptions.ConnectionError:
        yield "Error: Could not connect to the VLLM server. Please check if the server is running."
    except Exception as e:
        yield f"Error: {str(e)}"

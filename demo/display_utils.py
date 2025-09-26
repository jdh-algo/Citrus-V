# *************************** Report Generation ***************************
"""
Display utilities for medical report generation with bounding box support.
"""
import re
import os
from typing import Optional, List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
import qwen_vl_utils

#    {"Part":"label","bbox":"[x1, y1, x2, y2]"}
#    'label' with bbox [x1, y1, x2, y2]
#    'label' bbox [x1, y1, x2, y2]
BBOX_PATTERN = re.compile(
    r"""
    (?ix)                                   # case-insensitive, verbose

    # -------------------------
    # Alt A: {"Part":"label","bbox":"[x1, y1, x2, y2]"}
    # -------------------------
    \{\s*
      (?:"?(?:part|label)"?)\s*:\s*"(?P<label>[^"]+?)"\s*,\s*
      "?(?:bbox|box|rect)"?\s*:\s*"?\[\s*
        (?P<x1>-?\d+)\s*,\s*
        (?P<y1>-?\d+)\s*,\s*
        (?P<x2>-?\d+)\s*,\s*
        (?P<y2>-?\d+)\s*
      \]\s*"?\s*
    \}

    |

    # -------------------------
    # Alt B: 
    # 'right_lung' with bbox [127, 131, 485, 720]
    # 'lung_lower_lobe_right' bbox [126, 420, 485, 722]
    # label may be quoted with ' or ", or be a bare snake_case token
    # -------------------------
    (?P<label2>'[^']+'|"[^"]+"|[a-z][a-z0-9_]*)
    \s*
    (?:with\s*)?bbox
    \s*
    \[\s*
      (?P<x1b>-?\d+)\s*,\s*
      (?P<y1b>-?\d+)\s*,\s*
      (?P<x2b>-?\d+)\s*,\s*
      (?P<y2b>-?\d+)\s*
    \]
    """,
    re.VERBOSE | re.IGNORECASE,
)


def preprocess_text_for_bbox(text: str) -> str:
    """
    Preprocess text to replace Chinese characters with English equivalents
    to simplify bbox pattern matching
    """
    ch2en = {
        "、": ", ", "“": '"', "”": '"', "（": "(", "）": ")", "【": "[", "】": "]", 
        "，": ",", "。": ".", "；": ";", "：": ":", "！": "!", "？": "?"
    }
    for ch, en in ch2en.items():
        text = text.replace(ch, en)
    return text


def find_bbox_matches(text: str) -> List[re.Match]:
    """
    Return all matches for any supported bbox format in `text`.
    """
    preprocessed = preprocess_text_for_bbox(text)
    return list(re.finditer(BBOX_PATTERN, preprocessed))


def parse_bbox_from_match(m: re.Match) -> Dict[str, object]:
    """
    Convert a single regex match to a canonical dict:
      { "label": <str>, "coords": [x1, y1, x2, y2] }
    """

    def _strip_quotes(s: str) -> str:
        if not s:
            return s
        s = s.strip()
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            return s[1:-1]
        return s

    if m.group("label") is not None:
        # Alt A
        label = m.group("label")
        x1, y1, x2, y2 = m.group("x1", "y1", "x2", "y2")
    else:
        # Alt B
        label = m.group("label2")
        x1, y1, x2, y2 = m.group("x1b", "y1b", "x2b", "y2b")

    return {
        "label": _strip_quotes(label),
        "coords": [int(x1), int(y1), int(x2), int(y2)],
    }


def parse_bbox_from_text(text: str) -> List[Dict]:
    """
    Parse bounding boxes from text that contains bbox annotations
    Returns a list of dictionaries with label and coordinates
    """
    matches = find_bbox_matches(text)
    return [parse_bbox_from_match(m) for m in matches]


def extract_last_bbox_from_text(text: str) -> Optional[Dict]:
    """
    Find the most recent bbox mention in the text and return a single bbox dict.
    """
    matches = find_bbox_matches(text)
    if not matches:
        return None
    return parse_bbox_from_match(matches[-1])


def get_contrasting_text_color(background_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Determine the best text color (black or white) based on background color brightness
    """
    r, g, b = background_color
    # Calculate relative luminance
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    # Return white text for dark backgrounds, black text for light backgrounds
    return (255, 255, 255) if luminance < 0.5 else (0, 0, 0)


def draw_bbox_on_image(image: Image.Image, bbox: Dict, color: Tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
    """
    Draw a bounding box with label on the image
    """
    # Validate input
    if image is None:
        return None

    # Create a copy of the image
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # Extract coordinates
    x1, y1, x2, y2 = bbox['coords']
    label = bbox['label']

    # Validate coordinates
    img_width, img_height = img_copy.size

    # Clamp coordinates to image bounds instead of skipping
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(x1 + 1, min(x2, img_width))
    y2 = max(y1 + 1, min(y2, img_height))

    # Draw rectangle with width relative to image size
    rect_width = max(2, int(min(img_width, img_height) * 0.01))  # 1% of smaller dimension, min 2
    draw.rectangle([x1, y1, x2, y2], outline=color, width=rect_width)

    # Calculate font size relative to image dimensions
    img_width, img_height = img_copy.size
    # Use 6% of the smaller dimension as font size, with min 12 and max 120
    font_size = max(12, min(120, int(min(img_width, img_height) * 0.06)))

    # Try to use a font with calculated size, fallback to default if not available
    font = None

    font_path = os.path.join(os.path.dirname(__file__), "PingFang.ttc")
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        pass
    if font is None:
        font = ImageFont.load_default()

    # Calculate label position to ensure it stays within image bounds
    # Estimate label height based on font size
    label_height = int(font_size * 1.2)  # Approximate height for the label

    # Try to place label above the bbox, but if that would go outside, place it below
    if y1 - label_height >= 0:
        # Place above the bbox
        label_y = y1 - label_height
    elif y2 + label_height <= img_height:
        # Place below the bbox
        label_y = y2 + 10
    else:
        # Place inside the bbox (top-left corner)
        label_y = y1 + 10

    # Ensure x position doesn't go outside image bounds
    # Calculate margin based on font size and estimated text width
    text_margin = max(50, int(font_size * 2))  # 2x font size, min 50
    label_x = max(0, min(x1, img_width - text_margin))  # Leave some margin for text width

    # Draw label background
    text_bbox = draw.textbbox((label_x, label_y), label, font=font)
    # Calculate padding relative to font size
    padding = max(2, int(font_size * 0.1))  # 10% of font size, min 2
    # Ensure background rectangle stays within image bounds
    bg_x1 = max(0, text_bbox[0] - padding)
    bg_y1 = max(0, text_bbox[1] - padding)
    bg_x2 = min(img_width, text_bbox[2] + padding)
    bg_y2 = min(img_height, text_bbox[3] + padding)

    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color)

    # Draw label text with contrasting color
    text_color = get_contrasting_text_color(color)
    draw.text((label_x, label_y), label, fill=text_color, font=font)

    return img_copy


def split_answer_with_bboxes(answer: str) -> List[Dict]:
    """
    Split the answer into segments, identifying parts with bboxes
    Returns a list of dictionaries with type ('text' or 'bbox') and content
    """
    segments = []
    
    # Use the original answer text without tag replacements
    # Ensure proper newlines after </think> and <answer> tags
    answer = re.sub(r'</think>(?!\n\n)', '</think>\n\n', answer)
    answer = re.sub(r'<answer>(?!\n\n)', '<answer>\n\n', answer)
    
    # Find all bbox matches and their positions
    bbox_matches = find_bbox_matches(answer)
    
    if not bbox_matches:
        # No bboxes, just return the original text
        segments.append({
            'type': 'text',
            'content': answer
        })
        return segments
    
    # Process text and bbox segments
    # Note: We use the preprocessed version for finding positions but keep original text
    processed_answer = preprocess_text_for_bbox(answer)
    last_end = 0
    
    for match in bbox_matches:
        start, end = match.span()
        
        # Get the bbox text and check for trailing punctuation
        bbox_text = match.group()
        
        # Check if there's text immediately after this bbox that starts with punctuation
        next_text_start = end
        if next_text_start < len(processed_answer):
            # Look ahead to find the next non-whitespace character
            while next_text_start < len(processed_answer) and processed_answer[next_text_start].isspace():
                next_text_start += 1
            
            # If we found a character and it's punctuation, include it with the bbox
            if (next_text_start < len(processed_answer) and 
                processed_answer[next_text_start] in '.,;:!?'):
                # Find the end of this punctuation sequence
                punct_end = next_text_start
                while (punct_end < len(processed_answer) and 
                       processed_answer[punct_end] in '.,;:!?'):
                    punct_end += 1
                
                # Include the punctuation with the bbox
                bbox_text += processed_answer[next_text_start:punct_end]
                end = punct_end
        
        # Add text before bbox as separate text segment
        if start > last_end:
            text_before = processed_answer[last_end:start].strip()
            if text_before:
                segments.append({
                    'type': 'text',
                    'content': text_before
                })
        
        # Add the bbox for image display
        segments.append({
            'type': 'bbox',
            'content': bbox_text
        })
        
        last_end = end
    
    # Add any remaining text after the last bbox
    if last_end < len(processed_answer):
        remaining_text = processed_answer[last_end:].strip()
        if remaining_text:
            segments.append({
                'type': 'text',
                'content': remaining_text
            })
    
    return segments


def is_insignificant_text(text: str) -> bool:
    """
    Check if text between bboxes contains only insignificant content
    (whitespaces, "and", "和", "及", quotations, punctuation)
    """
    if not text:
        return True
    
    # Strip whitespace
    stripped = text.strip()
    if not stripped:
        return True
    
    # Check for common connecting words and punctuation
    insignificant_patterns = [
        r'^\s*and\s*$',  # "and" with optional whitespace
        r'^\s*和\s*$',    # "和" with optional whitespace
        r'^\s*及\s*$',    # "及" with optional whitespace
        r'^\s*[,，]\s*$', # comma (English or Chinese)
        r'^\s*[;；]\s*$', # semicolon (English or Chinese)
        r'^\s*[:：]\s*$', # colon (English or Chinese)
        r'^\s*[.!?。！？]\s*$', # sentence endings
        r'^\s*["""''""]\s*$', # various quotation marks
        r"^\s*['']\s*$", # single quotes
        r'^\s*[()（）]\s*$', # parentheses
        r'^\s*[\[\]【】]\s*$', # square brackets
        r'^\s*[、]\s*$', # Chinese enumeration comma
    ]
    
    for pattern in insignificant_patterns:
        if re.match(pattern, stripped, re.IGNORECASE):
            return True
    
    return False


def display_answer_with_images(answer: str, original_image: Optional[Image.Image], container):
    """
    Display answer with interleaved text and images with bounding boxes
    """
    if not original_image:
        # If no image, just display the text
        container.text(answer)
        return

    # Resize image using qwen_vl_utils
    resized_image = qwen_vl_utils.fetch_image({"image": original_image})

    # Split answer into segments
    segments = split_answer_with_bboxes(answer)

    # Group bbox segments together, considering insignificant text between them
    grouped_segments = []
    current_group = None
    i = 0

    while i < len(segments):
        segment = segments[i]

        if segment['type'] == 'bbox':
            if current_group is None:
                current_group = {'type': 'bbox_group', 'bboxes': []}
            current_group['bboxes'].append(segment['content'])
        else:
            # Check if this text segment is insignificant and if there's a bbox after it
            if (current_group is not None and 
                is_insignificant_text(segment['content']) and 
                i + 1 < len(segments) and 
                segments[i + 1]['type'] == 'bbox'):
                # Skip this insignificant text and continue with the next bbox
                pass
            else:
                # Add current group if it exists
                if current_group:
                    grouped_segments.append(current_group)
                    current_group = None
                # Add this text segment
                grouped_segments.append(segment)

        i += 1

    # Don't forget the last group if it exists
    if current_group:
        grouped_segments.append(current_group)

    # Process each segment
    bbox_group_count = 0
    for segment in grouped_segments:
        if segment['type'] == 'text':
            # Display text segment with better formatting
            text = segment['content']
            # Format numbered lists better - handle both 1. and 1) formats with proper line breaks
            # Only match when followed by a space to avoid decimal numbers like 1.1
            text = re.sub(r'(\d+)[.)]\s', r'\n\n\1. ', text)
            # Clean up multiple consecutive newlines
            text = re.sub(r'\n{3,}', '\n\n', text)
            # Remove leading/trailing whitespace from lines
            lines = text.split('\n')
            lines = [line.strip() for line in lines]
            text = '\n'.join(lines)
            container.markdown(text)

        elif segment['type'] == 'bbox_group':
            bbox_group_count += 1
            # Display the original bbox text first with markdown formatting
            bbox_text_display = " ".join(segment['bboxes'])
            # Apply the same markdown formatting as text segments
            formatted_bbox_text = re.sub(r'(\d+)[.)]', r'\n\n\1.', bbox_text_display)
            # Clean up multiple consecutive newlines
            formatted_bbox_text = re.sub(r'\n{3,}', '\n\n', formatted_bbox_text)
            # Remove leading/trailing whitespace from lines
            lines = formatted_bbox_text.split('\n')
            lines = [line.strip() for line in lines]
            formatted_bbox_text = '\n'.join(lines)
            container.markdown(formatted_bbox_text)

            # Parse all bboxes in this group
            all_bboxes = []
            for bbox_text in segment['bboxes']:
                bboxes = parse_bbox_from_text(bbox_text)
                all_bboxes.extend(bboxes)
            if all_bboxes:
                # Create image with all bboxes from this group
                img_with_bboxes = resized_image.copy()
                # Use different colors for multiple bboxes - high contrast colors for better readability
                colors = [
                    (255, 0, 0),      # Red
                    (0, 0, 255),      # Blue  
                    (255, 165, 0),    # Orange
                    (128, 0, 128),    # Purple
                    (255, 20, 147),   # Deep Pink
                    (0, 100, 0),      # Dark Green
                    (255, 69, 0),     # Red Orange
                    (75, 0, 130),     # Indigo
                ]

                for j, bbox in enumerate(all_bboxes):
                    color = colors[j % len(colors)]
                    img_with_bboxes = draw_bbox_on_image(img_with_bboxes, bbox, color)

                # Display the image with bboxes (controlled size)
                container.image(img_with_bboxes, caption=f"Region of Interest {bbox_group_count}", width=400)


# *************************** Segmentation Detection ***************************
"""
Display utilities for segmentation detection.
"""
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
from io import BytesIO
import json
# import re
# from PIL import Image

def extract_bbox_from_response(response, original_image_size=None, processed_image_size=None):
    """Extract bounding box from model response"""
    predict_bbox = None

    # Method 1: Standard format
    box_pattern = r"<\|box_start\|>\(([\d]+),([\d]+)\),\(([\d]+),([\d]+)\)<\|box_end\|>"
    matches = re.findall(box_pattern, response)
    if matches:
        predict_bbox = []
        for match in matches:
            x1, y1, x2, y2 = int(match[0]), int(match[1]), int(match[2]), int(match[3])
            predict_bbox.append([float(x1), float(y1), float(x2), float(y2)])

    # Method 2: JSON format
    if predict_bbox is None:
        try:
            json_pattern = r'\{[^{}]*?"bbox_2d"\s*:\s*\[[^\[\]]+\][^{}]*?\}'
            json_matches = re.findall(json_pattern, response)
            bboxes = []
            for json_str in json_matches:
                try:
                    bbox_data = json.loads(json_str)
                    if "bbox_2d" in bbox_data:
                        bbox_coords = bbox_data["bbox_2d"]
                        if isinstance(bbox_coords, list) and len(bbox_coords) == 4:
                            bboxes.append(
                                [
                                    float(bbox_coords[0]),
                                    float(bbox_coords[1]),
                                    float(bbox_coords[2]),
                                    float(bbox_coords[3]),
                                ]
                            )
                except Exception:
                    continue
            if bboxes:
                predict_bbox = bboxes
        except Exception as e:
            print(f"Error parsing JSON bbox: {e}")

    # Coordinate conversion if needed
    if predict_bbox is not None and original_image_size is not None and processed_image_size is not None:
        orig_width, orig_height = original_image_size
        proc_width, proc_height = processed_image_size

        scale_x = orig_width / proc_width
        scale_y = orig_height / proc_height

        converted_bbox = []
        for bbox in predict_bbox:
            x1, y1, x2, y2 = bbox
            orig_x1 = int(x1 * scale_x)
            orig_y1 = int(y1 * scale_y)
            orig_x2 = int(x2 * scale_x)
            orig_y2 = int(y2 * scale_y)

            orig_x1 = max(0, min(orig_x1, orig_width))
            orig_y1 = max(0, min(orig_y1, orig_height))
            orig_x2 = max(orig_x1, min(orig_x2, orig_width))
            orig_y2 = max(orig_y1, min(orig_y2, orig_height))

            converted_bbox.append([orig_x1, orig_y1, orig_x2, orig_y2])

        predict_bbox = converted_bbox

    return predict_bbox


def postprocess_segmentation_response(resp, img):
    """Post-process response to extract segmentation masks"""

    def rle_to_mask(rle):
        """Convert RLE to mask"""
        mask = mask_utils.decode(rle)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        return mask

    raw_response = getattr(resp, "raw_response", None)
    if raw_response is None and hasattr(resp, "to_dict"):
        raw_response = resp.to_dict().get("raw_response", None)
    if raw_response is None and isinstance(resp, dict):
        raw_response = resp

    masks = None

    if raw_response is not None and "seg_masks_rle" in raw_response:
        seg_masks_rle = raw_response["seg_masks_rle"]
        if seg_masks_rle and len(seg_masks_rle) > 0:
            masks = [rle_to_mask(rle) for rle in seg_masks_rle]
            return img, masks

    return img, None


def visualize_segmentation(img, masks=None, predict_bbox=None):
    """Visualize segmentation results"""
    orig_W, orig_H = img.size

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.set_title("Segmentation/Detection Result")
    ax.imshow(img)

    # Process masks
    if masks is not None and len(masks) > 0:
        masks_resized = [Image.fromarray(mask).resize((orig_W, orig_H), resample=Image.NEAREST) for mask in masks]
        mask_sum = np.zeros((orig_H, orig_W), dtype=np.uint8)
        for mask in masks_resized:
            mask_array = np.array(mask)
            mask_sum = np.maximum(mask_sum, mask_array)

        if np.any(mask_sum > 0):
            colored_mask = np.zeros((orig_H, orig_W, 4), dtype=np.float32)
            cmap = plt.get_cmap("jet")
            mask_colors = cmap(np.ones(mask_sum.shape))
            colored_mask[mask_sum > 0] = mask_colors[mask_sum > 0]
            colored_mask[..., 3] = mask_sum * 0.5
            ax.imshow(colored_mask)

    # Process bounding boxes
    if predict_bbox is not None:
        for bbox in predict_bbox:
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
            # Add label
            ax.text(x1, y1 - 5, f"Detection", color="red", fontsize=10, weight="bold")

    ax.axis("off")
    plt.tight_layout()

    # Convert to PIL image
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    result_img = Image.open(buf)
    plt.close()

    return result_img

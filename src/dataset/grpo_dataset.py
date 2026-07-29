import copy
import os
import re
import yaml
from typing import Dict, List, Tuple
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from PIL import Image

from src.params import DataArguments
from src.constants import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    SYSTEM_MESSAGE,
)


def parse_image_placeholders(text: str) -> List[Tuple[int, int, str]]:
    """Parse image placeholder tokens and return their positions.
    
    Supports multiple formats:
    - <image> (no number, for single image)
    - <image_1>, <image_2>, ... (underscore format)
    - <image1>, <image2>, ... (no separator)
    - <image 1>, <image 2>, ... (space format)
    
    Returns:
        List of tuples: (start_pos, end_pos, normalized_key)
        normalized_key is like "image", "image_1", "image_2", etc.
        All formats are normalized to underscore format for consistency.
    """
    # Pattern to match various image placeholder formats:
    # <image>, <image_N>, <imageN>, <image N>
    pattern = r'<image(?:[_ ]?(\d+))?>'
    matches = []
    for match in re.finditer(pattern, text):
        number = match.group(1)
        if number:
            # Normalize to underscore format: "image_1", "image_2", etc.
            normalized_key = f"image_{number}"
        else:
            # No number: just "image"
            normalized_key = "image"
        matches.append((match.start(), match.end(), normalized_key))
    return matches


def build_content_list_with_interleaved_images(
    text: str, 
    image_dict: Dict[str, Image.Image]
) -> List[Dict]:
    """Build content list with images interleaved at their placeholder positions.
    
    Args:
        text: Text containing placeholders like <image_1>, <image_2> or <image>
        image_dict: Dict mapping placeholder keys to PIL Images
                   e.g., {"image_1": PIL.Image, "image_2": PIL.Image}
                   or {"image": PIL.Image} for single image
    
    Returns:
        List of content dicts in TRL format:
        [{"type": "image", "image": PIL.Image}, {"type": "text", "text": "..."}, ...]
    """
    placeholders = parse_image_placeholders(text)
    
    if not placeholders:
        # No placeholders found, just return text
        return [{"type": "text", "text": text}]
    
    content_list = []
    last_end = 0
    
    for start, end, key in placeholders:
        # Add text before this placeholder (if any)
        if start > last_end:
            text_before = text[last_end:start].strip()
            if text_before:
                content_list.append({"type": "text", "text": text_before})
        
        # Add the image if it exists in the dict
        if key in image_dict:
            content_list.append({"type": "image", "image": image_dict[key]})
        else:
            # Fallback: try without underscore (e.g., "image1" -> check "image_1")
            print(f"Warning: Image key '{key}' not found in image_dict. Available keys: {list(image_dict.keys())}")
        
        last_end = end
    
    # Add remaining text after last placeholder
    if last_end < len(text):
        text_after = text[last_end:].strip()
        if text_after:
            content_list.append({"type": "text", "text": text_after})
    
    return content_list


def remove_vision_tokens(text):
    """Remove image/video placeholder tokens from text.
    
    Since we're using TRL's conversational format with images embedded in content_list,
    we should not have placeholder tokens in the text itself.
    """
    # Remove LLaVA format: <image> or <video>
    text = re.sub(r'\n?<image>\n?', '', text)
    text = re.sub(r'\n?<video>\n?', '', text)
    
    # Clean up any extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


class GRPODataset(Dataset):
    """Dataset for DPO training"""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(GRPODataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding

        if "Qwen3" in self.model_id:
            self.image_patch_size = 16
        else:
            self.image_patch_size = 14
        
        self.question_template = self._get_prompt_template(data_path)

    def _get_prompt_template(self, data_path: str) -> str:
        with open(self.data_args.prompt_path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        for key in prompts:
            if key in data_path:
                print(f"{'='*30}\n{prompts[key]}\n{'='*30}")
                return prompts[key].replace('\n', ' ')
        
        raise ValueError(f"No matching prompt template found for data_path: {data_path}!")

    def __len__(self):
        return len(self.list_data_dict)
    
    def _load_images(self, sources) -> Tuple[List[Image.Image], Dict[str, Image.Image]]:
        """Load images from sources.
        
        Supports two formats:
        1. Simple format: "image": "path.png" or "image": ["path1.png", "path2.png"]
        2. Dict format: "image": {"image_1": "path1.png", "image_2": "path2.png", ...}
        
        Returns:
            Tuple of:
            - pil_images: List of PIL Images (for TRL's "images" field)
            - image_dict: Dict mapping keys to PIL Images (for interleaved insertion)
              Keys are normalized: "image" for single, "image_1", "image_2" for multiple
        """
        pil_images = []
        image_dict = {}
        
        if "image" not in sources:
            return pil_images, image_dict
        
        image_files = sources["image"]
        image_folder = self.data_args.image_folder
        
        if isinstance(image_files, dict):
            # Dict format: {"image_1": "path1.png", "image_2": "path2.png", ...}
            for key, image_file in image_files.items():
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                try:
                    pil_img = Image.open(image_file).convert("RGB")
                    pil_images.append(pil_img)
                    # Normalize key: ensure underscore format (image_1, image_2, etc.)
                    normalized_key = key.replace(" ", "_")
                    image_dict[normalized_key] = pil_img
                except Exception as e:
                    print(f"Warning: Failed to load image {image_file}: {e}")
                    continue
                    
        elif isinstance(image_files, str):
            # Single image: "image": "path.png"
            image_file = image_files
            if not os.path.exists(image_file):
                if not image_file.startswith("http"):
                    image_file = os.path.join(image_folder, image_file)
            try:
                pil_img = Image.open(image_file).convert("RGB")
                pil_images.append(pil_img)
                # Use "image" as key for single image (matches <image> placeholder)
                image_dict["image"] = pil_img
            except Exception as e:
                print(f"Warning: Failed to load image {image_file}: {e}")
                
        elif isinstance(image_files, list):
            # List format: "image": ["path1.png", "path2.png"]
            for idx, image_file in enumerate(image_files):
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                try:
                    pil_img = Image.open(image_file).convert("RGB")
                    pil_images.append(pil_img)
                    # Use "image_1", "image_2", etc. for list format
                    image_dict[f"image_{idx + 1}"] = pil_img
                except Exception as e:
                    print(f"Warning: Failed to load image {image_file}: {e}")
                    continue
        
        return pil_images, image_dict
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        # Load images - supports string, list, and dict formats
        pil_images, image_dict = self._load_images(sources)

        conversations = sources['conversations']
        user_input = conversations[0]
        gpt_response = conversations[1]
        
        # Get raw content
        user_content = user_input['value']
        user_content_with_template = self.question_template.format(question=user_content)
        
        # Check if there are any image placeholders in the text
        placeholders = parse_image_placeholders(user_content_with_template)
        
        if pil_images and placeholders:
            # Has images AND has placeholders -> insert images at placeholder positions
            content_list = build_content_list_with_interleaved_images(
                user_content_with_template, 
                image_dict
            )
        elif pil_images:
            # Has images but NO placeholders -> put all images before text
            user_content_clean = remove_vision_tokens(user_content)
            user_content_with_template = self.question_template.format(question=user_content_clean)
            
            content_list = []
            # Add all images first (before text)
            for pil_img in pil_images:
                content_list.append({"type": "image", "image": pil_img})
            # Add text content
            content_list.append({"type": "text", "text": user_content_with_template})
        else:
            # No images -> just text
            user_content_clean = remove_vision_tokens(user_content)
            user_content_with_template = self.question_template.format(question=user_content_clean)
            content_list = [{"type": "text", "text": user_content_with_template}]
        
        # Return conversational format expected by TRL GRPOTrainer
        # prompt should be a list of message dicts with content as list
        prompt = [{"role": "user", "content": content_list}]
        
        assistant_prompt = gpt_response['value']

        # Build data_dict with required fields for TRL GRPOTrainer VLM support
        # CRITICAL: TRL checks for "images" or "image" field to detect VLM data
        # See: trl/trainer/grpo_trainer.py#L1857-L1873
        data_dict = dict(
            prompt=prompt,
            assistant=assistant_prompt,
        )
        
        # Pass tolerance for FinMME numerical questions (always include to keep keys consistent across samples)
        data_dict['tolerance'] = sources.get('tolerance', None)

        # Pass Instruction Following fields
        data_dict['instruction_id_list'] = sources.get('instruction_id_list', None)
        data_dict['if_kwargs'] = sources.get('if_kwargs', None)
        
        # Add images field for TRL VLM detection
        # TRL expects: "images" (list of PIL Images) or "image" (single PIL Image)
        if pil_images:
            data_dict["images"] = pil_images

        return data_dict
    
def make_grpo_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    grpo_dataset = GRPODataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )

    return dict(train_dataset=grpo_dataset,
                eval_dataset=None)
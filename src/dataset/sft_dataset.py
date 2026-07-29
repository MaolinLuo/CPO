import os
from typing import Dict
import torch
import transformers
import ujson as json
import yaml
import re
from PIL import Image
from typing import List, Tuple
from pathlib import Path
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

from src.params import DataArguments
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import get_image_info, get_video_info, llava_to_openai, pad_sequence


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
    image_dict: Dict[str, Image.Image],
    min_pixel=None,
    max_pixel=None
) -> List[Dict]:
    """Build content list with images interleaved at their placeholder positions.
    
    Args:
        text: Text containing placeholders like <image_1>, <image_2> or <image>
        image_dict: Dict mapping placeholder keys to image_file_path
                   e.g., {"image_1": image_file_path1, "image_2": image_file_path2}
                   or {"image": image_file_path} for single image
    
    Returns:
        List of content dicts in TRL format:
        [{"type": "image", "image": image_file_path}, {"type": "text", "text": "..."}, ...]
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
            content_list.append({"type": "image", "image": image_dict[key],"min_pixels": min_pixel,"max_pixels": max_pixel})
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

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.nframes = data_args.nframes
        self.question_template = self._get_prompt_template(data_path)

        if "Qwen3" in self.model_id:
            self.image_patch_size = 16
            self.return_video_metadata = True
        else:
            self.image_patch_size = 14
            self.return_video_metadata = False

    def __len__(self):
        return len(self.list_data_dict)

    def _get_prompt_template(self, data_path: str) -> str:
        with open(self.data_args.prompt_path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        for key in prompts:
            if key in data_path:
                print(f"{'='*30}\n{prompts[key]}\n{'='*30}")
                return prompts[key].replace('\n', ' ')
        
        raise ValueError(f"No matching prompt template found for data_path: {data_path}!")
            
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        processor = self.processor
        if "image" in sources:
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            image_files_init = sources["image"]
            image_files = {}
            image_folder = self.data_args.image_folder

            if isinstance(image_files_init, dict):
                # Dict format: {"image_1": "path1.png", "image_2": "path2.png", ...}
                for key, image_file in image_files_init.items():
                    key = key.replace(" ", "_")
                    full_image_path = os.path.join(image_folder, image_file)
                    image_files[key] = full_image_path
            elif isinstance(image_files_init, str):
                full_image_path = os.path.join(image_folder, image_files_init)
                image_files["image"] = full_image_path
            elif isinstance(image_files_init, list):
                # List format: "image": ["path1.png", "path2.png"]
                for idx, image_file in enumerate(image_files_init):
                    image_files[f"image_{idx + 1}"] = os.path.join(image_folder, image_file)


        else:
            grid_key = None
            pixel_key = None

        conversations = sources['conversations']
        user_input = conversations[0]
        gpt_response = conversations[1]
        
        # Get raw content
        user_content = user_input['value']
        user_content_with_template = self.question_template.format(question=user_content)
        
        # Check if there are any image placeholders in the text
        placeholders = parse_image_placeholders(user_content_with_template)
        
        if image_files and placeholders:
            # Has images AND has placeholders -> insert images at placeholder positions
            content_list = build_content_list_with_interleaved_images(
                user_content_with_template, 
                image_files,
                min_pixel=self.image_min_pixel,
                max_pixel=self.image_max_pixel
            )
        elif image_files:
            # Has images but NO placeholders -> put all images before text
            user_content_clean = remove_vision_tokens(user_content)
            user_content_with_template = self.question_template.format(question=user_content_clean)
            
            content_list = []
            # Add all images first (before text)
            for image_file in image_files:
                content_list.append({"type": "image", "image": image_file, "min_pixels": self.image_min_pixel, "max_pixels": self.image_max_pixel})
            # Add text content
            content_list.append({"type": "text", "text": user_content_with_template})
        else:
            # No images -> just text
            user_content_clean = remove_vision_tokens(user_content)
            user_content_with_template = self.question_template.format(question=user_content_clean)
            content_list = [{"type": "text", "text": user_content_with_template}]
        
        
        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        # Return conversational format expected by TRL GRPOTrainer
        # prompt should be a list of message dicts with content as list
        prompt = [{"role": "user", "content": content_list}]
        assistant_prompt = [{"role": "assistant", "content": gpt_response['value']}]
        inputs = processor.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(prompt)
        inputs = processor(
            text=[inputs],
            images=image_inputs if image_inputs else None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        prompt_input_ids = inputs['input_ids']
        response_input = processor.apply_chat_template(
            assistant_prompt, tokenize=False, add_generation_prompt=False
        )
        response_input_ids = processor(
            text=[response_input],
            images=image_inputs if image_inputs else None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )['input_ids']
        all_pixel_values.append(inputs[pixel_key])
        all_image_grid_thw.append(inputs[grid_key])

        response_labels = response_input_ids.squeeze(0).clone()
        input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
        labels = torch.cat(
            [
                torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  # Ignore prompt tokens
                response_labels,
            ],
            dim=0,
        )

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []

        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict

def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    eval_dataset = None
    if data_args.eval_path is not None:
        eval_dataset = SupervisedDataset(
              data_path=data_args.eval_path,
              processor=processor,
              data_args=data_args,
              model_id=model_id
          )
        
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)

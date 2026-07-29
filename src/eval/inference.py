import os
import io
import json
from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from vllm import LLM, SamplingParams
import argparse
import re
from qwen_vl_utils import process_vision_info
import ast
from PIL import Image
import yaml
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset
from PIL import Image
from PIL.WebPImagePlugin import WebPImageFile


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = " ".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def process_image(image_bytes):
    try:
        image_stream = io.BytesIO(image_bytes)
        image = Image.open(image_stream)
        
        # Convert non-WebP inputs to WebP.
        if image.format != "WEBP":
            webp_stream = io.BytesIO()
            image.save(webp_stream, format="WEBP")
            webp_stream.seek(0)
            image = Image.open(webp_stream)
        
        assert isinstance(image, WebPImageFile), f"Unexpected image type: {type(image)}"
        return image
    except Exception as e:
        print(f"Failed to convert image to WebP: {e}")
        return None

def extract_last_boxed(text: str) -> str:
    """
    Helper function to extract the content of the last \boxed{...}.
    It handles nested braces correctly.
    """
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return ""
    
    content = ""
    open_braces = 0
    for char in text[idx + 7:]:
        if char == '{':
            open_braces += 1
            content += char
        elif char == '}':
            if open_braces == 0:
                return content
            open_braces -= 1
            content += char
        else:
            content += char
    return content


class VLMDataset(Dataset):
    """Dataset used for VLM inference."""
    def __init__(self, data: List[Dict], media_dir: str = None):
        self.data = data
        self.media_dir = media_dir
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        if "POPE" in self.media_dir:
            question_id = item['question_id']
            image_path = item['image']
            question = item['text']
            ground_truth = item['label']
            
            # Resolve image paths.
            if self.media_dir and not os.path.isabs(image_path):
                full_image_path = os.path.join(self.media_dir, image_path)
            else:
                full_image_path = image_path
            
            try:
                image = Image.open(full_image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {full_image_path}: {e}")
                raise e
                
            return {
                'question_id': question_id,
                'question': question,
                'ground_truth': ground_truth,
                'image': image
            }
            
        elif "InstructFollow" in self.media_dir:
            question_id = item.get('id', 0)
            question = item['conversations'][0]['value']
            return {
                'question_id': question_id,
                'question': question,
                'ground_truth': '',
                'instruction_id_list': item.get('instruction_id_list', []),
                'if_kwargs': item.get('if_kwargs', []),
            }

        elif "We-Math2" in self.media_dir or "Navigation" in self.media_dir or "CVQA" in self.media_dir or "MedBookVQA" in self.media_dir or "FinMME" in self.media_dir or "Puzzle" in self.media_dir:
            question_id = item['id']
            image_path = item['image']
            question = item['conversations'][0]['value']
            ground_truth = item['conversations'][1]['value']
            
            # Resolve image paths.
            if self.media_dir and not os.path.isabs(image_path):
                full_image_path = os.path.join(self.media_dir, image_path)
            else:
                full_image_path = image_path
            
            # Preload the image.
            try:
                image = Image.open(full_image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {full_image_path}: {e}")
                raise e

            return {
                'question_id': question_id,
                'question': question,
                'ground_truth': ground_truth,
                'image': image,
                **({'tolerance': item['tolerance']} if 'tolerance' in item else {})
            }

        elif "Chemistry" in self.media_dir or "Coding" in self.media_dir:
            question_id = item['id']
            image_paths = item['image']
            question = item['conversations'][0]['value']
            ground_truth = item['conversations'][1]['value']
            
            images = {}
            for k, imp in image_paths.items():
                if self.media_dir and not os.path.isabs(imp):
                    fimp = os.path.join(self.media_dir, imp)
                else:
                    fimp = imp
                try:
                    images[k] = Image.open(fimp).convert('RGB')
                except Exception as e:
                    print(f"Error loading image {fimp}: {e}")
                    raise e
            return {
                'question_id': question_id,
                'question': question,
                'ground_truth': ground_truth,
                'image': images
            }
            
        else:
            raise NotImplementedError(f"Not support {self.media_dir}!")

class VLMInference:
    def __init__(
        self,
        base_model_path: str,
        test_file: str,
        output_dir: str,
        media_dir: str = None,
        tensor_parallel_size: int = 1,
        device: str = "cuda",
        disable_flash_attn2: bool = False,
        batch_size: int = 1,
        prompt_config: Dict = None,
        max_completion_length: int = 2048
    ):
        self.base_model_path = base_model_path
        self.test_file = test_file
        self.output_dir = output_dir
        self.media_dir = media_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.device = device
        self.disable_flash_attn2 = disable_flash_attn2
        self.batch_size = batch_size
        self.prompt_config = prompt_config
        self.max_completion_length=max_completion_length
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure vLLM.
        llm_kwargs = {
            "model": base_model_path,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.9,
            "tensor_parallel_size": tensor_parallel_size,
            "limit_mm_per_prompt": {"image": 50},
        }
        
        self.llm = LLM(**llm_kwargs)
        self.chat_tokenizer = self._init_chat_tokenizer(base_model_path)
        
        # Configure sampling.
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_completion_length,
            repetition_penalty = 1.05
        )
    
    def _init_chat_tokenizer(self, base_model_path):
        tokenizer = None
        get_tokenizer = getattr(self.llm, "get_tokenizer", None)
        if callable(get_tokenizer):
            try:
                tokenizer = get_tokenizer()
            except Exception:
                tokenizer = None
        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            if hasattr(tokenizer, "apply_chat_template"):
                return tokenizer
        except Exception:
            pass
        return None

    def load_test_data_json(self) -> List[Dict]:
        """Load test data."""
        if "POPE" in self.test_file:
            test_data = []
            # Load JSON Lines records one at a time.
            with open(self.test_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        # Parse and append one JSON object.
                        data_item = json.loads(line)
                        test_data.append(data_item)
                    except json.JSONDecodeError as e:
                        # Report malformed records and continue with the next line.
                        print(
                            f"Warning: failed to parse {self.test_file} at line "
                            f"{line_num}; skipping the record. Details: {e}"
                        )
                        continue
        else:
            with open(self.test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        return test_data
    
    def load_test_data_parquet(self) -> List[Dict]:
        if "DocVQA" in self.test_file or "MMMU_eval" in self.test_file or "Charxiv" in self.test_file:
            test_data = load_dataset("parquet", data_dir=self.test_file, split="validation")
        else:
            test_data = load_dataset("parquet", data_dir=self.test_file, split="test")
        if "Charxiv" in self.test_file:
        # Define five question-answer pairs.
            qa_pairs = [
                ("descriptive_q1", "descriptive_a1"),
                ("descriptive_q2", "descriptive_a2"),
                ("descriptive_q3", "descriptive_a3"),
                ("descriptive_q4", "descriptive_a4"),
                ("reasoning_q", "reasoning_a")
            ]
            processed_data = []
            
            for raw_item in test_data:
                raw_dict = dict(raw_item)
                for q_col, a_col in qa_pairs:
                    new_item = raw_dict.copy()
                    q_value = new_item[q_col]
                    a_value = new_item[a_col]
                    for del_q, del_a in qa_pairs:
                        new_item.pop(del_q, None)
                        new_item.pop(del_a, None)

                    if q_col.startswith("descriptive_q"):
                        new_item["descriptive_q"] = q_value
                        new_item["ground_truth"] = a_value
                        new_item["reasoning_q"] = None
                        new_item["Type"] = "DQ"
                    else:
                        new_item["reasoning_q"] = q_value
                        new_item["ground_truth"] = a_value
                        new_item["descriptive_q"] = None
                        new_item["Type"] = "RQ"

                    processed_data.append(new_item)
            test_data = HFDataset.from_list(processed_data)
        return test_data

    def _build_prompt(self, item):
        test_file = self.test_file
        prompt_config = self.prompt_config
        
        images = []
        
        if "POPE" in test_file:
            prompt = prompt_config["POPE"].replace("{question}", item["question"]).replace('\n', ' ')
            images = [item["image"]]
        
        elif "InstructFollow" in test_file:
            prompt = item["question"]
            # No images for InstructFollow

        elif "We-Math2" in test_file or "Navigation" in test_file or "CVQA" in test_file or "MedBookVQA" in test_file or "FinMME" in test_file or "Puzzle" in test_file:
            if "We-Math2" in test_file:
                prompt = prompt_config["Math"].replace("{question}", item["question"]).replace('\n', ' ')
            elif "Navigation" in test_file:
                prompt = prompt_config["Navigation"].replace("{question}", item["question"]).replace('\n', ' ')
            elif "CVQA" in test_file:
                prompt = prompt_config["CVQA"].replace("{question}", item["question"]).replace('\n', ' ')
            elif "MedBookVQA" in test_file:
                prompt = prompt_config["MedBookVQA"].replace("{question}", item["question"]).replace('\n', ' ')
            elif "FinMME" in test_file:
                prompt = prompt_config["FinMME"].replace("{question}", item["question"]).replace('\n', ' ')
            elif "Puzzle" in test_file:
                prompt = prompt_config["Puzzle"].replace("{question}", item["question"]).replace('\n', ' ')
            images = [item["image"]]
        
        elif "Chemistry" in test_file or "Coding" in test_file:
            if "Chemistry" in test_file:
                prompt = prompt_config["Chemistry"].replace("{question}", item["question"]).replace('\n', ' ')
            elif "Coding" in test_file:
                prompt = prompt_config["Coding"].replace("{question}", item["question"]).replace('\n', ' ')
            images = []
            image_tags = re.findall(r"<image(?:[\s_]+\d+)?>", prompt)
            for tag in image_tags:
                key = tag.strip("<>")
                if key in item["image"]:
                    images.append(item["image"][key])

        elif "standard" in test_file:
            question = item["question"]
            parsed_options = parse_options(ast.literal_eval(str(item["options"])))
            prompt = f"{question} {parsed_options} {prompt_config['standard']}".replace('\n', ' ')
            image_order = [int(num) for num in re.findall(r"<image\s+(\d+)>", prompt)]
            for idx in image_order:
                images.append(item[f"image_{idx}"])

        elif "vision" in test_file:
            prompt = prompt_config["vision"].replace('\n', ' ')
            images = [item["image"]]

        elif "MathVision" in test_file:
            question = item["question"]
            parsed_options = parse_options(ast.literal_eval(str(item["options"])))
            question = f"{question}\t{parsed_options}"
            prompt = prompt_config["Math"].replace("{question}", question).replace('\n', ' ')
            images = []
            image = item["decoded_image"]['bytes']
            image = process_image(image)
            images.append(image)

        elif "MathVista" in test_file:
            question = item["query"]
            prompt = prompt_config["Math2"]
            prompt = prompt.replace("{question}", question).replace('\n', ' ')
            images = []
            image = item["decoded_image"]
            images.append(image)

        elif "RealworldQA" in test_file or "MMStar" in test_file or "CountBenchQA" in test_file or "DocVQA" in test_file or "OCRBenchv2" in test_file :
            if "CountBenchQA" in test_file:
                prompt = prompt_config["CountBenchQA"]
            elif "RealworldQA" in test_file:
                prompt = prompt_config["RealworldQA"]
            else:
                prompt = prompt_config["VQA"]
            question = item["question"]
            prompt = prompt.replace("{question}", question).replace('\n', ' ')
            images = []
            image = item["image"]
            if "MMStar" in test_file:
                image = process_image(image)
            images.append(image)
            
        elif "Charxiv" in test_file: 
            num_subplots = item.get("num_subplots", 1)
            if item["Type"] == "DQ":
                q_id = item["descriptive_q"]-1
                prompt = prompt_config["CharxivDQ"][q_id]
                if num_subplots == 1:
                    location = "For the current plot"
                elif num_subplots > 1:
                    if item.get("subplot_row") and item.get("subplot_col"):
                        location = f"For the subplot at row {item.get('subplot_row', 1)} and column {item.get('subplot_col', 1)}"
                    else:
                        location = f"For {item.get('subplot_loc')}"
                prompt = prompt.replace("{location}", location).replace('\n', ' ')
            elif item["Type"] == "RQ":
                q_id = item["reasoning_a_type"]-1
                ans = item["ground_truth"]
                if q_id == 3:
                    integer_pattern = r'^[-+]?\d+$'
                    decimal_pattern = r'^[-+]?\d+\.\d+$'
                    if re.fullmatch(integer_pattern, ans):
                        prompt = prompt_config["CharxivRQ"][3]
                    elif re.fullmatch(decimal_pattern, ans):
                        # Count digits after the decimal point.
                        num_decimal = len(ans.split('.')[1])
                        prompt = prompt_config["CharxivRQ"][4].replace("{num_decimal}", str(num_decimal))
                else:
                    prompt = prompt_config["CharxivRQ"][q_id]
                question = item["reasoning_q"]
                prompt = prompt.replace("{question}", question).replace('\n', ' ')
            images = []
            image = item["image"]
            images.append(image)

        elif "MathVerse" in test_file:
            question = item["query_cot"]
            prompt = prompt_config["Math2"]
            prompt = prompt.replace("{question}", question).replace('\n', ' ')
            images = []
            image = item["image"]['bytes']
            image = process_image(image)
            images.append(image)
        
        else:
            raise NotImplementedError(f"Test file {test_file} not supported for prompt building.")

        min_pixels = 8 * 32 * 32
        max_pixels = 512 * 32 * 32
        
        # Check for image tags using regex to handle interleaving
        split_pattern = r"(<image(?:[\s_]+\d+)?>)"
        content = []
        
        if re.search(split_pattern, prompt) and len(images) > 0:
            parts = re.split(split_pattern, prompt)
            image_idx = 0
            for part in parts:
                if not part:
                   continue
                if re.match(split_pattern, part):
                    # It's an image placeholder
                    if image_idx < len(images):
                        content.append({"type": "image", "image": images[image_idx], "min_pixels": min_pixels, "max_pixels": max_pixels})
                        image_idx += 1
                else:
                    # It's text
                    content.append({"type": "text", "text": part})
        else:
            # Default: Images first, then Text
            for image in images:
                content.append({"type": "image", "image": image, "min_pixels": min_pixels, "max_pixels": max_pixels})
            content.append({"type": "text", "text": prompt})

        conversation = [{"role": "user", "content": content}]

        if self.chat_tokenizer:
            try:
                final_prompt = self.chat_tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            except Exception:
                raise RuntimeError("Failed to apply chat template using the tokenizer.")
        else:
             raise RuntimeError("Tokenizer not initialized.")

        image_inputs, _ = process_vision_info(conversation)
        return final_prompt, image_inputs


    def predict(self, batch: List[Dict]) -> List[tuple]:
        """Run batched prediction."""
        
        # Build vLLM inputs.
        inputs = []
        for item in batch:
            try:
                prompt, image_inputs = self._build_prompt(item)
                input_item = {"prompt": prompt}
                if image_inputs:
                    input_item["multi_modal_data"] = {"image": image_inputs}
                inputs.append(input_item)
            except Exception as e:
                print(f"Error building prompt for item: {e}")
                raise e

        # Generate the batch with vLLM.
        outputs = self.llm.generate(inputs, self.sampling_params)
        
        return [(output.outputs[0].text.strip(), output.outputs[0].finish_reason) for output in outputs]

    
    def run_inference_on_json(self) -> List[Dict]:
        """Run inference over a JSON test set."""
        test_data = self.load_test_data_json()
        dataset = VLMDataset(test_data, self.media_dir)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, collate_fn=lambda x: x)
        
        results = []
        for batch in tqdm(dataloader, desc="Inference"):
            predictions = self.predict(batch)
            
            for item, (prediction, finish_reason) in zip(batch, predictions):
                if "InstructFollow" in self.test_file:
                    # IF: keep full response, pass through instruction metadata
                    results.append({
                        'model_pred': prediction,
                        'model_answer': prediction,
                        'question_id': item['question_id'],
                        'question': item['question'],
                        'ground_truth': '',
                        'instruction_id_list': item.get('instruction_id_list', []),
                        'if_kwargs': item.get('if_kwargs', []),
                        'finish_reason': finish_reason,
                    })
                else:
                    boxed_ans = extract_last_boxed(prediction)
                    if boxed_ans:
                        final_answer = boxed_ans.strip()
                    else:
                        final_answer = prediction

                    option_match = re.search(r'^([^:]+):.*', final_answer, re.DOTALL)
                    if option_match:
                        # Extract and trim the option before the colon.
                        option_str = option_match.group(1).strip()
                        if len(option_str) == 1 and option_str.isalpha():
                            final_answer = option_str
                    results.append({
                        'model_pred': prediction,
                        'model_answer': final_answer,
                        'question_id': item['question_id'],
                        'question': item['question'],
                        'ground_truth': item['ground_truth'],
                        **({'tolerance': item['tolerance']} if 'tolerance' in item else {}),
                        'finish_reason': finish_reason
                    })
        
        self.save_results(results)
        return results
    
    def run_inference_on_parquet(self) -> List[Dict]:
        results = []
        dataset = self.load_test_data_parquet()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, collate_fn=lambda x: x)
        for batch in tqdm(dataloader, desc="Inference"):
            predictions = self.predict(batch)

            for item, (prediction, finish_reason) in zip(batch, predictions):
                boxed_ans = extract_last_boxed(prediction)
                if boxed_ans:
                    final_answer = boxed_ans.strip()
                else:
                    final_answer = prediction

                option_match = re.search(r'^([^:]+):.*', final_answer, re.DOTALL)
                if option_match:
                    # Extract and trim the option before the colon.
                    option_str = option_match.group(1).strip()
                    if len(option_str) == 1 and option_str.isalpha():
                        final_answer = option_str
                result_sample = {"model_pred": prediction, "model_answer": final_answer, "finish_reason": finish_reason, **item}
                result_sample = {k: v for k, v in result_sample.items() if not k.startswith("image") and not k.startswith("decoded_image")}
                results.append(result_sample)

        self.save_results(results)
        return results
    
    def run_inference_on_MMMU_pro(self) -> List[Dict]:
        results = []
        dataset = self.load_test_data_parquet()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, collate_fn=lambda x: x)
        for batch in tqdm(dataloader, desc="Inference"):
            predictions = self.predict(batch)

            for item, (prediction, finish_reason) in zip(batch, predictions):
                result_sample = {"response": prediction, "finish_reason": finish_reason, **item}
                result_sample = {k: v for k, v in result_sample.items() if not k.startswith("image")}
                results.append(result_sample)

        self.save_results(results)
        return results

    def save_results(self, results: List[Dict]):
        """Save inference results."""
        results_file = os.path.join(self.output_dir, 'merge.jsonl')
        with open(results_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='VLM Inference Script with vLLM')
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--media_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--prompts_file', type=str, required=True)
    parser.add_argument('--max_completion_length', type=int, required=True)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--disable_flash_attn2', action='store_true', help='Disable Flash Attention 2 and use SDPA instead')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    
    args = parser.parse_args()
    with open(args.prompts_file, "r") as file:
        prompt_config = yaml.safe_load(file)
    print(f"Prompt configuration loaded:\n{prompt_config}")
    inferencer = VLMInference(
        base_model_path=args.base_model,
        test_file=args.test_file,
        output_dir=args.output_dir,
        media_dir=args.media_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        device=args.device,
        disable_flash_attn2=args.disable_flash_attn2,
        batch_size=args.batch_size,
        prompt_config=prompt_config,
        max_completion_length=args.max_completion_length
    )

    datasets_parquet = [
        "RealworldQA", "MMStar", "CountBenchQA", "DocVQA", 
        "OCRBenchv2", "Charxiv", "MathVerse", "MathVision", "MathVista"
        ]
    if any(dataset in args.test_file for dataset in datasets_parquet):
        inferencer.run_inference_on_parquet()
    elif "MMMU" in args.test_file:
        inferencer.run_inference_on_MMMU_pro()
    else:
        inferencer.run_inference_on_json()


if __name__ == "__main__":
    main()

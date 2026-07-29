import json
import argparse
import re
import os
import ast
import numpy as np
from typing import List, Dict
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig

def mmmu_process_results(results):
    pred = results['response']
    if isinstance(pred, dict):
        pred = ''
    index2ans, all_choices = get_multi_choice_info(ast.literal_eval(str(results["options"])))
    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    if parsed_pred == results["answer"]:
        if_right = True
    else:
        if_right = False

    results['pred_indexs'] = parsed_pred
    results['if_right'] = if_right
    
    return results


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index, e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        # Extract the string after "Answer:"
        answer_str = response[last_answer_pos + len("Answer:"):].strip()
        
        # Find a unique match in the options
        matching_options = [option for option in all_choices if option in answer_str]
        
        # If a unique match is found, return that option
        if len(matching_options) == 1:
            return matching_options[0]

        
    if isinstance(response, str):
        for char in [",", ".", "!", "?", ";", ":", "'"]:
            response = response.strip(char)
        response = " " + response + " "  # add space to avoid partial match
    else:
        print (response)
        response = ""
    

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        #pred_index = random.choice(all_choices)
        pred_index = ""  # empty prediction
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index

def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices

class ExactMatchEvaluator:
    """Evaluate merged inference results with exact-match rules."""
    
    def __init__(self, dataset_name: str, merged_file: str, output_dir: str):
        self.dataset_name = dataset_name
        self.merged_file = merged_file
        self.output_dir = output_dir
    
    def load_results(self) -> List[Dict]:
        """Load merged inference results."""
        results = []
        with open(self.merged_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results
    
    def clean_expression(self, ground_truth: str, prediction: str):
        clean_pattern = r"^(\s*(?:\$\$?)?)\s*(?:f\s*\(\s*x\s*\)|y|f\s*\\left\(\s*x\s*\\right\))\s*=\s*(?![^=]*=)"
        # 1. ^(\s*(?:\$\$?)?) captures leading whitespace and one or two dollar signs.
        # 2. (?:f\s*\(\s*x\s*\)|y) matches f(x) or y with optional inner whitespace.
        # 3. \s*=\s* matches the equality sign and surrounding whitespace.
        # 4. (?![^=]*=) ensures that no second equality sign follows.
        ground_truth = re.sub(clean_pattern, r"\1", ground_truth, flags=re.IGNORECASE)
        prediction = re.sub(clean_pattern, r"\1", prediction, flags=re.IGNORECASE)

        # Roman numeral replacement for quadrants
        roman_map = {"I": "First", "II": "Second", "III": "Third", "IV": "Fourth"}
        if ground_truth.strip() in roman_map:
            ground_truth = roman_map[ground_truth.strip()]
        if prediction.strip() in roman_map:
            prediction = roman_map[prediction.strip()]

        return ground_truth, prediction
    
    def check_single_match(self, ground_truth: str, prediction: str, tolerance=None) -> bool:
        """Check whether one predicted expression matches the reference."""
        ground_truth, prediction = self.clean_expression(ground_truth, prediction)
        content_to_eval = prediction
        score = False

        # 1. Try math_verify
        try:
            gold_parsed = parse(ground_truth, extraction_mode="first_match")
            if len(gold_parsed) != 0:
                answer_parsed = parse(
                    f"${content_to_eval}$",
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=True,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed=False,
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=True,
                        )
                    ],
                    extraction_mode="first_match",
                )
                if verify(gold_parsed, answer_parsed):
                    score = True
        except Exception:
            pass

        # 2. Fallback to custom string matching if math_verify failed
        if not score:
            pred: str = content_to_eval.lower()
            gt: str = ground_truth.lower()
            ignore_chars = {'\\', ',', ';',' ', '{', '}', '$'}
            for char in ignore_chars:
                pred = pred.replace(char, '')
            for char in ignore_chars:
                gt = gt.replace(char, '')
            
            if gt == pred:
                score = True
            elif tolerance is not None:
                # Match numerical values within the configured tolerance.
                try:
                    pred_value = float(pred)
                    gt_value = float(gt)
                    if abs(pred_value - gt_value) <= tolerance:
                        score = True
                except (ValueError, TypeError):
                    pass
        
        return score
    
    def evaluate(self) -> Dict:
        """Evaluate merged results."""
        results = self.load_results()
        total = len(results)
        correct = 0
        finish_reasons = {}

        for result in results:
            if "finish_reason" in result:
                finish_reasons[result["finish_reason"]] = finish_reasons.get(result["finish_reason"], 0) + 1
            if "CountBenchQA" in self.dataset_name:
                ground_truth = result['number']
            elif "MathVerse" in self.dataset_name or "RealworldQA" in self.dataset_name or "MMStar" in self.dataset_name or "MathVision" in self.dataset_name:
                ground_truth = result['answer']
            elif "DocVQA" in self.dataset_name:
                ground_truth = result['answers']
            elif "MathVista" in self.dataset_name:
                raw_answer = result['answer']
                choices = result.get("choices", [])  
                if choices: 
                    try:
                        answer_index = choices.index(raw_answer)
                        ground_truth = chr(ord('A') + answer_index)
                    except ValueError:
                        ground_truth = raw_answer
                    result['answer']= ground_truth  
                else:
                    ground_truth = raw_answer
                    result['answer']= ground_truth 
            else:
                ground_truth = result['ground_truth']
            prediction = result['model_answer']
            
            score = 0.0
            if isinstance(ground_truth, list):
                # Try each accepted reference answer.
                for gt_item in ground_truth:
                    # Split by semicolon and ensure parts are clean
                    if isinstance(gt_item, str) and isinstance(prediction, str):
                        gt_parts = [p.strip() for p in gt_item.split(';') if p.strip()]
                        pred_parts = [p.strip() for p in prediction.split(';') if p.strip()]
                    else:
                        gt_parts = [str(ground_truth)]
                        pred_parts = [str(prediction)]
                    # Calculate score based on GT parts order
                    num_gt = len(gt_parts)
                    
                    for i, gt_part in enumerate(gt_parts):
                        if i < len(pred_parts) and self.check_single_match(gt_part, pred_parts[i]):
                            score += 1.0 / num_gt
                    
                    if score > 0:
                        correct += score
                        break  # Stop after the first matching reference.
            else:
                # Split by semicolon and ensure parts are clean
                if isinstance(ground_truth, str) and isinstance(prediction, str):
                    gt_parts = [p.strip() for p in ground_truth.split(';') if p.strip()]
                    pred_parts = [p.strip() for p in prediction.split(';') if p.strip()]
                else:
                    gt_parts = [str(ground_truth)]
                    pred_parts = [str(prediction)]

                # Calculate score based on GT parts order
                num_gt = len(gt_parts) 
                for i, gt_part in enumerate(gt_parts):
                    if i < len(pred_parts) and self.check_single_match(gt_part, pred_parts[i]):
                        score += 1.0 / num_gt
                
                if score > 0:
                    correct += score
            result['correct'] = score
        
        accuracy = correct / total if total > 0 else 0
        
        self.save_results(results, accuracy, correct, total, finish_reasons)
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def evaluate_FinMME(self) -> Dict:
        """Evaluate FinMME single-choice, multiple-choice, and numerical items."""
        results = self.load_results()
        total = len(results)
        correct = 0
        finish_reasons = {}

        for result in results:
            if "finish_reason" in result:
                finish_reasons[result["finish_reason"]] = finish_reasons.get(result["finish_reason"], 0) + 1
            
            ground_truth = result['ground_truth']
            prediction = result['model_answer']
            
            score = 0.0
            
            # Treat comma-separated labels as a multiple-choice answer.
            if isinstance(ground_truth, str) and ',' in ground_truth and isinstance(prediction, str):
                # Compare answer sets without considering order.
                gt_options = set([opt.strip() for opt in ground_truth.split(',') if opt.strip()])
                pred_options = set([opt.strip() for opt in prediction.split(',') if opt.strip()])
                
                if gt_options == pred_options:
                    score = 1.0
                elif pred_options and pred_options.issubset(gt_options):
                    score = 0.5
            else:
                # Single-choice or numerical item.
                tolerance = result.get('tolerance', None)
                if self.check_single_match(str(ground_truth), str(prediction), tolerance=tolerance):
                    score = 1.0
            
            if score > 0:
                correct += score
            result['correct'] = score
        
        accuracy = correct / total if total > 0 else 0
        
        self.save_results(results, accuracy, correct, total, finish_reasons)
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def evaluate_InstructFollow(self, grading_mode) -> Dict:
        """Evaluate InstructFollow with verifiable-instructions rules."""
        try:
            import nltk
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            import nltk
            nltk.download('punkt_tab', quiet=True)
        from verifiable_instructions import instructions_registry

        results = self.load_results()
        total = len(results)
        total_reward = 0.0
        finish_reasons = {}

        for result in results:
            if "finish_reason" in result:
                finish_reasons[result["finish_reason"]] = finish_reasons.get(result["finish_reason"], 0) + 1

            response = result["model_answer"]
            inst_ids = result.get("instruction_id_list", [])
            kw_list = result.get("if_kwargs", [{}] * len(inst_ids))

            following = []
            for inst_id, kw in zip(inst_ids, kw_list):
                try:
                    cls = instructions_registry.INSTRUCTION_DICT[inst_id]
                    inst = cls(inst_id)
                    kw = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in (kw or {}).items() if v is not None}
                    inst.build_description(**kw)
                    following.append(bool(inst.check_following(response)))
                except Exception as e:
                    print(f"Error checking instruction {inst_id}: {e}")
                    following.append(False)

            score = sum(following) / len(following) if following else 0.0
            if grading_mode == "binary":
                score = 1.0 if all(following) else 0.0
            total_reward += score
            result["follow_instruction_list"] = following
            result["follow_all_instructions"] = all(following)
            result["correct"] = score

        accuracy = total_reward / total if total > 0 else 0.0
        self.save_results(results, accuracy, total_reward, total, finish_reasons)
        return {"accuracy": accuracy, "total_reward": total_reward, "total": total}

    def evaluate_POPE(self):
        answers = [json.loads(q) for q in open(self.merged_file, 'r')]
        label_list = [json.loads(q)['ground_truth'] for q in open(self.merged_file, 'r')]
        finish_reasons = {}
        total = len(answers)
        for answer in answers:
            if "finish_reason" in answer:
                finish_reasons[answer["finish_reason"]] = finish_reasons.get(answer["finish_reason"], 0) + 1

        for i in range(len(label_list)):
            if label_list[i] == 'no':
                label_list[i] = 0
            else:
                label_list[i] = 1

        pred_list = []
        for answer in answers:
            if answer['model_answer'].lower() == 'no':
                pred_list.append(0)
            else:
                pred_list.append(1)

        pos = 1
        neg = 0
        TP, TN, FP, FN = 0, 0, 0, 0
        for pred, label in zip(pred_list, label_list):
            if pred == pos and label == pos:
                TP += 1
            elif pred == pos and label == neg:
                FP += 1
            elif pred == neg and label == neg:
                TN += 1
            elif pred == neg and label == pos:
                FN += 1

        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        f1 = 2*precision*recall / (precision + recall)
        print('F1 score: {}'.format(f1))
        print_lines = []
        print_lines.append(f'F1 score: {f1:.4f}\n')
        if finish_reasons:
            for reason, count in finish_reasons.items():
                percentage = (count / total) * 100 if total > 0 else 0
                print_lines.append(f'{reason}: {count} ({percentage:.2f}%)')

        os.makedirs(self.output_dir, exist_ok=True) 
        with open(f'{self.output_dir}/evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)

        output_file_path = os.path.join(self.output_dir, 'evaluation_stats.txt')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for print_line in print_lines:
                f.write(print_line + '\n')

    def evaluate_MMMU(self):
        print_lines = []
        results = self.load_results()

        processed_results = []
        true_nums = 0
        false_nums = 0
        finish_reasons = {}
        for i, result in enumerate(results):
            if "finish_reason" in result:
                finish_reasons[result["finish_reason"]] = finish_reasons.get(result["finish_reason"], 0) + 1
            new_data = mmmu_process_results(result)
            processed_results.append(new_data)
            
            if new_data['if_right']:
                true_nums += 1
                result['correct'] = True
            else:
                false_nums += 1
                result['correct'] = False

        # Calculate and output the accuracy
        total = true_nums + false_nums
        acc = true_nums / total * 100 if total > 0 else 0
        print_line = f"Total samples: {total}\nCorrect predictions: {true_nums}\nAccuracy: {acc:.2f}%\nFinish Reason Statistics:\n"
        if finish_reasons:
            for reason, count in finish_reasons.items():
                percentage = (count / total) * 100 if total > 0 else 0
                print_line += f'{reason}: {count} ({percentage:.2f}%)\n'
        
        print(print_line)
        print_lines.append(print_line)

        with open(f'{self.output_dir}/evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with open(f'{self.output_dir}/evaluation_stats.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(print_lines))

    def evaluate_Charxiv(self):
        """Evaluate merged CharXiv results."""
        results = self.load_results()
        total = len(results)
        correct = 0
        finish_reasons = {}
        dq_correct = 0.0   
        dq_total = 0       
        rq_correct = 0.0   
        rq_total = 0       

        for result in results:
            if "finish_reason" in result:
                finish_reasons[result["finish_reason"]] = finish_reasons.get(result["finish_reason"], 0) + 1
            result_type = result.get("Type", "") 
            if result_type == "DQ":
                dq_total += 1
            elif result_type == "RQ":
                rq_total += 1
            ground_truth = result['ground_truth']
            prediction = result['model_answer']
            
            score = 0.0
            if isinstance(ground_truth, list):
                # Try each accepted reference answer.
                for gt_item in ground_truth:
                    # Split by semicolon and ensure parts are clean
                    if isinstance(gt_item, str) and isinstance(prediction, str):
                        gt_parts = [p.strip() for p in gt_item.split(';') if p.strip()]
                        pred_parts = [p.strip() for p in prediction.split(';') if p.strip()]
                    else:
                        gt_parts = [str(ground_truth)]
                        pred_parts = [str(prediction)]
                    # Calculate score based on GT parts order
                    num_gt = len(gt_parts)
                    
                    for i, gt_part in enumerate(gt_parts):
                        if i < len(pred_parts) and self.check_single_match(gt_part, pred_parts[i]):
                            score += 1.0 / num_gt
                    
                    if score > 0:
                        correct += score
                        if result_type == "DQ":
                            dq_correct += score
                        elif result_type == "RQ":
                            rq_correct += score
                        break  # Stop after the first matching reference.
            else:
                # Split by semicolon and ensure parts are clean
                if isinstance(ground_truth, str) and isinstance(prediction, str):
                    gt_parts = [p.strip() for p in ground_truth.split(';') if p.strip()]
                    pred_parts = [p.strip() for p in prediction.split(';') if p.strip()]
                else:
                    gt_parts = [str(ground_truth)]
                    pred_parts = [str(prediction)]

                # Calculate score based on GT parts order
                num_gt = len(gt_parts) 
                for i, gt_part in enumerate(gt_parts):
                    if i < len(pred_parts) and self.check_single_match(gt_part, pred_parts[i]):
                        score += 1.0 / num_gt
                
                if score > 0:
                    correct += score
                    if result_type == "DQ":
                        dq_correct += score
                    elif result_type == "RQ":
                        rq_correct += score
            result['correct'] = score
        
        accuracy = correct / total if total > 0 else 0
        dq_accuracy = dq_correct / dq_total if dq_total > 0 else 0
        rq_accuracy = rq_correct / rq_total if rq_total > 0 else 0

        base_dir = self.output_dir  
        self.output_dir = f'{base_dir}/Total'
        os.makedirs(self.output_dir, exist_ok=True)
        self.save_results(results, accuracy, correct, total, finish_reasons)
        self.output_dir = f'{base_dir}/DQ'
        os.makedirs(self.output_dir, exist_ok=True)
        self.save_results(results, dq_accuracy, dq_correct, dq_total)
        self.output_dir = f'{base_dir}/RQ'
        os.makedirs(self.output_dir, exist_ok=True)
        self.save_results(results, rq_accuracy, rq_correct, rq_total)

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def save_results(self, results: List[Dict], accuracy: float, correct: int, total: int, finish_reasons: Dict[str, int] = None):
        """Save detailed evaluation results and aggregate statistics."""
        # Save detailed results as JSON.
        with open(f'{self.output_dir}/evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save aggregate statistics.
        with open(f'{self.output_dir}/evaluation_stats.txt', 'w', encoding='utf-8') as f:
            f.write(f'Total samples: {total}\n')
            f.write(f'Correct predictions: {correct}\n')
            f.write(f'Accuracy: {accuracy*100:.2f}%\n')
            
            if finish_reasons:
                f.write('\nFinish Reason Statistics:\n')
                for reason, count in finish_reasons.items():
                    percentage = (count / total) * 100 if total > 0 else 0
                    f.write(f'{reason}: {count} ({percentage:.2f}%)\n')
        
        print(f'\n========================================')
        print(f'Evaluation Complete!')
        print(f'========================================')
        print(f'Total samples: {total}')
        print(f'Correct predictions: {correct}')
        print(f'Accuracy: {accuracy*100:.2f}%')
        print(f'========================================')
        print(f'Results saved to: {self.output_dir}')
        print(f'========================================')


def main():
    parser = argparse.ArgumentParser(description='Exact Match Evaluation Script')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--merged_file', type=str, required=True, help='Path to merged inference results')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation results')

    args = parser.parse_args()
    
    evaluator = ExactMatchEvaluator(
        dataset_name=args.dataset_name,
        merged_file=args.merged_file,
        output_dir=args.output_dir
    )
    if "POPE" in args.dataset_name:
        evaluator.evaluate_POPE()
    elif "MMMU" in args.dataset_name:
        evaluator.evaluate_MMMU()
    elif "FinMME" in args.dataset_name:
        evaluator.evaluate_FinMME()
    elif "InstructFollow" in args.dataset_name:
        evaluator.evaluate_InstructFollow(grading_mode='binary')  # 'fraction' or 'binary'
    elif "Charxiv" in args.dataset_name:
        evaluator.evaluate_Charxiv()
    else:
        evaluator.evaluate()


if __name__ == "__main__":
    main()

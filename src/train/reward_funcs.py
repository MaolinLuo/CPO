import re
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig

def clean_expression(ground_truth: str, prediction: str):
    clean_pattern = r"^(\s*(?:\$\$?)?)\s*(?:f\s*\(\s*x\s*\)|y|f\s*\\left\(\s*x\s*\\right\))\s*=\s*(?![^=]*=)"
    ground_truth = re.sub(clean_pattern, r"\1", ground_truth, flags=re.IGNORECASE)
    prediction = re.sub(clean_pattern, r"\1", prediction, flags=re.IGNORECASE)

    # Roman numeral replacement for quadrants
    roman_map = {"I": "First", "II": "Second", "III": "Third", "IV": "Fourth"}
    if ground_truth.strip() in roman_map:
        ground_truth = roman_map[ground_truth.strip()]
    if prediction.strip() in roman_map:
        prediction = roman_map[prediction.strip()]

    return ground_truth, prediction

def check_single_match(ground_truth: str, prediction: str) -> bool:
    """Check if single expression matches"""
    ground_truth, prediction = clean_expression(ground_truth, prediction)
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
        ignore_chars = {'\\', ',', ';', ' ', '{', '}', '$'}
        for char in ignore_chars:
            pred = pred.replace(char, '')
        for char in ignore_chars:
            gt = gt.replace(char, '')
        
        if gt == pred:
            score = True
    
    return score

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

def accuracy_reward(completions, assistant, **kwargs):
    """Reward function that extracts the last \boxed{} and checks accuracy."""
    rewards = []

    for comp, sol in zip(completions, assistant):
        completion = comp[0]["content"]
        
        content_to_eval = extract_last_boxed(completion).strip()

        score = 0.0
        ground_truth = sol
        prediction = content_to_eval

        # An empty extraction skips matching and receives zero reward.
        if prediction:
            # Split by semicolon and ensure parts are clean
            gt_parts = [p.strip() for p in ground_truth.split(';') if p.strip()]
            if not gt_parts: gt_parts = [ground_truth]
            
            pred_parts = [p.strip() for p in prediction.split(';') if p.strip()]
            
            num_gt = len(gt_parts)
            
            for i, gt_part in enumerate(gt_parts):
                if i < len(pred_parts) and check_single_match(gt_part, pred_parts[i]):
                    score += 1.0 / num_gt

        rewards.append(score)
    try:
        print(f"+ Reward: {score}; Answer: {content_to_eval}; GT: {sol}; Output: {completion.replace('\n', ' ')}")  # Just check a sample
    except:
        pass

    return rewards

def length_reward(completions, **kwargs):
    """Penalize responses that are too short to discourage lazy one-liner answers.

    Returns:
        -0.5  if word count < MIN_WORDS
         0.0  otherwise
    """
    MIN_WORDS = 20
    rewards = []
    for comp in completions:
        response = comp[0]["content"]
        word_count = len(response.split())
        rewards.append(-0.5 if word_count < MIN_WORDS else 0.0)
    return rewards

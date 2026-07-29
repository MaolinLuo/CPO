import re
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig


def clean_expression(ground_truth: str, prediction: str):
    clean_pattern = r"^(\s*(?:\$\$?)?)\s*(?:f\s*\(\s*x\s*\)|y|f\s*\\left\(\s*x\s*\\right\))\s*=\s*(?![^=]*=)"
    ground_truth = re.sub(clean_pattern, r"\1", ground_truth, flags=re.IGNORECASE)
    prediction = re.sub(clean_pattern, r"\1", prediction, flags=re.IGNORECASE)

    roman_map = {"I": "First", "II": "Second", "III": "Third", "IV": "Fourth"}
    if ground_truth.strip() in roman_map:
        ground_truth = roman_map[ground_truth.strip()]
    if prediction.strip() in roman_map:
        prediction = roman_map[prediction.strip()]

    return ground_truth, prediction


def check_single_match(ground_truth: str, prediction: str, tolerance=None) -> bool:
    """Check if single expression matches, with optional numeric tolerance."""
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
        elif tolerance is not None:
            try:
                pred_value = float(pred)
                gt_value = float(gt)
                if abs(pred_value - gt_value) <= tolerance:
                    score = True
            except (ValueError, TypeError):
                pass

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

def accuracy_reward(completions, assistant, tolerance=None, **kwargs):
    """Reward function for FinMME: handles single-choice, multi-choice, and numerical questions."""
    rewards = []

    for idx, (comp, sol) in enumerate(zip(completions, assistant)):
        completion = comp[0]["content"]
        content_to_eval = extract_last_boxed(completion).strip()

        ground_truth = sol
        prediction = content_to_eval
        tol = tolerance[idx] if tolerance is not None else None

        score = 0.0

        if isinstance(ground_truth, str) and ',' in ground_truth:
            # Multi-choice: order-independent comparison
            gt_options = set(opt.strip() for opt in ground_truth.split(',') if opt.strip())
            pred_options = set(opt.strip() for opt in prediction.split(',') if opt.strip())

            if gt_options == pred_options:
                score = 1.0
            elif pred_options and pred_options.issubset(gt_options):
                score = 0.5
        else:
            # Single-choice or numerical
            if check_single_match(str(ground_truth), str(prediction), tolerance=tol):
                score = 1.0

        rewards.append(score)
    try:
        print(f"+ Reward: {score}; Answer: {content_to_eval}; GT: {sol}; Tol: {tol}; Output: {completion.replace('\n', ' ')}")  # Just check a sample
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

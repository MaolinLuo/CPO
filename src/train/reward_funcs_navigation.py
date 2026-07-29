import re

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
    """Reward function that checks if the completion is correct for navigation."""
    rewards = []

    for comp, sol in zip(completions, assistant):
        completion = comp[0]["content"]
        content_to_eval = extract_last_boxed(completion).strip()

        score = 0.0
        ground_truth = sol
        prediction = content_to_eval

        # Split by comma
        gt_moves = [x.strip() for x in ground_truth.split(',') if x.strip()]
        pred_moves = [x.strip() for x in prediction.split(',') if x.strip()]

        if not gt_moves:
            rewards.append(0.0)
            continue
        
        num_gt = len(gt_moves)
        num_pred = len(pred_moves)
        matches = 0
        
        for i in range(min(num_gt, num_pred)):
            if gt_moves[i] == pred_moves[i]:
                score += 1.0 / num_gt
                matches += 1
            else:
                break
        
        # Redundancy penalty: 
        # If preceding is all correct (matches == num_gt), but there are extra paths (num_pred > num_gt), 0 points.
        if matches == num_gt and num_pred > num_gt:
            score = 0.0
            
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

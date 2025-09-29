import re
from typing import Any, Dict, List

from mathruler.grader import grade_answer, extract_boxed_content

def format_reward(response: str) -> float:
    patterns = [
        re.compile(r"###\s*Image Description:", re.IGNORECASE),
        re.compile(r"###\s*Rationales:", re.IGNORECASE),
        re.compile(r"###\s*Step\s*1:", re.IGNORECASE),
        re.compile(r"###\s*The final answer is:", re.IGNORECASE)
    ]

    if all(p.search(response) for p in patterns):
        return 1.0
    return 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    content_match = re.search(r"### The final answer is:\s*([^\n]*)", response)
    given_answer = content_match.group(1).strip() if content_match else response.strip()
    # ground_truth = extract_boxed_content(ground_truth)
    
    # with open("debug_accuracy_reward.txt", "a", encoding="utf-8") as log_file:
    #     log_file.write("==== SAMPLE START ====\n")
    #     log_file.write(f"📝 Raw response:\n{response}\n")
    #     log_file.write(f"✅ Ground truth:\n{ground_truth}\n")
    #     log_file.write(f"🔍 Extracted answer:\n{given_answer}\n")
    #     log_file.write("==== SAMPLE END ====\n\n")

    reward = 1.0 if grade_answer(given_answer, ground_truth.strip()) else 0.0
    return reward


def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores

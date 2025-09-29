import re
from typing import Dict, List, Any
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import ray
import torch
from mathruler.grader import grade_answer, extract_boxed_content

# 初始化 Ray（只初始化一次）
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)
    
# 定义 Ray 远程 worker
@ray.remote(num_gpus=0.2)
class ClipScorer:
    def __init__(self, model_root="qihoo360/fg-clip-large", image_size=336):
        self.device = torch.device("cuda")
        self.image_size = image_size
        self.clip_model = AutoModelForCausalLM.from_pretrained(
            model_root, trust_remote_code=True, local_files_only=True
        ).to(self.device)
        self.clip_tokenizer = AutoTokenizer.from_pretrained(model_root)
        self.clip_processor = AutoImageProcessor.from_pretrained(model_root)

    def compute_clip_score(self, response: str, image: Image.Image) -> float:
        image = image.convert("RGB").resize((self.image_size, self.image_size))
        image_input = self.clip_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(self.device)

        caption_match = re.search(r"###\s*Image Description:\s*(.*?)###\s*Rationales:", response, re.DOTALL | re.IGNORECASE)
        caption = caption_match.group(1).strip() if caption_match else ""
        caption_input = torch.tensor(
            self.clip_tokenizer([caption], max_length=248, padding="max_length", truncation=True).input_ids,
            dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            image_feat = self.clip_model.get_image_features(image_input)
            text_feat = self.clip_model.get_text_features(caption_input, walk_short_pos=False)
            image_feat = image_feat / image_feat.norm(p=2, dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
            return (image_feat @ text_feat.T).item()

clip_worker = None  # 全局变量

def get_clip_worker():
    global clip_worker
    if clip_worker is None:
        print("🚀 初始化 Ray ClipScorer worker...")
        clip_worker = ClipScorer.remote()
    return clip_worker





def clip_score_reward(response: str, image: Image.Image) -> float:
    worker = get_clip_worker()
    try:
        future = worker.compute_clip_score.remote(response, image)
        clip_score =  ray.get(future)
        if clip_score >= 0.4:
            return 1.0
        else:
            return clip_score / 0.4
    except Exception as e:
        print(f"❌ Ray call failed in clip_score_reward: {e}")
        return 0.0 
    
def key_word_reward(response: str, keyword: list) -> float:
    if not keyword:
        return 0.0  # 防止除以 0

    response_lower = response.lower()
    match_count = sum(1 for kw in keyword if kw.lower() in response_lower)
    return match_count / len(keyword)

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

def length_penalty(response_length: int, expected_length: int, max_penalty: float = 1.0) -> float:
    """
    给响应 token 长度一个惩罚项。token 越多，得分越低。
    """
    if response_length <= expected_length:
        return 1.0
    else:
        penalty = expected_length / (response_length + 1e-5)
        return max(0.0, min(penalty, max_penalty))
    

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

def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1, clip_weight: float = 0.4, key_word_weight: float = 0.4, length_penalty_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        response_length = reward_input["response_length"]
        format_score = format_reward(response)
        clip_score = clip_score_reward(response, reward_input["image"])
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        key_word_score = key_word_reward(response, reward_input["key_word"])
        length_penalty_score = length_penalty(response_length, expected_length=512)
        
        
        scores.append(
            {
                "overall": clip_weight * clip_score + format_weight * format_score + key_word_weight * key_word_score + length_penalty_weight * length_penalty_score,
                "format": format_score,
                "clip": clip_score,
                "key_word": key_word_score,
                "accuracy": accuracy_score,
                "length": length_penalty_score
            }
        )
    return scores
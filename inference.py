from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info
import torch, os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_path = "./PeBR_R1_3B"     # Downloaded model from https://huggingface.co/cythu/PeBR_R1

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path)

SYSTEM_PROMPT = """Generate an image description based on the question.
Then, provide a rationale to analyze the question.
Next, generate a step-by-step reasoning process to solve the problem.
Finally, summarize with: 'The final answer is: xxx'.

Format:
### Image Description:
### Rationales:
### Let's think step by step.
### Step 1:
### Step 2:
...
### The final answer is: 

{problem}
"""

problem = "Subtract all brown blocks. Subtract all large blue rubber things. How many objects are left?"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "./demo.jpg"},
            {"type": "text", "text": SYSTEM_PROMPT.format(problem=problem)},
        ],
    }
]

print("🚀 Running inference...")

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

gen_config = GenerationConfig(
    max_new_tokens=4096,
    temperature=1e-6
)

with torch.inference_mode():
    generated_ids = model.generate(**inputs, generation_config=gen_config)

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print("\n=================  Model Output =================\n")
print(output_text)
print("\n===================================================")

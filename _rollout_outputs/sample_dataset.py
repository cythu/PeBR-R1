import json
from collections import defaultdict
import os

# ==== Paths ====
dataset_path = "../dataset/pebr_grpo_dataset.json"
result_path = "./rollout_results.jsonl"

# ==== Save JSON file ====
def save_json(data_list, filename):
    # Find the directory of dataset_path
    out_dir = os.path.dirname(os.path.abspath(dataset_path))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, filename)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {out_file} with {len(data_list)} samples.")

# ==== Step 1: Categorize qids ====
qid_to_accs = defaultdict(list)

with open(result_path, "r") as f:
    for line in f:
        try:
            data = json.loads(line)
            qid = data.get("qid")
            score = data.get("score", {})
            acc = score.get("accuracy", None)
            if qid is not None and acc is not None:
                qid_to_accs[qid].append(acc)
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)

easy_qids, medium_qids, hard_qids = set(), set(), set()

for qid, accs in qid_to_accs.items():
    if len(accs) != 8:
        print(f"⚠️ Warning: qid={qid} has {len(accs)} samples, expected 8.")
    correct = sum(1 for a in accs if a == 1.0)
    if correct == 8:
        easy_qids.add(qid)
    elif correct == 0:
        hard_qids.add(qid)
    else:
        medium_qids.add(qid)

print("📊 Case Statistics:")
print(f"Easy cases:   {len(easy_qids)}")
print(f"Medium cases: {len(medium_qids)}")
print(f"Hard cases:   {len(hard_qids)}")
print(f"Total qids:   {len(qid_to_accs)}")

# ==== Step 2: Load original JSON and filter ====
print("📥 Loading original JSON...")
with open(dataset_path, "r", encoding="utf-8") as f:
    all_data = json.load(f)

# Filter data by qid categories
easy_data = [sample for sample in all_data if sample.get("qid") in easy_qids]
medium_data = [sample for sample in all_data if sample.get("qid") in medium_qids]
hard_data = [sample for sample in all_data if sample.get("qid") in hard_qids]

# ==== Step 3: Save categorized JSON files ====
print("💾 Saving JSON files...")
save_json(easy_data, "easycase.json")
save_json(medium_data, "mediumcase.json")
save_json(hard_data, "hardcase.json")

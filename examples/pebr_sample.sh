#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=./pebr_r1_warm_up  # replace it with your local file path

python3 -m verl.trainer.pebr_sample_main \
    config=examples/pebr_sample.yaml \
    data.train_files=dataset/pebr_grpo_dataset.json \
    data.val_files=hiyouga/math12k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=./examples/reward_function/pebr_sample.py:compute_score \
    worker.rollout.tensor_parallel_size=4 \
    trainer.experiment_name=dataset_sample \
    trainer.n_gpus_per_node=8 

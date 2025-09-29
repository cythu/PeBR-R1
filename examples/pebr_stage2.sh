#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=./output/pebr_r1_stage1_rl  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/pebr_stage2.yaml \
    data.train_files=dataset/mediumcase.json \
    data.val_files=dataset/mediumcase.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=./examples/reward_function/pebr_2.py:compute_score \
    worker.rollout.tensor_parallel_size=4 \
    trainer.experiment_name=stage2_rl \
    trainer.n_gpus_per_node=8 

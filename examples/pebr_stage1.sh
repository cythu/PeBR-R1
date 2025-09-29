#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=./pebr_r1_warm_up  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/pebr_stage1.yaml \
    data.train_files=dataset/easycase.json \
    data.val_files=dataset/easycase.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=./examples/reward_function/pebr_1.py:compute_score \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=stage1_rl \
    trainer.n_gpus_per_node=7 

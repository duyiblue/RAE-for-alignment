#! /bin/bash

python alignment_training.py \
    --source_config $(pwd)/configs/stage1/pretrained/DINOv2-B_512.yaml \
    --target_config $(pwd)/configs/stage1/pretrained/DINOv2-B_512.yaml \
    --dataset robot \
    --batch_size 6
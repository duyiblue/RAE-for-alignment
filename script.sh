#! /bin/bash

python alignment_training.py \
    --source_config $(pwd)/configs/stage1/pretrained/DINOv2-B.yaml \
    --target_config $(pwd)/configs/stage1/pretrained/DINOv2-B.yaml \
    --dataset voc \
    --apply_mask \
    --batch_size 32
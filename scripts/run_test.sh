#!/bin/bash
# 测试脚本
export CUDA_VISIBLE_DEVICES=0

python scripts/hercules_test.py \
    --sequence Library \
    --checkpoint checkpoints/ddp_v1/hercules_best.pt \
    --output_dir results/ddp_v1 \
    --batch_size 4 \
    --gpu 0

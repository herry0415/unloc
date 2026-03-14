#!/bin/bash
# 训练脚本 - 3卡 DDP
export CUDA_VISIBLE_DEVICES=1,2,3

python -m torch.distributed.launch \
    --nproc_per_node=3 \
    scripts/hercules_train_ddp.py \
    --sequence Library \
    --batch_size 6 \
    --epochs 40 \
    --num_gpus 3 \
    --output_dir checkpoints/ddp_v1 \
    --lr 0.0001 \
    --num_workers 4

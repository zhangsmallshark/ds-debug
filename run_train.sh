#!/bin/bash --login

deepspeed --include="localhost:1,2" diffu.py --deepspeed

# python -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMCwgMSwgMiwgM119 --master_addr=127.0.0.1 --master_port=29500 --enable_each_rank_log=None train_mvdiffusion_image_sd21_unclip_joint.py --config configs/train/ortho/joint-512.yaml --deepspeed

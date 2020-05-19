#!/bin/bash
python test_RGBD_ms.py --arch_F fconv_ms --arch_map map_conv --no_testset \
  --img_path ./sample_pic/seven_rgb --depth_path ./sample_pic/seven_depth --d_scale 40000 --img_rows 480 --img_cols 640 \
  --model_savepath ./checkpoint/FCONV_MS/  --model_full_name fconv_ms_matterport_l1_2_hybrid_best.pkl \
  --out_path /home/maximilianm/00_NVIDIA/research/RGBD/RGBD2Normal/result/demo_rgbd_mp/

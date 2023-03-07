#!/bin/bash
torchrun --nproc_per_node=48 gen_ranking_result.py --scene_graph_path /home/lzj/code/sgg/outputs/filted_scene_graphs.pkl --target_folder /home/lzj/code/sgg/outputs
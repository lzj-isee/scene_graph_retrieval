#!/bin/bash
torchrun --nproc_per_node=48 Evaluation/gen_ranking_result.py
#!/bin/bash
export PYTHONPATH=src/third_party:$PYTHONPATH
ulimit -n 65535 || true

GPU=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

WANDB_MODE=offline EMBODIEDSPLAT_OPENSEG_CPU=1 CUDA_VISIBLE_DEVICES=$GPU python -m src.main +experiment=scannet/embodiedsplat_test.yaml +output_dir='' mode=test dataset/view_sampler=evaluation checkpointing.load=pretrained/embodiedsplat_scannet.ckpt dataset.view_sampler.num_context_views=103 model.encoder.num_views=30

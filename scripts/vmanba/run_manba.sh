#/bin/bash

torchrun --standalone --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
    main_mlc.py --cfg config/vmanba_fc/coco2014_small.yaml \
    --output checkpoint/vmanba_fc/coco2014_small/seed42/work14 \
    --prob 0.5 \
    --gpus 4,5,6,7 \
    --seed 42 \
    --print-freq 400


torchrun --standalone --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
    main_mlc.py --cfg config/vmanba_fc/voc2007_small.yaml \
    --output checkpoint/vmanba_fc/voc2007_small/seed42/work14 \
    --prob 0.5 \
    --gpus 4,5,6,7 \
    --seed 42 \
    --print-freq 400
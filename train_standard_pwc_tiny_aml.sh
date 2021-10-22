#!/bin/bash
python -u RAFT/train.py --model pwc_tiny --name raft-chairs --stage chairs --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 40 --lr 0.0004 --image_size 368 496 --wdecay 0.0001
python -u RAFT/train.py --model pwc_tiny --name raft-things --stage things --validation sintel --restore_ckpt raft_output/pwc_tiny_default/raft-chairs.pth --gpus 0 1 --num_steps 100000 --batch_size 24 --lr 0.000125 --image_size 400 720 --wdecay 0.0001
python -u RAFT/train.py --model pwc_tiny --name raft-sintel --stage sintel --validation sintel --restore_ckpt raft_output/pwc_tiny_default/raft-things.pth --gpus 0 1 --num_steps 100000 --batch_size 24 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma=0.85
python -u RAFT/train.py --model pwc_tiny --name raft-kitti  --stage kitti --validation kitti --restore_ckpt raft_output/pwc_tiny_default/raft-sintel.pth --gpus 0 1 --num_steps 50000 --batch_size 24 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85

import os


datasets = ['blender_old', 'gen_mobilenet', 'turk_test']
modes =  ['bw', 'fw']

for d, dataset in enumerate(datasets):
    for m, mode in enumerate(modes):
        device = d * len(modes) + m
        os.system(f"export CUDA_VISIBLE_DEVICES={device} && python RAFT/demo_seg_data.py --dataset {dataset} --mode {mode}")


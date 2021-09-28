import argparse
import multiprocessing as mp
import os

from .generate_seg_flow_seqs import generate


parser = argparse.ArgumentParser()
parser.add_argument('--model', help="restore checkpoint", default='RAFT/models/raft-kitti.pth')
parser.add_argument('--path', help="dataset for evaluation", default='PSEG')
parser.add_argument('--mode', help="forward or backward")
parser.add_argument('--dataset', help='dataset')
parser.add_argument('--fw_output_path', help="output path for evaluation")
parser.add_argument('--bw_output_path', help="output path for evaluation")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
args = parser.parse_args()

data_root = args.path
datasets = ['gen_mobilenet']
# modes =  ['bw', 'fw']
modes = ['fw_t-1->t']
challenges = ['Turkdata_201911', 'Youtube2']

process_num = 2

def process_files(process_id, datasets, modes, args, data_root):
    # mode_idx = process_id % len(modes)
    # dataset_idx = process_id // len(modes)
    # dataset = datasets[dataset_idx]
    # mode = modes[mode_idx]
    dataset = 'gen_mobilenet'
    mode = 'fw_t-1->t'
    cha = challenges[process_id]
    device = f'cuda:{process_id%2}'
    print(f"process {process_id}, dataset {dataset}, mode {mode}")

    generate(args, dataset, mode, device, data_root, process_id, cha)


processes = [mp.Process(target=process_files,
                        args=(process_id, datasets, modes, args, data_root))
                        for process_id in range(process_num)]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

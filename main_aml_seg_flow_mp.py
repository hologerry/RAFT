import argparse
import multiprocessing as mp
import os

from generate_seg_flow import generate


parser = argparse.ArgumentParser()
parser.add_argument('--model', help="restore checkpoint", default='RAFT/models/raft-things.pth')
parser.add_argument('--path', help="dataset for evaluation")
parser.add_argument('--mode', help="forward or backward")
parser.add_argument('--dataset', help='dataset')
parser.add_argument('--fw_output_path', help="output path for evaluation")
parser.add_argument('--bw_output_path', help="output path for evaluation")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
args = parser.parse_args()


datasets = ['blender_old', 'gen_mobilenet', 'turk_test']
modes =  ['bw', 'fw']

process_num = len(datasets) * len(modes)

def process_files(process_id, datasets, modes, args):
    dataset = datasets[process_id]
    mode = modes[process_id]
    device = f'cuda:{process_id}'
    generate(args, dataset, mode, device)


processes = [mp.Process(target=process_files,
                        args=(process_id, datasets, modes, args))
                        for process_id in range(process_num)]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()
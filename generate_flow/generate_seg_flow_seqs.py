import argparse
import glob
import multiprocessing as mp
import os

import cv2
import numpy as np
import torch
from cv2 import imread, imwrite
from PIL import Image
from tqdm import tqdm
from tqdm.contrib import tzip

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.frame_utils import writeFlow
from core.utils.utils import InputPadder


def load_image(imfile, device):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)


def save_flow(flo, out_flowfile):
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    writeFlow(out_flowfile, flo)


def generate(args, dataset, mode, device, data_root, process_id, cha):

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(device)
    model.eval()

    dataset_root = f'./data/{data_root}'
    fw_flow_data_root = f'./data/{data_root}_flow_fw'
    fwt_flow_data_root = f'./data/{data_root}_flow_fwt'
    bw_flow_data_root = f'./data/{data_root}_flow_bw'

    datasets = ['blender_old', 'gen_mobilenet', 'turk_test']

    print(f"dataset {dataset} mode {mode}")
    with torch.no_grad():
        jpeg_path = os.path.join(dataset_root, dataset, 'JPEGImages', '480p')
        if dataset == 'gen_mobilenet':
            challenges = sorted(os.listdir(jpeg_path))
            sequences = sorted(os.listdir(os.path.join(jpeg_path, cha)))
            for i in range(len(sequences)):
                seq = sequences[i]
                seq_path = os.path.join(jpeg_path, cha, seq)
                fw_output_seq_path = seq_path.replace(dataset_root, fw_flow_data_root)
                fwt_output_seq_path = seq_path.replace(dataset_root, fwt_flow_data_root)
                bw_output_seq_path = seq_path.replace(dataset_root, bw_flow_data_root)
                images = glob.glob(os.path.join(seq_path, '*.png')) + glob.glob(os.path.join(seq_path, '*.jpg'))

                os.makedirs(fw_output_seq_path, exist_ok=True)
                os.makedirs(fwt_output_seq_path, exist_ok=True)
                os.makedirs(bw_output_seq_path, exist_ok=True)

                images = sorted(images)

                for imfile_p, imfile_c in tzip(images[:-1], images[1:], leave=False, desc=f'dataset {dataset} mode {mode} sequence {seq}'):
                    fwt_out_flowfile1 = imfile_c.replace(dataset_root, fwt_flow_data_root).replace(".jpg", '.flo')
                    if os.path.exists(fwt_out_flowfile1):
                        continue
                    image_p = load_image(imfile_p, device)
                    image_c = load_image(imfile_c, device)

                    padder = InputPadder(image_p.shape)
                    image_p, image_c = padder.pad(image_p, image_c)

                    fwt_flow_low, fwt_flow_up = model(image_p, image_c, iters=20, test_mode=True)

                    save_flow(fwt_flow_up, fwt_out_flowfile1)


import argparse
import glob
import os


import cv2
import numpy as np
from tqdm.contrib import tzip
from tqdm import tqdm
import torch
from PIL import Image
from cv2 import imwrite, imread

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder


DEVICE = 'cuda'


def demo(args):

    dataset_root = '/D_data/Seg/data/PSEG'
    v_flip_dataset_data_root = f'/D_data/Seg/data/PSEG_v_flip'
    h_flip_dataset_data_root = f'/D_data/Seg/data/PSEG_h_flip'
    hv_flip_dataset_data_root = f'/D_data/Seg/data/PSEG_hv_flip'

    datasets = ['blender_old', 'gen_mobilenet', 'turk_test']
    assert args.dataset in datasets

    print(f"Fliping dataset {args.dataset}")
    with torch.no_grad():
        jpeg_path = os.path.join(dataset_root, args.dataset, 'Annotations', '480p')

        if args.dataset == 'blender_old' or args.dataset == 'turk_test':
            if args.dataset == 'turk_test':
                jpeg_path = jpeg_path.replace('Annotations', 'GroundTruthAll')
            sequences = sorted(os.listdir(jpeg_path))

            for i, seq in tqdm(enumerate(sequences), total=len(sequences)):
                seq_path = os.path.join(jpeg_path, seq)
                v_flip_output_seq_path = seq_path.replace(dataset_root, v_flip_dataset_data_root)
                h_flip_output_seq_path = seq_path.replace(dataset_root, h_flip_dataset_data_root)
                hv_flip_output_seq_path = seq_path.replace(dataset_root, hv_flip_dataset_data_root)
                images = glob.glob(os.path.join(seq_path, '*.png')) + glob.glob(os.path.join(seq_path, '*.jpg'))

                os.makedirs(v_flip_output_seq_path, exist_ok=True)
                os.makedirs(h_flip_output_seq_path, exist_ok=True)
                os.makedirs(hv_flip_output_seq_path, exist_ok=True)

                images = sorted(images)

                for imfile in tqdm(images, leave=False):
                    img = imread(imfile)
                    img_v_flip = cv2.flip(img, flipCode=0)
                    img_h_flip = cv2.flip(img, flipCode=1)
                    img_hv_flip = cv2.flip(img, flipCode=-1)
                    v_out_imfile = imfile.replace(dataset_root, v_flip_dataset_data_root)
                    h_out_imfile = imfile.replace(dataset_root, h_flip_dataset_data_root)
                    hv_out_imfile = imfile.replace(dataset_root, hv_flip_dataset_data_root)
                    imwrite(v_out_imfile, img_v_flip)
                    imwrite(h_out_imfile, img_h_flip)
                    imwrite(hv_out_imfile, img_hv_flip)


        elif args.dataset == 'gen_mobilenet':
            challenges = sorted(os.listdir(jpeg_path))
            for cha in tqdm(challenges, total=len(challenges)):

                v_jpeg_path = os.path.join(v_flip_dataset_data_root, args.dataset, 'Annotations', '480p')
                v_challenges = sorted(os.listdir(v_jpeg_path))
                if cha in v_challenges:
                    continue
                sequences = sorted(os.listdir(os.path.join(jpeg_path, cha)))
                for i, seq in tqdm(enumerate(sequences), total=len(sequences)):
                    seq_path = os.path.join(jpeg_path, cha, seq)
                    v_flip_output_seq_path = seq_path.replace(dataset_root, v_flip_dataset_data_root)
                    h_flip_output_seq_path = seq_path.replace(dataset_root, h_flip_dataset_data_root)
                    hv_flip_output_seq_path = seq_path.replace(dataset_root, hv_flip_dataset_data_root)
                    images = glob.glob(os.path.join(seq_path, '*.png')) + glob.glob(os.path.join(seq_path, '*.jpg'))

                    os.makedirs(v_flip_output_seq_path, exist_ok=True)
                    os.makedirs(h_flip_output_seq_path, exist_ok=True)
                    os.makedirs(hv_flip_output_seq_path, exist_ok=True)

                    images = sorted(images)

                    for imfile in tqdm(images, leave=False):
                        img = imread(imfile)
                        img_v_flip = cv2.flip(img, flipCode=0)
                        img_h_flip = cv2.flip(img, flipCode=1)
                        img_hv_flip = cv2.flip(img, flipCode=-1)
                        v_out_imfile = imfile.replace(dataset_root, v_flip_dataset_data_root)
                        h_out_imfile = imfile.replace(dataset_root, h_flip_dataset_data_root)
                        hv_out_imfile = imfile.replace(dataset_root, hv_flip_dataset_data_root)
                        imwrite(v_out_imfile, img_v_flip)
                        imwrite(h_out_imfile, img_h_flip)
                        imwrite(hv_out_imfile, img_hv_flip)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--dataset', help='dataset')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
    args = parser.parse_args()

    demo(args)

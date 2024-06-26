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
from core.utils.frame_utils import writeFlow


def load_image(imfile, device):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)


def viz(img, flo, out_imfile):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    imwrite(out_imfile, flo)
    # img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.savefig(out_imfile)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2, 1, 0]]/255.0)
    # cv2.waitKey()

def save_flow(flo, out_flowfile):
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    writeFlow(out_flowfile, flo)


def generate(args, dataset, mode, device, data_root):

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

        if dataset == 'blender_old' or dataset == 'turk_test':
            sequences = sorted(os.listdir(jpeg_path))

            for i, seq in tqdm(enumerate(sequences), total=len(sequences), desc=f'dataset {dataset} mode {mode} sequences'):
                seq_path = os.path.join(jpeg_path, seq)
                fw_output_seq_path = seq_path.replace(dataset_root, fw_flow_data_root)
                fwt_output_seq_path = seq_path.replace(dataset_root, fwt_flow_data_root)
                bw_output_seq_path = seq_path.replace(dataset_root, bw_flow_data_root)
                images = glob.glob(os.path.join(seq_path, '*.png')) + glob.glob(os.path.join(seq_path, '*.jpg'))


                os.makedirs(fw_output_seq_path, exist_ok=True)
                os.makedirs(fwt_output_seq_path, exist_ok=True)
                os.makedirs(bw_output_seq_path, exist_ok=True)

                images = sorted(images)

                if mode == 'fw':
                    for imfile1, imfile2 in tzip(images[:-1], images[1:], leave=False, desc=f'dataset {dataset} mode {mode} sequence {seq}'):
                        fw_out_imfile1 = imfile1.replace(dataset_root, fw_flow_data_root).replace(".jpg", '.png')
                        if os.path.exists(fw_out_imfile1):
                            continue
                        image1 = load_image(imfile1, device)
                        image2 = load_image(imfile2, device)

                        padder = InputPadder(image1.shape)
                        image1, image2 = padder.pad(image1, image2)

                        fw_flow_low, fw_flow_up = model(image1, image2, iters=20, test_mode=True)

                        viz(image1, fw_flow_up, fw_out_imfile1)

                elif mode == 'bw':
                    for imfile_p, imfile_c in tzip(images[:-1], images[1:], leave=False, desc=f'dataset {dataset} mode {mode} sequence {seq}'):
                        bw_out_imfile1 = imfile_c.replace(dataset_root, bw_flow_data_root).replace(".jpg", '.png')
                        if os.path.exists(bw_out_imfile1):
                            continue
                        image_p = load_image(imfile_p, device)
                        image_c = load_image(imfile_c, device)

                        padder = InputPadder(image_p.shape)
                        image_p, image_c = padder.pad(image_p, image_c)

                        bw_flow_low, bw_flow_up = model(image_c, image_p, iters=20, test_mode=True)

                        viz(image_c, bw_flow_up, bw_out_imfile1)

                elif mode == 'fw_t-1->t':
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


        elif dataset == 'gen_mobilenet':
            challenges = sorted(os.listdir(jpeg_path))
            for cha in tqdm(challenges, total=len(challenges), desc=f'dataset {dataset} mode {mode} challenges'):
                sequences = sorted(os.listdir(os.path.join(jpeg_path, cha)))
                for i, seq in tqdm(enumerate(sequences), total=len(sequences), desc=f'dataset {dataset} mode {mode} challenge {cha} sequences'):
                    seq_path = os.path.join(jpeg_path, cha, seq)
                    fw_output_seq_path = seq_path.replace(dataset_root, fw_flow_data_root)
                    fwt_output_seq_path = seq_path.replace(dataset_root, fwt_flow_data_root)
                    bw_output_seq_path = seq_path.replace(dataset_root, bw_flow_data_root)
                    images = glob.glob(os.path.join(seq_path, '*.png')) + glob.glob(os.path.join(seq_path, '*.jpg'))

                    os.makedirs(fw_output_seq_path, exist_ok=True)
                    os.makedirs(fwt_output_seq_path, exist_ok=True)
                    os.makedirs(bw_output_seq_path, exist_ok=True)

                    images = sorted(images)

                    if mode == 'fw':
                        for imfile1, imfile2 in tzip(images[:-1], images[1:], leave=False, desc=f'dataset {dataset} mode {mode} challenge {cha} sequence {seq}'):
                            fw_out_imfile1 = imfile1.replace(dataset_root, fw_flow_data_root).replace(".jpg", '.png')
                            if os.path.exists(fw_out_imfile1):
                                continue
                            image1 = load_image(imfile1, device)
                            image2 = load_image(imfile2, device)

                            padder = InputPadder(image1.shape)
                            image1, image2 = padder.pad(image1, image2)

                            fw_flow_low, fw_flow_up = model(image1, image2, iters=20, test_mode=True)

                            viz(image1, fw_flow_up, fw_out_imfile1)

                    elif mode == 'bw':
                        for imfile_p, imfile_c in tzip(images[:-1], images[1:], leave=False, desc=f'dataset {dataset} mode {mode} challenge {cha} sequence {seq}'):
                            bw_out_imfile1 = imfile_c.replace(dataset_root, bw_flow_data_root).replace(".jpg", '.png')
                            if os.path.exists(bw_out_imfile1):
                                continue
                            image_p = load_image(imfile_p, device)
                            image_c = load_image(imfile_c, device)

                            padder = InputPadder(image_p.shape)
                            image_p, image_c = padder.pad(image_p, image_c)

                            bw_flow_low, bw_flow_up = model(image_c, image_p, iters=20, test_mode=True)
                            viz(image_c, bw_flow_up, bw_out_imfile1)

                    elif mode == 'fw_t-1->t':
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



if __name__ == '__main__':
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

    generate(args)

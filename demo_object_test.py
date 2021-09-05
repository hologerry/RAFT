import argparse
import glob
import os


import cv2
import numpy as np
from tqdm.contrib import tzip
import torch
from PIL import Image
from skimage.io import imsave

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder


DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, out_imfile):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    out_filename = out_imfile.replace(".jpg", '.png')
    imsave(out_filename, flo)
    # img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.savefig(out_imfile)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2, 1, 0]]/255.0)
    # cv2.waitKey()


def demo(args):

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    os.makedirs(args.fw_output_path, exist_ok=True)
    os.makedirs(args.bw_output_path, exist_ok=True)

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
            glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)

        if args.stage == 'fw':
            for imfile1, imfile2 in tzip(images[:-1], images[1:]):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                fw_flow_low, fw_flow_up = model(image1, image2, iters=args.iters, test_mode=True)

                fw_out_imfile1 = imfile1.replace(args.path, args.fw_output_path)
                viz(image1, fw_flow_up, fw_out_imfile1)

        elif args.stage == 'bw':
            for imfile_p, imfile_c in (tzip(images[:-1], images[1:])):
                image_p = load_image(imfile_p)
                image_c = load_image(imfile_c)

                padder = InputPadder(image_p.shape)
                image_p, image_c = padder.pad(image_p, image_c)

                bw_flow_low, bw_flow_up = model(image_c, image_p, iters=args.iters, test_mode=True)

                bw_out_imfile1 = imfile_c.replace(args.path, args.bw_output_path)
                viz(image_c, bw_flow_up, bw_out_imfile1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default='models/raft-things.pth')
    parser.add_argument('--path', help="dataset for evaluation", default='/D_data/Seg/object_test_raw_data/frames')
    parser.add_argument('--stage', help="forward or backward", default='fw')
    parser.add_argument('--fw_output_path', help="output path for evaluation")
    parser.add_argument('--bw_output_path', help="output path for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--iters', type=int, help="iterations for raft", default=10)
    parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
    args = parser.parse_args()
    args.fw_output_path = args.path.replace('object_test_raw_data/frames', f'object_test_raft_output_fw_iter{args.iters}/things')
    args.bw_output_path = args.path.replace('object_test_raw_data/frames', f'object_test_raft_output_bw_iter{args.iters}/things')

    demo(args)

import os

import matplotlib.pyplot as plt
from cv2 import imread
from genericpath import exists
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import trange

visualization_path = 'visualization'

real_image_root_path = '/D_data/Seg/data/object_test/img'

flow_img_root = '/D_data/Seg/'
flow_result_folders = ['object_test_raft_output_fw_iter1',
                       'object_test_raft_output_fw_iter5',
                       'object_test_raft_output_fw_iter10',
                       'object_test_raft_output_fw']


labels = ['image', 'image next', '1 iter fw', '1 iter bw', '5 iters fw', '5 iters bw', '10 iters fw', '10 iters bw', '20 iters fw', '20 iters bw']

os.makedirs(visualization_path, exist_ok=True)


for i in trange(2, 2000):
    real_image_path = os.path.join(real_image_root_path, f"out-{i:05d}.jpg")
    real_image_path_next = os.path.join(real_image_root_path, f"out-{i+1:05d}.jpg")
    fw_flow_img_paths = [os.path.join(flow_img_root, flow_path, 'things', f"out-{i:05d}.png") for flow_path in flow_result_folders]
    bw_flow_img_paths = [os.path.join(flow_img_root, flow_path.replace('fw', 'bw'), 'things', f"out-{i:05d}.png") for flow_path in flow_result_folders]
    real_img = [imread(real_image_path), imread(real_image_path_next)]
    fw_flow_imgs = [imread(flow_img_path) for flow_img_path in fw_flow_img_paths]
    bw_flow_imgs = [imread(flow_img_path) for flow_img_path in bw_flow_img_paths]
    flow_imgs = []
    for fw, bw in zip(fw_flow_imgs, bw_flow_imgs):
        flow_imgs.append(fw)
        flow_imgs.append(bw)

    imgs = real_img + flow_imgs

    # plt.axis('off')
    fig = plt.figure(figsize=(10., 10.))

    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(5, 2),  # creates 2x2 grid of axes
                     axes_pad=0.4,  # pad between axes in inch.
                     aspect=True,
                    )

    for ax, im, l in zip(grid, imgs, labels):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(l)
        ax.axis('off')

    out_path = os.path.join(visualization_path, f"out-{i:05d}.jpg")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


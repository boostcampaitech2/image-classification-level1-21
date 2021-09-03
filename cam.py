import os
import random
import re
from importlib import import_module
from pathlib import Path
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model.models import Customresnet50
from dataset.mask_base_dataset import MaskBaseDataset
from dataset.split_by_profile_dataset import SplitByProfileDataset

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def denormalize_img(input_img, mean, std):
    plot_img = input_img.clone()

    plot_img[:, 0, :, :] = plot_img[:, 2, :, :] * std[2] + mean[2]
    plot_img[:, 1, :, :] = plot_img[:, 1, :, :] * std[1] + mean[1]
    plot_img[:, 2, :, :] = plot_img[:, 0, :, :] * std[0] + mean[0]
    plot_img = transforms.functional.to_pil_image(plot_img[0])

    return plot_img


def create_cam(save_to, data_dir, model_dir, args, rows: int = 3, cols: int = 3):

    device = torch.device("cuda")
    res50 = Customresnet50(18)
    print(res50)

    state_dict = torch.load(model_dir, map_location = device)
    for key in list(state_dict.keys()):
        state_dict[key.replace('res50.', 'model.')] = state_dict.pop(key)

    res50.load_state_dict(state_dict)

    train_dataset = SplitByProfileDataset(data_dir)

    mean = (0.548, 0.504, 0.479)
    std  = (0.237, 0.247, 0.246)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset.set_transform(transform)
    train_loader = DataLoader(train_dataset, batch_size = rows * cols, shuffle = True)

    target_layer = res50.model.fc
    cam = GradCAM(model=res50, target_layer=target_layer, use_cuda=True)

    res50.eval()

    for input_img, label in train_loader:

        input_img = input_img.to(device)
        print(input_img.shape)

        target_category = label

        score = res50(input_img.clone())
        _, pred = score.max(dim=1)
        pred_label = pred.cpu().numpy()

        plot_img = denormalize_img(input_img.clone(), mean, std)
        grayscale_cam = cam(input_tensor=input_img, target_category=pred_label)
        
        fig, axes = plt.subplots(rows, cols)

        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                visualization = show_cam_on_image(np.float32(plot_img[idx]) / 255, grayscale_cam[idx, :], use_rgb=True)

                axes[r][c].set_title("true: " + str(label[idx]) + " // " + "pred: " + str(pred_label[idx]))
                axes[r][c].imshow(visualization)
                axes[r][c].axis('off')
        
        fig.savefig(save_to, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './saved/model.pt'))
    parser.add_argument("--save_to", type=str, default="cam.png", help='save cam image to an image file')

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    save_to = args.save_to

    create_cam(save_to, data_dir, model_dir, args)
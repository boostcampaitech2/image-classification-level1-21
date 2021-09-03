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

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def create_cam(img_dir, data_dir, model_dir, args, rows: int = 3, cols: int = 3):

    device = torch.device("cuda")
    model = Customresnet50(18)
    print(model)
    model.load_state_dict(torch.load(model_dir, map_location=device))

    train_dataset = SplitByProfileDataset(data_dir)
    train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True)

    target_layer = model.model.fc
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)

    mean = (0.548, 0.504, 0.479)
    std  = (0.237, 0.247, 0.246)

    target_category = 10 # 확인해보고 싶은 class (0~17)

    model.eval()
    fig, axes = plt.subplots(rows, cols)

    for r in range(rows):
        for c in range(cols):

            data_idx = np.random.randint(len(train_loader.dataset))
            input_img = train_loader.dataset[data_idx][0].unsqueeze(dim=0).to(device)
            label = train_loader.dataset[data_idx][1]

            score = model(input_img.clone())
            _, pred = score.max(dim=1)
            pred_label = pred.cpu().numpy()[0]

            if pred_label == target_category:
                break

            if pred_label == label:
                plt.title(str(label)+" // "+str(pred_label)+' (O)')
            else:
                plt.title(str(label)+" // "+str(pred_label)+' (X)')
            
            plot_img = train_loader.dataset[data_idx][0]
            plot_img[0, :, :] = plot_img[2, :, :] * std[2] + mean[2]
            plot_img[1, :, :] = plot_img[1, :, :] * std[1] + mean[1]
            plot_img[2, :, :] = plot_img[0, :, :] * std[0] + mean[0]
            plot_img = transforms.functional.to_pil_image(plot_img)
        
            grayscale_cam = cam(input_tensor=input_img.clone().detach(), target_category=target_category)
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(np.float32(plot_img) / 255, grayscale_cam, use_rgb=True)
        
            axes[r][c].imshow(visualization)
            axes[r][c].axis('off')
    
    plt.savefig(img_dir, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './saved/model.pt'))
    parser.add_argument("--save_to", type=str, default="cam.png", help='save cam image to an image file')

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    img_dir = args.save_to

    create_cam(img_dir, data_dir, model_dir, args)
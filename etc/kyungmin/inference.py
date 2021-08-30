import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

import argparse

from model.models import *
from tqdm import tqdm

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='inference_path')
    args.add_argument('-i', '--image_dir', default='./eval', type=str, help='eval image folder path')
    args.add_argument('-l', '--load_state', default='./saved/model.pt', type=str, help='load state file path (Default ./saved/model.pt)')
    args.add_argument('-m', '--model', default='PretrainedResnet', type=str, help='choose model')
    args = args.parse_args()

    # meta 데이터와 이미지 경로를 불러옵니다.
    test_dir = args.image_dir
    load_file_path = args.load_state
    model_name = args.model

    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    transform = transforms.Compose([
        Resize((224, 224), Image.BILINEAR),
        #transforms.CenterCrop((300)),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ])
    dataset = TestDataset(image_paths, transform)

    loader = DataLoader(
        dataset,
        shuffle=False
    )

    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
    device = torch.device('cuda')
    #model = ModelList.parse_model(model_name)(xdim=[3,512,384], ksize=3, cdims=[], hdims=[], ydim=18).to(device)
    num_classes = 18
    model_name = "ViT"
    model = ModelList.parse_model(model_name)(num_classes).to(device)
    print(model)
    #model.init_param()
    model.load_state_dict(torch.load(load_file_path))
    model.eval()

    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    all_predictions = []
    for images in tqdm(loader):
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
    print('test inference is done!')

import os
import math
import random
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T

device = torch.device("cuda")
model = torch.load()

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

# meta 데이터와 이미지 경로를 불러옵니다.
test_dir = "input/data/eval"
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'images')

# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
transform = T.Compose([
    T.CenterCrop((384, 384)),
    T.Resize((96, 96), Image.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
])
dataset = TestDataset(image_paths, transform)

loader = DataLoader(
    dataset,
    shuffle=False
)

# 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
device = torch.device('cuda')
model.eval()

# 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
all_predictions = []
for images in loader:
    with torch.no_grad():
        images = images.to(device)
        out, _ = model(images)
        m_lbl = out[:, 0:3].argmax(dim=-1)
        g_lbl = out[:, 3] > 0
        a_lbl = out[:, 4:7].argmax(dim=-1)
        
        pred = m_lbl * 6 + g_lbl * 3 + a_lbl
        # pred에는 label을 넣어주거나, 
        # age, mask, gender 개별적으로 구했을 경우
        # 위처럼 계산해줘도 됩니다

        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
print('test inference is done!')

# Confirm Test Result
data_dir = os.path.join(test_dir, "submission.csv")
results = pd.read_csv(data_dir)
results.head()

# 출력할 행, 열의 개수를 지정합니다
row = 3
col = 6

# 출력할 label을 지정합니다
ans = 17
start_idx = 0
corrects = 0

# subset을 구한 뒤, 랜덤한 순서로 섞습니다
subset = results.loc[results["ans"] == ans, :]
subset = subset.sample(frac=1).reset_index(drop = True)
subset.head()

# 0을 입력하면 while 루프가 종료됩니다...
while True:
    fig, axes = plt.subplots(row, col, figsize=(10, 5))
    for i in range(row):
        for j in range(col):
            idx = i * col + j
            img_dir = os.path.join(test_dir, "images", subset.ImageID.iloc[idx])
            img = Image.open(img_dir)
            axes[i][j].imshow(img)
            axes[i][j].set_title(str(subset.ans.iloc[idx]))
            axes[i][j].xaxis.set_visible(False)
            axes[i][j].yaxis.set_visible(False)
    plt.show()
    response = input("Continue? Type # of correct images! ")
    if response == "0":
        break
    else:
        corrects += int(response)
        start_idx += 1
print(corrects)
print(start_idx)
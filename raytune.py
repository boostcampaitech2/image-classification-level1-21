from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, transforms
import os

from tqdm import tqdm
from dataset import MaskSplitByProfileDataset

from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import CLIReporter
from ray import tune
import ray


def load_data(data_dir) :
    transform = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset_module = getattr(import_module("dataset"), "MaskSplitByProfileDataset")  # default: BaseAugmentation
    image_datasets = dataset_module(
                            data_dir=data_dir
                            )
    
    image_datasets.set_transform(transform)
    train_datasets, test_datasets = image_datasets.split_dataset()    

    return train_datasets, test_datasets

def training( config ):
    # 통제 변인
    model_module = getattr(import_module("model"), "Customresnet50")
    target_model = model_module(
        num_classes=18
    )
    target_model = torch.nn.DataParallel(target_model)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_model.to(device)

    # 조작 변인 
    # BatchSize & LR
    data_dir = "../../../res_baseline/train/images"

    train_datasets, test_datasets = load_data(data_dir)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=config['BatchSize'],shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=config['BatchSize'],shuffle=True, num_workers=4)
    
    dataloaders = {
        "train" : train_dataloader,
        "test"  : test_dataloader
    }
    
    # optim 
    optimizer = torch.optim.SGD(target_model.parameters(), lr=config["LearningRate"],momentum=0.85)



    loss_fn = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    

    ### 학습 코드 시작
    best_test_accuracy = 0.
    best_test_loss = 0.

    for epoch in tqdm(range(10)):
        for phase in ["train", "test"]:
            running_loss = 0.
            running_acc = 0.
            if phase == "train":
                target_model.train() # 네트워크 모델을 train 모드로 두어 gradient을 계산하고, 여러 sub module (배치 정규화, 드롭아웃 등)이 train mode로 작동할 수 있도록 함
            elif phase == "test":
                target_model.eval() # 네트워크 모델을 eval 모드 두어 여러 sub module들이 eval mode로 작동할 수 있게 함
            
            for ind, (images, labels) in enumerate(dataloaders[phase]):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # parameter gradient를 업데이트 전 초기화함

                with torch.set_grad_enabled(phase == "train"): # train 모드일 시에는 gradient를 계산하고, 아닐 때는 gradient를 계산하지 않아 연산량 최소화
                    logits = target_model(images)
                    _, preds = torch.max(logits, 1) # 모델에서 linear 값으로 나오는 예측 값 ([0.9,1.2, 3.2,0.1,-0.1,...])을 최대 output index를 찾아 예측 레이블([2])로 변경함  
                    loss = loss_fn(logits, labels)

                    if phase == "train":
                        loss.backward() # 모델의 예측 값과 실제 값의 CrossEntropy 차이를 통해 gradient 계산
                        optimizer.step() # 계산된 gradient를 가지고 모델 업데이트

                running_loss += loss.item() * images.size(0) # 한 Batch에서의 loss 값 저장
                running_acc += torch.sum(preds == labels.data) # 한 Batch에서의 Accuracy 값 저장



            # 한 epoch이 모두 종료되었을 때,
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_acc / len(dataloaders[phase].dataset)

            if phase == "test" : 
                scheduler.step(epoch_loss)

            if phase == "test" and best_test_accuracy < epoch_acc: # phase가 test일 때, best accuracy 계산
                best_test_accuracy = epoch_acc
            if phase == "test" and best_test_loss < epoch_loss: # phase가 test일 때, best loss 계산
                best_test_loss = epoch_loss
        # epoch 종료
    tune.report(accuracy=best_test_accuracy.item(), loss=best_test_loss)
    
if __name__ == "__main__" :

    config_space = {
        "LearningRate" : tune.uniform(0.0001, 0.001),
        "BatchSize" : tune.choice([16,32,64]),
    }

    optim = HyperOptSearch( # HyperOptSearch 통해 Search를 진행합니다. 
        metric='accuracy',  # hyper parameter tuning 시 최적화할 metric을 결정합니다. 본 실험은 test accuracy를 target으로 합니다
        mode="max",         # target objective를 maximize 하는 것을 목표로 설정합니다
    )
    

    NUM_TRIAL = 3 # Hyper Parameter를 탐색할 때에, 실험을 최대 수행할 횟수를 지정합니다.

    reporter = CLIReporter( # jupyter notebook을 사용하기 때문에 중간 수행 결과를 command line에 출력하도록 함
        parameter_columns=["LearningRate","BatchSize"],
        metric_columns=["accuracy", "loss"])

    ray.shutdown() # ray 초기화 후 실행

    analysis = tune.run(
        training,
        config=config_space,
        search_alg=optim,
        progress_reporter=reporter,
        num_samples=NUM_TRIAL,
        resources_per_trial={'gpu': 1}, # Colab 런타임이 GPU를 사용하지 않는다면 comment 처리로 지워주세요
    )

    
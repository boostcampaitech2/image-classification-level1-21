import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


from dataset import MaskBaseDataset
from loss import create_criterion


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    dataset = dataset_module(
        data_dir=data_dir,
    )
#     num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=0,# multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers= 0,# multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        # num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
        momentum = 0.8 # if SGD Momentum
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    # scheduler = ReduceLROnPlateau(optimizer, patience = 3)


# -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(tqdm(train_loader)):

            inputs,gender,mask,age = train_batch
            inputs = inputs.to(device)
            gender = gender.to(device)
            mask = mask.to(device)
            age = age.to(device)

            optimizer.zero_grad()

            #outs = model(inputs)
            x1,x2,x3 = model(inputs)

            # preds = torch.argmax(outs, dim=-1)
            y1 = torch.argmax(x1,dim=-1)
            y2 = torch.argmax(x2,dim=-1)
            y3 = torch.argmax(x3,dim=-1)

            loss1 = criterion(x1, gender)
            loss2 = criterion(x2, mask)
            loss3 = criterion(x3, age)

            loss = loss1 + loss2+ loss3

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            gender_matches = (y1 == gender).sum().item()
            mask_matches = (y2 == mask).sum().item()
            age_matches = (y3 == age).sum().item()


            # print(f"성별{gender_matches}, 마스크 {mask_matches}, 나이 {age_matches}")
            matches = (gender_matches + mask_matches + age_matches)/3

            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch+1}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        
        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss1_items = []
            val_acc1_items = []

            val_loss2_items = []
            val_acc2_items = []

            val_loss3_items = []
            val_acc3_items = []

            figure = None
            for val_batch in val_loader:
                inputs,gender,mask,age = val_batch
                inputs = inputs.to(device)
                gender = gender.to(device)
                mask = mask.to(device)
                age = age.to(device)

                x1,x2,x3 = model(inputs)
                y1 = torch.argmax(x1,dim=-1)
                y2 = torch.argmax(x2,dim=-1)
                y3 = torch.argmax(x3,dim=-1)



                loss1_item = criterion(x1, gender).item()
                loss2_item = criterion(x2, mask).item()
                loss3_item = criterion(x3, age).item()

                acc1_item = (y1 == gender).sum().item()
                acc2_item = (y2 == mask).sum().item()
                acc3_item = (y3 == mask).sum().item()

                val_loss1_items.append(loss1_item)
                val_loss2_items.append(loss2_item)
                val_loss3_items.append(loss3_item)

                val_acc1_items.append(acc1_item)
                val_acc2_items.append(acc2_item)
                val_acc3_items.append(acc3_item)


            val_loss1 = np.sum(val_loss1_items) / len(val_loader)
            val_acc1 = np.sum(val_acc1_items) / len(val_set)

            val_loss2 = np.sum(val_loss2_items) / len(val_loader)
            val_acc2 = np.sum(val_acc2_items) / len(val_set)

            val_loss3 = np.sum(val_loss3_items) / len(val_loader)
            val_acc3 = np.sum(val_acc3_items) / len(val_set)

            print(f"[VAL]-loss 성별: {val_loss1:4.2}, 마스크: {val_loss2:4.2}, 나이: {val_loss3:4.2}")
            print(f"[VAL]-acc 성별: {val_acc1:4.2%}, 마스크: {val_acc2:4.2%}, 나이: {val_acc3:4.2%}")

            val_loss = (val_loss1 + val_loss2 + val_loss3)/3
            val_acc = (val_acc1 + val_acc2 + val_acc3)/3

            best_val_loss = min(best_val_loss, val_loss)

            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[VAL] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"[VAL] best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=7, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[256, 256], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)

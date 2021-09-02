import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset


# def load_model(saved_model, num_classes, device):
def load_model(saved_model,device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        # num_classes=num_classes
    )
    model_path = os.path.join(saved_model, 'best.pth')
    # model_path = os.path.join(saved_model, 'last.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # num_classes = MaskBaseDataset.num_classes  # 18
    # model = load_model(model_dir, num_classes, device).to(device)
    model = load_model(model_dir, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            out1,out2,out3 = model(images)
            pred1 = out1.argmax(dim=-1)
            pred2 = out2.argmax(dim=-1)
            pred3 = out3.argmax(dim=-1)
            pred = pred1*3+pred2*6+pred3
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    # 출력 결과 확인 한 후에 주석처리 해제 후에 csv 생성
    # info.to_csv(os.path.join(output_dir, f'submission_0901_02.csv'), index=False)
    print(f'Inference Done! Compare the first ten!!')
    print(preds[:10])
    print([14,5,14,14,12,0,8,4,5,3])
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/submission/'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)

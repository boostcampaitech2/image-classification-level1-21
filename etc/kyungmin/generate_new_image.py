import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch.models.mtcnn import MTCNN
import os, cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=True, device=device)
new_img_dir = './train/new_images'
#new_val_dir = './train/new_vaild'
img_path = '../input/data/train/images'

cnt = 0
img_path_list = [f for f in os.listdir(img_path) if not f.startswith('.')]

for paths in tqdm(img_path_list):
    if paths[0] == '.': continue
    
    sub_dir = os.path.join(img_path, paths)
    
    new_img_path = os.path.join(new_img_dir, paths)
    #new_val_path = os.path.join(new_val_dir, paths)
    os.system('mkdir {}'.format(new_img_path))
    #os.system('mkdir {}'.format(new_val_path))

    for imgs in os.listdir(sub_dir):
        if imgs[0] == '.': continue
        
        img_dir = os.path.join(sub_dir, imgs)
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        #mtcnn 적용
        boxes,probs = mtcnn.detect(img)
        #        print(new_img_path, '/', imgs)
        #        print(boxes)
        #        print()
        # boxes 확인
        #if len(probs) > 1: 
        #    print(boxes)
        if not isinstance(boxes, np.ndarray):
            #print('Nope!', new_img_path, '/', imgs)
            # 직접 crop
            img=img[100:400, 50:350, :]
        elif abs(boxes[0,0]-boxes[0, 2]) < 60:
            img=img[100:400, 50:350, :]
        elif abs(boxes[0,1]-boxes[0, 3]) < 60:
            img=img[100:400, 50:350, :]
        # boexes size 확인
        else:
            xmin = int(boxes[0, 0])-30
            ymin = int(boxes[0, 1])-30
            xmax = int(boxes[0, 2])+30
            ymax = int(boxes[0, 3])+30
            
            if xmin < 0: xmin = 0
            if ymin < 0: ymin = 0
            if xmax > 384: xmax = 384
            if ymax > 512: ymax = 512
            
            img = img[ymin:ymax, xmin:xmax, :]
            
        cnt += 1
        #if np.random.rand() < 0.9:
        plt.imsave(os.path.join(new_img_path, imgs), img)
        #else:
        #    plt.imsave(os.path.join(new_val_path, imgs), img)
        
print(cnt)

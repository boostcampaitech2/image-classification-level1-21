import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch

class CustomDataset(Dataset):
    def __init__(self, path, csv):
        self.path = path
        self.data = pd.read_csv(self.path + csv)

        self.data['ans'] = 0

        self.data.loc[self.data.gender == 'male', 'ans'] = 0
        self.data.loc[self.data.gender == 'female', 'ans'] = 3
        self.data.loc[self.data.age < 30, 'ans'] += 0
        self.data.loc[(self.data.age >= 30), 'ans'] += 1
        self.data.loc[self.data.age >= 60, 'ans'] += 1
        
        self.image_path_list = self.data.path.values        
        return
    
    def __len__(self):
        return len(self.data) * 7
    
    def __getitem__(self, idx):
        image_path = self.path + 'images/' + self.image_path_list[idx%2700] +'/'
        file_list = [f for f in os.listdir(image_path) if not f.startswith('.')]  
        image_file = file_list[idx%7]

        y = self.data.ans[idx%2700]
        #if image_file.find('mask') != -1:
        #    y = self.data.ans[idx%2700] + 0
        if image_file.find('incorrect') != -1:
            y = self.data.ans[idx%2700] + 6
        elif image_file.find('normal') != -1:
            y = self.data.ans[idx%2700] + 12
        img = Image.open(image_path+image_file)
        X = transforms.ToTensor()(img)
        y = torch.tensor(y)
        return X, y

if __name__ == "__main__":
    print([f for f in os.listdir('./train/images') if not f.startswith('.')] )
    mydataset = CustomDataset('./train/', 'train.csv')
    #print(mydataset.__getitem__(0))
    BATCH_SIZE = 9
    train_iter = torch.utils.data.DataLoader(mydataset,
                         batch_size = BATCH_SIZE,
                         shuffle = True,
                         num_workers = 0)
    for i in train_iter:
        pass
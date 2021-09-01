import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN

# baesline 코드에 face-crop만 추가했습니다.
# face_crop_dataset 과 동일한 방법으로 face_crop합니다. 
class TestDataset(Dataset):
    def __init__(self, img_paths, resize, face_crop=False, face_model=None, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            transforms.Resize(resize, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.face_model = face_model
        self.face_crop = face_crop

    def face(self, img) :
        mtcnn = MTCNN(image_size=384)
        img = np.asarray(img)[:,:,::-1]

        bbox , prob = mtcnn.detect(img)
        

        h,w = img.shape[:2]
        face,isFace = None, True
        for i,p in enumerate(prob) :
            if p is None or p < 0.5 : 
                isFace = False
                break
            
            d = 40
            x1 = int(bbox[i][0])-d; y1 = int(bbox[i][1])-d
            x2 = int(bbox[i][2])+d; y2 = int(bbox[i][3])+d
            
            x1 = max(0,x1); y1 = max(0,y1)
            x2 = min(w,x2); y2 = min(h,y2)
            
            face = img[y1:y2,x1:x2,:]
            
        if not isFace or face.shape[0] < 200 or face.shape[0] < 200 :
            annotation = self.face_model.predict_jsons(img)[0]
            if annotation['score'] == -1 :
                return Image.fromarray(img[h//2-150:h//2+150,w//2-150:w//2+150,:][:,:,::-1])
            
            d = 40
            x1 = int(annotation['bbox'][0])-d; y1 = int(annotation['bbox'][1])-d
            x2 = int(annotation['bbox'][2])+d; y2 = int(annotation['bbox'][3])+d

            x1 = max(0,x1); y1 = max(0,y1)
            x2 = min(w,x2); y2 = min(h,y2)
            
            face = img[y1:y2,x1:x2,:]

        return Image.fromarray(face[:,:,::-1]) 
    
    def __getitem__(self, index):
        print(self.img_paths[index])
        image = Image.open(self.img_paths[index])

        if self.face_crop :
            image = self.face(image)

        if self.transform:
            image = self.transform(image)
        return self.img_paths[index], image

    def __len__(self):
        return len(self.img_paths)
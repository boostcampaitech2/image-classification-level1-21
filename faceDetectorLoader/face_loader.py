from torchvision import datasets
import cv2
import numpy as np
from PIL import Image

# Dataset 에서 face 만 crop 해서 load 해주는 데이터셋입니다.
# face detector 못하면 전체 이미지를 load 해줍니다.
# ImageFolder 기준이라 아래 수정하실 땐 faceDectector 함수랑 __getitem__ 보시고 쓰시면 됩니다.

# 같이 올려드린 파일명으로 불러주시면 됩니다. ssd 기반 facedetecor
# model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
# config = "deploy.prototxt"

class FaceDatasets(datasets.ImageFolder) :
    def __init__(self,root, dnn_model, dnn_config, transform) :
        super(FaceDatasets,self).__init__(root)
        self.transform = transform
        self.net = cv2.dnn.readNet(dnn_model,dnn_config)
    
    def faceDectector(self,frame) :
        frame = np.asarray(frame)
        blob = cv2.dnn.blobFromImage(frame,1,(300,300),(104,177,123))
        self.net.setInput(blob)
        out = self.net.forward()
        detect = out[0,0,:,:]

        h,w = frame.shape[:2]
        face = None
        for d in detect :
            _,_,c,x1,y1,x2,y2 = d

            if c < 0.5 : break
            left,right = 30,20
            x1 = int(x1 * w); y1 = int(y1 * h)
            x2 = int(x2 * w); y2 = int(y2 * h)
            
            face = frame[y1-left:y2+right,x1-left:x2+right,:]
        if face is None :
            face = frame
            
        face = Image.fromarray(face)
        
        return face
    
    def __getitem__(self,index) : 
        path, target = self.samples[index]
        sample = self.loader(path)

        face = self.faceDectector(sample)
        
        
        if self.transform is not None:
            face = self.transform(face)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return face, target        
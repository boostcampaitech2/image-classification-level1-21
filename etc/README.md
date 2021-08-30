# 하루 한걸 부담없이 올리시면됩니다.

## 현재 하고 있는 것 + 올린 내용.(간단히)
김상현 : MTCNN 분해 시도... 그러나 MTCNN은 neural network를 제외하고도 `.forward()` 혹은 `.detect()` 메서드에서 추가적으로 neural network output의 post-processing이 진행됨... 따라서 raw output을 활용하는 것이 쉽지는 않을듯... 그러나 학습된 conv layers들의 weight를 가져오는 방안도 고민 중!

나경훈 : 

원상혁 : mask, gender, age 나눠서 학습 후 추론하는 앙상블 모델 짜는 중입니다 데이터가 워낙 불균형해서 다시 하는중입니다...

이경민 : 

이노아 : 

손지아 : EfficientNetV2 추론만 돌렸으나 결과는 처참했고 학습은 실패했다. timm 사용법을 몰라서 그런 것 같기도 하고... 이제는 Wide ResNet과 LDNN을 보고 있다.

최민서 : 

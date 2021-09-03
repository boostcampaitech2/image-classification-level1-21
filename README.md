# TEAM 21 "BLACKJACK"
## Image Classification Leaderboard

## Model Visualization

### ResNet50

![ResNet50](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/eb0f8b97-7a0b-4c79-820f-4609daa160fe/Slide1.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210903%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210903T162851Z&X-Amz-Expires=86400&X-Amz-Signature=7435056ff740211661db26a28da7fe0a5a33e1087a7109190f9d4f314a574c34&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Slide1.png%22)

### EfficientNet b2

![EfficientNet](https://1.bp.blogspot.com/-DjZT_TLYZok/XO3BYqpxCJI/AAAAAAAAEKM/BvV53klXaTUuQHCkOXZZGywRMdU9v9T_wCLcBGAs/s1600/image2.png)

## Usage

### train
```
$ python3 train.py
```


### Inference
```
( '--batch_size' , type=int   , default=1           )
( '--resize'     , type=tuple , default=(224, 224)  )
( '--model'      , type=str   , default='EfficientResnet' )
( '--data_dir'   , type=str   )
( '--model_dir'  , type=str   )
( '--output_dir' , type=str   )
```

```
$ python3 inference.py --model Resnet50 --model_dir your/model/path.pt --data_dir /opt/ml/input/data/eval --output_dir output/
```

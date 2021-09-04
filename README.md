# TEAM 21 "BLACKJACK"

* [TEAM 21 "BLACKJACK"](https://github.com/boostcampaitech2/image-classification-level1-21#team-21-blackjack)
  * [Image Classification Leaderboard](https://github.com/boostcampaitech2/image-classification-level1-21#image-classification-leaderboard)
  * [Model Visualization](https://github.com/boostcampaitech2/image-classification-level1-21#model-visualization)
    * [ResNet50](https://github.com/boostcampaitech2/image-classification-level1-21#resnet50)
    * [EfficientNet b2](https://github.com/boostcampaitech2/image-classification-level1-21#efficientnet-b2)
  * [Usage](https://github.com/boostcampaitech2/image-classification-level1-21#usage)
    * [Train](https://github.com/boostcampaitech2/image-classification-level1-21#train)
      * [Argument](https://github.com/boostcampaitech2/image-classification-level1-21#argument)
      * [Using Argument](https://github.com/boostcampaitech2/image-classification-level1-21#using-argument)
    * [Inference](https://github.com/boostcampaitech2/image-classification-level1-21#inference)
      * [Argument](https://github.com/boostcampaitech2/image-classification-level1-21#argument-1)
      * [Using Argument](https://github.com/boostcampaitech2/image-classification-level1-21#using-argument-1)

## Image Classification Leaderboard

## Model Visualization

### ResNet50

![ResNet50](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/eb0f8b97-7a0b-4c79-820f-4609daa160fe/Slide1.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210903%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210903T162851Z&X-Amz-Expires=86400&X-Amz-Signature=7435056ff740211661db26a28da7fe0a5a33e1087a7109190f9d4f314a574c34&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Slide1.png%22)

### EfficientNet b2

![EfficientNet](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/1ad8edab-28e6-4e9f-87fb-d11679bc98c9/EfficientNet.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210904%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210904T050232Z&X-Amz-Expires=86400&X-Amz-Signature=41de44e002b52836c8122e00b1e1ff4e0bad838d8a5cc2e610afcbecc53ceaf8&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22EfficientNet.png%22)

## Usage

### Train
#### Argument
```
( '--model'      , type=str   , default='Customresnet50'    )
( '--dataset'    , type=str   , default='MaskBaseDataset'   )
( '--data_dir'   , type=str   )
```
#### Using argument
```
$ python3 train.py --model Customresnet50 --dataset SplitByProfileDataset --data_dir /your/data/dir 
```


### Inference
#### Argument
```
( '--model'      , type=str   , default='EfficientResnet'   )
( '--data_dir'   , type=str   )
( '--model_dir'  , type=str   )
```

#### Using argument
```
$ python3 inference.py --model EfficientResnet --model_dir your/model/path.pth --data_dir /your/data/dir
```

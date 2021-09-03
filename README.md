# TEAM 21 "BLACKJACK"
## Image Classification Leaderboard

## Model Visualization

### ResNet50

![ResNet50](https://raw.githubusercontent.com/boostcampaitech2/image-classification-level1-21/master/resources/resnet50.png?token=AQ6WU4Q4KETPC2ZSLXNTJFLBGI3HI)


## Usage

### train
```
$ python3 train.py
```


### Inference
```
( '--batch_size' , type=int   , default=1           )
( '--resize'     , type=tuple , default=(224, 224)  )
( '--model'      , type=str   , default='MainModel' )
( '--data_dir'   , type=str   )
( '--model_dir'  , type=str   )
( '--output_dir' , type=str   )
```

```
$ python3 inference.py --model Resnet50 --model_dir your/model/path.pt --data_dir /opt/ml/input/data/eval --output_dir output/
```

# TEAM 21 "BLACKJACK"
## Image Classification Leaderboard


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

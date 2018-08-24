
## train
```
cd src
python -m train \
  --base_model resnet18 \
  --batch_size 4 \
  --nb_epochs 20 \
  --nb_workers 1 \
  --img_dir ../data/TrainVal \
  --train_filelist ../data/train.csv \
  --test_filelist ../data/test.csv \
  --log_freq 100 \
  --checkpoint_suffix resnet101 \
  --checkpoint_dir /tmp \
  --checkpoint_freq 1
```

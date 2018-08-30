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
  --checkpoint_suffix resnet18 \
  --checkpoint_dir /tmp \
  --checkpoint_freq 1
```


### gpu
```
CUDA_VISIBLE_DEVICES=0 python -m train \
  --gpu \
  --base_model resnet101 \
  --batch_size 32 \
  --nb_epochs 20 \
  --nb_workers 4 \
  --img_dir ../data/TrainVal \
  --train_filelist ../data/train.csv \
  --test_filelist ../data/test.csv \
  --log_freq 100 \
  --checkpoint_suffix resnet101-basic \
  --checkpoint_dir /tmp \
  --checkpoint_freq 1
```

## predict
```
cd src
python -m predict \
  --base_model resnet101 \
  --batch_size 4 \
  --nb_workers 1 \
  --img_dir ../data/Public \
  --log_freq 100 \
  --model_state_path []
```

## retrieve
```
cd src
python -m retrieve \
    --feats_index_npz ../tmp/resnet101_120_trainval_feats-gap.npz \
    --feats_query_npz ../tmp/resnet101_120_public_feats-gap.npz \
    --norm \
    --k 64 \
    --index_choice flat \
    --out_fname ../tmp/resnet101_120_feats-spp_retrieve_flat.npz"
```

## knn classify
```
python -m knn_classify \
    --input_npz ../tmp/resnet101_120_feats-gap_retrieve_flat.npz" \
    --ref_csv ../data/train_val.csv" \
    --k 11 \
    --nb_picks 3 \
    --out_fname ../tmp/resnet101_120_feats-gap_retrieve_flat_knn_k=11.csv"
```

## ConMLP: MLP-based Self-Supervised Contrastive Learning for Skeleton Data Analysis and Action Recognition


### Download datasets
NTU RGB+D and NTU RGB+D 120 datasets can be obtained at [here](https://github.com/shahroudy/NTURGB-D).


### Data pre-processing
```
cd ./dataset/ntu60 # or cd ./dataset/ntu120

# Get skeleton of each performer
python get_raw_skes_data.py

# Remove the bad skeleton
python get_raw_denoised_data.py

# Transform the skeleton to the center of the first frame
python seq_transformation.py
```


### For Self-supervised training with contrastive loss function
```
python main_train.py 
--method=Self-supervised 
--model=mlp 
--dataset=NTU60CV 
--epochs=5000 
--batch_size=512
--learning_rate=0.001 
--weight_decay=0.0005
--temp=0.07 
--cosine
```

### For Supervised training with contrastive loss function
```
python main_train.py 
--method=Supervised 
--model=resnet50
--dataset=NTU60CV 
--epochs=5000 
--batch_size=512
--learning_rate=0.001 
--weight_decay=0.0005
--temp=0.07 
--cosine
```

### For Supervised training and inference with Cross-Entropy loss function
```
python main_ce.py
--model=mlp
--dataset=NTU60CV
--epochs=5000
--batch_size=512
--learning_rate=0.001
--weight_decay=0.0005
--cosine
```

### Acknowledgement
The framework is extended from the following repositories. We sincerely thank the authors for releasing the codes.

1. The framework is based on [SupCon](https://github.com/HobbitLong/SupContrast/).

2. The encoder is based on [Graph-MLP](https://github.com/yanghu819/Graph-MLP/).

3. The data extractor is based on [PSUMNet](https://github.com/skelemoa/psumnet/).

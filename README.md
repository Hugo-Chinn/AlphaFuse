:smile_cat: Welcome to AlphaFuse, this is a implementation of ***AlphaFuse: Learn ID Embeddings for Sequential
Recommendation in Null Space of Language Embeddings***

## :one:  ​ Guide for Running AlphaFuse



### :walking_man: SASRec backbone

```sh
python main.py --model=PreferDiff --sd=O --td=O --loss_type=cosine  --lamda=0.4 --w=2 --hidden_size=3072  --ab=iids
```



### :runner: DreamRec backbone

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port=12330 main.py --model=PreferDiff --sd=O --td=O --loss_type=cosine  --lamda=0.4 --w=2 --hidden_size=3072 --ab=iids
```



## :two: Best Hyperparameters

| Dataset    | learning rate | Weight Decay | lambda | w    | Embedding Size |
| ---------- | ------------- | ------------ | ------ | ---- | -------------- |
| **Sports** | 1e-4          | 0            | 0.4    | 2    | 3072           |
| **Beauty** | 1e-4          | 0            | 0.4    | 8    | 3072           |
| **Toys**   | 1e-4          | 0            | 0.6    | 6    | 3072           |



## :three: Guide for Running Baselines



```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port=12330 main.py --model=SASRec --sd=O --td=O 
```

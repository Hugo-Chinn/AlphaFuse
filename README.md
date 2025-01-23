:smile_cat: Welcome to AlphaFuse, this is a implementation of ***AlphaFuse: Learn ID Embeddings for Sequential
Recommendation in Null Space of Language Embeddings***

## :one:  ​ Guide for Running AlphaFuse



### :walking_man: SASRec backbone

```sh
# nohup python -u train.py --data ATV --random_seed 22 --model_type AlphaFuse --cuda 5 --language_model_type 3small --ID_embs_init_type zeros --hidden_dim 128 --null_dim 64 --lr 0.001 -loss_type infoNCE --neg_ratio 64  > ./log/ATV_SASRec_AlphaFuse_rs22_dim128null64_infoNCE64_lr3 2>&1 &
```

### :runner: DreamRec backbone

```sh
nohup python -u train_diffusion.py --data ATV --random_seed 22 --model_type AlphaFuse --cuda 2 --language_model_type 3small --null_thres 0.25 --hidden_dim 1536 --lr 0.00001  > ./log/ATV_DreamRec_AlphaFuse_rs22_dim1536null0.25_lr5 2>&1 &
```



## :two: Hyperparameters for Cold-Start User Settings

### :walking_man: SASRec backbone

| Dataset    | learning rate | $d_s+d_n$    | $d_n$  | $\mathbf{E}_\text{ID}$ Init |
| ---------- | ------------- | ------------ | ------ | ----------------- |
| **Movies** | 1e-3          | 128           | 64    |      zeros     |
| **Toys**   | 1e-3          | 128           | 64    |            |
| **Sports** | 1e-2          | 128           | 64    |            |

| Dataset    | learning rate | Weight Decay | lambda | w    | Embedding Size |
| ---------- | ------------- | ------------ | ------ | ---- | -------------- |
| **Sports** | 1e-4          | 0            | 0.4    | 2    | 3072           |
| **Beauty** | 1e-4          | 0            | 0.4    | 8    | 3072           |
| **Toys**   | 1e-4          | 0            | 0.6    | 6    | 3072           |



## :three: Guide for Running Baselines



```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port=12330 main.py --model=SASRec --sd=O --td=O 
```

🐡: Welcome to AlphaFuse, this is a implementation of ***AlphaFuse: Learn ID Embeddings for Sequential
Recommendation in Null Space of Language Embeddings***

## :one:  ​ Guide for Running AlphaFuse



### :walking_man: SASRec backbone

```sh
nohup python -u train.py --data ATV --random_seed 22 --model_type AlphaFuse --cuda 0 --language_model_type 3small --ID_embs_init_type zeros --hidden_dim 128 --null_dim 64 --lr 0.001 -loss_type infoNCE --neg_ratio 64  > ./log/ATV_SASRec_AlphaFuse_rs22_dim128null64_infoNCE64_lr3 2>&1 &
nohup python -u train.py --data ATG --random_seed 22 --model_type AlphaFuse --cuda 1 --language_model_type 3large --ID_embs_init_type normal --hidden_dim 128 --null_dim 64 --lr 0.001 -loss_type infoNCE --neg_ratio 64  > ./log/ATG_SASRec_AlphaFuse_rs22_dim128null64_infoNCE64_lr3 2>&1 &
nohup python -u train.py --data ASO --random_seed 22 --model_type AlphaFuse --cuda 2 --language_model_type 3large --ID_embs_init_type normal --hidden_dim 128 --null_dim 64 --lr 0.01 -loss_type infoNCE --neg_ratio 64  > ./log/ASO_SASRec_AlphaFuse_rs22_dim128null64_infoNCE64_lr2 2>&1 &
```

### :runner: DreamRec backbone

```sh
nohup python -u train_diffusion.py --data ATG --random_seed 22 --model_type AlphaFuse --cuda 1 --language_model_type 3large --null_thres 0.25 --hidden_dim 3072 --lr 0.00001  > ./log/ATG_DreamRec_AlphaFuse_rs22_dim3072null0.25_lr5 2>&1 &
nohup python -u train_diffusion.py --data ASO --random_seed 22 --model_type AlphaFuse --cuda 2 --language_model_type 3large --null_thres 0.25 --hidden_dim 3072 --lr 0.00001  > ./log/ASO_DreamRec_AlphaFuse_rs22_dim3072null0.25_lr5 2>&1 &
```
> For the Amazon Movies & TV dataset, the results of DreamRec-based AlphaFuse presented in the paper were obtained under the old framework. The actual algorithm is the same, but there are slight differences in the network architecture parameters and training parameters.
```sh
cd ATV_DreamRec
nohup python -u trainer.py --data ATV --cuda 0 --random_seed 22 --lr 0.00001 --timesteps 2000 --emb_type 3small --trans_type Null --null_thres 0.25 --emb_std 40 --beta_start 0.0001 --beta_end 0.02 --linespace 100 --beta_sche linear --w 5.0  > ./log/ATV_AlphaFuse_rs22_Null0.25_TU+ID_CF5_lr5 2>&1 &
```

## :two: Hyperparameters for Cold-Start User Settings

### :walking_man: SASRec backbone

| Dataset    | learning rate | $d_s+d_n$    | $d_n$  | $\mathbf{E}_\text{ID}$ Init |
| ---------- | ------------- | ------------ | ------ | ----------------- |
| **Movies** | 1e-3          | 128           | 64    |      zeros        |
| **Toys**   | 1e-3          | 128           | 64    |      normal       |
| **Sports** | 1e-2          | 128           | 64    |      normal       |

### :runner: DreamRec backbone

| Dataset    | learning rate | langauge API | $d_l$  | threshold    | $\mathbf{E}_\text{ID}$ Init |
| ---------- | ------------- | ------------ | ------ | ---- | -------------- |
| **Sports** | 1e-5          |  text-embedding-3-small    | 1536    | 0.25    | normal          |
| **Beauty** | 1e-5          | text-embedding-3-large     | 3072    | 0.25    | normal           |
| **Toys**   | 1e-5          | text-embedding-3-large     | 3072    | 0.25    | normal           |

## :three: Hyperparameters for Long-tail Settings following LLM-ESR

### :walking_man: SASRec backbone

| Dataset    | learning rate | $d_s+d_n$    | $d_n$  | $\mathbf{E}_\text{ID}$ Init |
| ---------- | ------------- | ------------ | ------ | ----------------- |
| **Yelp**   | 1e-4          | 128           | 64    |      zeros       |
| **Fashion**| 1e-4          | 128           | 64    |      zeros       |
| **Beauty** | 1e-4          | 128           | 64    |      zeros        |

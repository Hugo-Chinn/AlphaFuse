import os
import time
import torch
import random
import numpy as np
import pandas as pd
import argparse
import logging
import pickle
from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import time as time
from datetime import datetime

#from Sdiffusion import *
from models.diffusion import *
from models.backbone_DreamRec import DreamRec,MoRec, WhitenRec, UniSRec, RLMRec, LLMESR, LLMInit, iDreamRec, AlphaFuse
from utils import evaluate_diff

class SeqDataset(Dataset):
    def __init__(self, data):
        self.seq_data = [torch.tensor(seq, dtype=torch.long) for seq in data['seq']]
        self.len_seq_data = [torch.tensor(len_seq, dtype=torch.long) for len_seq in data['len_seq']]
        self.next_data = [torch.tensor(next_val, dtype=torch.long) for next_val in data['next']]

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        return {'seq': self.seq_data[idx], 'len_seq': self.len_seq_data[idx], 'next': self.next_data[idx]}

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

logging.getLogger().setLevel(logging.INFO)

def setup_seed(seed): 
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")
    parser.add_argument('--random_seed', type=int, default=22)
    ### training settings
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--lr_delay_rate', type=float, default=0.99)
    parser.add_argument('--lr_delay_epoch', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='ATV')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--cuda', type=int, default=7,
                        help='cuda device.')
    parser.add_argument('--l2_decay', type=float, default=0,
                        help='l2 loss reg coef.')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='beta for addtional loss function')
    ### DreamRec backbone settings
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    ### diffusion params
    parser.add_argument('--timesteps', type=int, default=2000,
                        help='timesteps for diffusion')
    parser.add_argument('--linespace', type=int, default=100,
                        help='linespace of DDIM sampling')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='beta end of diffusion')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                        help='beta start of diffusion')
    parser.add_argument('--beta_sche', type=str, default='linear',
                        help='')
    parser.add_argument('--w', type=float, default=5.0,
                        help='classifier-free guidance strength ')
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    ### languange embeddings settings
    parser.add_argument('--language_model_type', default="3small", type=str)
    parser.add_argument('--language_embs_scale', default=40, type=int)
    ### ID embeddings settings
    parser.add_argument('--hidden_dim', type=int, default=1536,
                        help='Number of hidden factors, i.e., Item embedding size.')
    parser.add_argument('--ID_embs_init_type', type=str, default="normal")
    ### model selection
    parser.add_argument('--model_type', type=str, default="UniSRec")
    parser.add_argument('--SR_aligement_type', type=str, default="con")

    # AlphaFuse
    parser.add_argument('--null_thres', type=float, default=None,)
    parser.add_argument('--null_dim', type=int, default=64,)
    parser.add_argument('--item_frequency_flag', type=str2bool, default=False)
    parser.add_argument('--standardization', type=str2bool, default=True)
    parser.add_argument('--cover', type=str2bool, default=False)
    parser.add_argument('--ID_space', type=str, default="singular")
    parser.add_argument('--inject_space', type=str, default="singular")

    return parser.parse_args()


#@profile
def main():

    args = parse_args()
    setup_seed(args.random_seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    key_words = vars(args)
    data_directory = '../ours_DiT/data/' + args.data
    model_directory = './saved/' +args.data

    key_words["language_embs_path"] = data_directory

    if args.model_type == "DreamRec":
        model = DreamRec(device, **key_words).to(device)
    if args.model_type == "iDreamRec":
        model = iDreamRec(device, **key_words).to(device)
    elif args.model_type == "MoRec":
        model = MoRec(device, **key_words).to(device)
    elif args.model_type == "WhitenRec":
        model = WhitenRec(device, **key_words).to(device)
    elif args.model_type == "UniSRec":
        model = UniSRec(device, **key_words).to(device)
    elif args.model_type == "LLMInit":
        model = LLMInit(device, **key_words).to(device)
    elif args.model_type == "RLMRec":
        model = RLMRec(device, **key_words).to(device)
    elif args.model_type == "LLMESR":
        model = LLMESR(device, **key_words).to(device)
    elif args.model_type == "AlphaFuse":
        model = AlphaFuse(device, **key_words).to(device)

    diff = diffusion(
        args.timesteps, 
        args.beta_start, 
        args.beta_end, 
        args.beta_sche, 
        args.w, 
        args.linespace)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)

    # 计算参数量
    total_params, trainable_params = count_parameters(model)

    # 输出结果
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

    print(key_words)

    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    val_data = pd.read_pickle(os.path.join(data_directory, 'val_data.df'))
    test_data = pd.read_pickle(os.path.join(data_directory, 'test_data.df'))

    train_dataset = SeqDataset(train_data)
    val_dataset = SeqDataset(val_data)
    test_dataset = SeqDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size))
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size))
    test_loader = DataLoader(test_dataset, batch_size=int(args.batch_size))
    
    epoch_num = len(train_loader)
    best_ndcg20 = 0  # 初始化最佳验证集指标为0
    patience = 50  # 设置容忍次数
    counter = 0  # 计数器
    print("Loading data & model is done.")
    t0 = time.time()
    model.eval()
    val_ndcg20 = evaluate_diff(model, diff, val_loader, device)
    t1 = time.time() - t0
    print("\n using ",t1, "s ", "Eval Time Cost.")
    ''''''
    for i in range(args.epoch):
        avg_loss = 0
        start_time = time.time()
        model.train()
        for batch in train_loader:

            seq = batch["seq"].to(device)
            len_seq = batch["len_seq"].to(device)
            target = batch["next"].to(device)

            optimizer.zero_grad()

            x_0 = model.embed_ID(target)
      
            c = model.cacu_condition(seq, args.dropout_prob)
            n = torch.randint(0, args.timesteps, (len(seq), ), device=device).long()
            loss, _ = diff.p_losses(model, target, x_0, c, n, loss_type='l2')
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
        avg_train_loss = avg_loss / epoch_num

        if i % 1 == 0:
            print("Epoch {:03d}; ".format(i) + 'Train loss: {:.4f}; '.format(avg_train_loss) + "Time cost: " + time.strftime(
                    "%H: %M: %S", time.gmtime(time.time()-start_time)))

        if (i + 1) % 500 == 0:
                
            model.eval()
            print('-------------------------- Train PHRASE --------------------------')
            #train_hr20 = evaluate(model, 'train_data.df', diff, device)
            _ = evaluate_diff(model, diff, train_loader, device)
            
        if (i + 1) % 5 == 0:
                
            model.eval()
            print('-------------------------- VAL PHRASE --------------------------')
            val_ndcg20 = evaluate_diff(model, diff, val_loader, device)
             
            if val_ndcg20 > best_ndcg20:
                best_ndcg20 = val_ndcg20
                counter = 0  # 重置计数器
                # 保存模型
                print("\n best NDCG20 is updated to ",best_ndcg20,"at epoch",i)
    
                epoch_str = f"/Dream_{args.model_type}_rs{args.random_seed}_dim{args.hidden_dim}_null{args.null_dim}_{args.lr}.pth"
                torch.save(model.state_dict(), model_directory + epoch_str)
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping at epoch", i)
                    break  # 停止训练循环
            print('----------------------------------------------------------------')

    model.load_state_dict(torch.load(model_directory +epoch_str))
    model.eval()
    print('-------------------------- TEST PHRASE -------------------------')
    _ = evaluate_diff(model, diff, test_loader, device)
''''''
if __name__ == '__main__':
    main()



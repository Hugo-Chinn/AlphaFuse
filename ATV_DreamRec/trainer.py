import numpy as np
import pandas as pd
import math
import random
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import logging
import time as time
from datetime import datetime

#from Sdiffusion import *
from diffusion import *
from models_ori import Tenc
from utility import evaluate

#import line_profiler
def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='goodreads',
                        help='yc, ks, zhihu')
    parser.add_argument('--random_seed', type=int, default=100,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_size', type=int, default=1536,
                        help='the size of hidden vector, i.e., embedding dim.')
    parser.add_argument('--cuda', type=int, default=7,
                        help='cuda device.')
    parser.add_argument('--emb_std', type=float, default=300,
                        help='std of item embeddings')
    parser.add_argument('--emb_type', type=str, default="3small",
                        help='type of item embeddings')
    parser.add_argument('--trans_type', type=str, default="ZCA",
                        help='type of item embeddings')
    # DiTRec params
    parser.add_argument('--num_heads', type=int, default=1,
                        help='the number of heads')
    parser.add_argument('--depth', type=int, default=1,
                        help='Depth of the DiTblocks')
    parser.add_argument('--mlp_ratio', type=int, default=1,
                        help='hidden_size of MLP: mlp_ratio * hidden_size')
    # diffusion params
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
    parser.add_argument('--w', type=float, default=2.0,
                        help='classifier-free guidance strength ')
    # training params
    parser.add_argument('--lamda', type=float, default=0.01,
                        help='Weight of softmax loss.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--l2_decay', type=float, default=0,
                        help='l2 loss reg coef.')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--dropout_prob', type=float, default=0.1,
                        help='dropout_prob ')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='type of optimizer.')
    # NSI
    parser.add_argument('--emb_dim', type=int, default=None,)
    parser.add_argument('--null_thres', type=float, default=None,)
    parser.add_argument('--null_dim', type=int, default=None,)
    parser.add_argument('--standardization', type=str2bool, default=False)
    parser.add_argument('--cover', type=str2bool, default=False)
    #parser.add_argument('--standardization', action='store_false', help="default:False）")
    #parser.add_argument('--cover', action='store_false', help="default:False）")
    #parser.add_argument('--standardization', type=lambda x: x.lower() == 'True', default=False, help="启用或禁用标准化（默认：True）")
    #parser.add_argument('--cover', type=lambda x: x.lower() == 'True', default=False, help="启用或禁用覆盖（默认：False）")
    parser.add_argument('--ID_space', type=str, default="singular")
    parser.add_argument('--inject_space', type=str, default="singular")
    parser.add_argument('--emb_init_type', type=str, default="normal")
    return parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class SeqDataset(Dataset):
    def __init__(self, data):
        self.seq_data = [torch.tensor(seq, dtype=torch.long) for seq in data['seq']]
        self.len_seq_data = [torch.tensor(len_seq, dtype=torch.long) for len_seq in data['len_seq']]
        self.next_data = [torch.tensor(next_val, dtype=torch.long) for next_val in data['next']]

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        return {'seq': self.seq_data[idx], 'len_seq': self.len_seq_data[idx], 'next': self.next_data[idx]}


#@profile
def main():
    args = parse_args()
    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    data_directory = '../data/' + args.data
    model_directory = '../model/' + args.data +"/"
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing seq_size and item_num
    seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
    item_num = data_statis['item_num'][0]  # total number of items

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    # +str(args.cuda)
    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    val_data = pd.read_pickle(os.path.join(data_directory, 'val_data.df'))
    test_data = pd.read_pickle(os.path.join(data_directory, 'test_data.df'))

    train_dataset = SeqDataset(train_data)
    val_dataset = SeqDataset(val_data)
    test_dataset = SeqDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size))
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size))
    test_loader = DataLoader(test_dataset, batch_size=int(args.batch_size))

    #model  = SASRec(data_directory, item_num,seq_size,device,args)
    model = Tenc(
        args = args,
        data_directory=data_directory, 
        hidden_size=args.hidden_size,
        item_num=item_num, 
        state_size=seq_size, 
        dropout=args.dropout_rate,  
        device=device, 
        emb_type=args.emb_type,
        emb_std=args.emb_std,
        trans_type=args.trans_type,
        null_thres=args.null_thres)
    #model = Caser(args.hidden_factor,item_num, seq_size, args.diffuser_type, device, args.emb_mean_flag, args.emb_mean, args.emb_std, args.num_filters, args.filter_sizes, args.dropout_rate)
    #diff = S_Diff(
    #    args.hidden_size,
    #    args.timesteps, 
    #    args.beta_sche, 
    #    args.w, 
    #    args.linespace,
    #    args.emb_std,
    #    args.beta_start,
    #    args.beta_end,
    #    device)
    diff = diffusion(
        args.timesteps, 
        args.beta_start, 
        args.beta_end, 
        args.beta_sche, 
        args.w, 
        args.linespace,
        args.lamda)


    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer =='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer =='adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    elif args.optimizer =='rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    
    model.to(device)
    # optimizer.to(device)

    total_step=0
    hr_max = 0
    best_epoch = 0

    
    best_ndcg10 = 0  # 初始化最佳验证集指标为0
    patience = 20  # 设置容忍次数
    counter = 0  # 计数器
    epoch_num = len(train_loader)
    print("Loading data & model is done.")
    t0 = time.time()
    model.eval()
    #best_ndcg10 = evaluate(model, diff, val_loader, device, 0, args.timesteps)
    best_ndcg10 = evaluate(model, diff, val_loader, device)
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
            #x_start = model.cacu_x(target)
   
            #x_start = model.x_embedder(target)
            x_0 = model.cacu_x(target)
            #print(x_0.mean())
            #print(x_0.std())
            #y_out = model.cacu_cond(seq, args.dropout_prob)
            h = model.cacu_h(seq, len_seq, args.dropout_prob)
            n = torch.randint(0, args.timesteps, (len(seq), ), device=device).long()
            #loss = diff.training_losses(model, x_0, h, n)
            #loss, _ = diff.p_losses(model, x_0, h, n, loss_type='l2')
            loss, _ = diff.p_losses(model, target, x_0, h, n, loss_type='l2')
            #loss = loss.mean()
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
        avg_train_loss = avg_loss / epoch_num


        # scheduler.step()
        if True:
            if i % 1 == 0:
                print("Epoch {:03d}; ".format(i) + 'Train loss: {:.4f}; '.format(avg_train_loss) + "Time cost: " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-start_time)))

            if (i + 1) % 50 == 0:
                
                model.eval()
                print('-------------------------- Train PHRASE --------------------------')
                #train_hr20 = evaluate(model, 'train_data.df', diff, device)
                _ = evaluate(model, diff, train_loader, device)
            
            if (i + 1) % 5 == 0:
                
                model.eval()
                eval_start = time.time()
                print('-------------------------- VAL PHRASE --------------------------')
                #train_hr20 = evaluate(model, 'val_data.df', diff, device)
                val_ndcg10 = evaluate(model, diff, val_loader, device)
                #val_ndcg10 = evaluate(model, diff, val_loader, device, 0, args.timesteps)
                #print('-------------------------- Test PHRASE --------------------------')
                #train_hr20 = evaluate(model, 'train_data.df', diff, device)
                #_ = evaluate(model, diff, test_loader, device)
                #for i in range(0, args.timesteps, args.linespace):
                #    print("the eval timesteps interval is [", args.timesteps - args.linespace - i, ",", args.timesteps - i, "]:")
                #    _ = evaluate(model, diff, test_loader, device, i, args.linespace)
             
                if val_ndcg10 > best_ndcg10:
                    best_ndcg10 = val_ndcg10
                    counter = 0  # 重置计数器
                    # 保存模型
                    print("\n best hr20 is updated to ",best_ndcg10,"at epoch",i)
                    current_time = datetime.now()
                    #epoch_str = f"/Re_w{args.w}_beta_sche{args.beta_sche}_std{args.emb_std}_epoch{epoch_num}_bestmodel_time{current_time}.pth"
                    epoch_str = f"/AlphaFuse_woStand_rs{args.random_seed}_{args.data}_dim{args.hidden_size}Null{args.null_thres}_{args.lr}.pth"
                    torch.save(model.state_dict(), model_directory + epoch_str)
                else:
                    counter += 1
                    if counter >= patience:
                        print("Early stopping at epoch", i)
                        break  # 停止训练循环
                print("Evalution cost: " + time.strftime("%H: %M: %S", time.gmtime(time.time()-eval_start)))
                print('----------------------------------------------------------------')

    model.load_state_dict(torch.load(model_directory +epoch_str))
    model.eval()
    print("Best Result on val:train=10:0 data")
    print('-------------------------- TEST PHRASE -------------------------')
    _ = evaluate(model, diff, test_loader, device)
    print('-------------------------- Train PHRASE -------------------------')
    _ = evaluate(model, diff, train_loader, device)      
    print('-------------------------- VAL PHRASE --------------------------')
    _ = evaluate(model, diff, val_loader, device)
''''''
if __name__ == '__main__':
    main()



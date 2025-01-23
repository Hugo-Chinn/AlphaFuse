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

from models.backbone_SASRec import SASRec,MoRec, WhitenRec, UniSRec, RLMRec, LLMESR, LLMInit, WhitenRec, AlphaFuse
from utils import evaluate, evaluate_diff

#torch.autograd.set_detect_anomaly(True)

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
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='ATV',
                        help='yc, ks, zhihu')
    parser.add_argument('--cuda', type=int, default=7,
                        help='cuda device.')
    parser.add_argument('--l2_decay', type=float, default=1e-6,
                        help='l2 loss reg coef.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    ### SASRec backbone settings
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    ### loss function parameters
    parser.add_argument('--loss_type', type=str, default="infoNCE")
    parser.add_argument('--neg_ratio', type=int, default=1,
                        help='#Negative:#Positive = neg_ratio.')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='tao.')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='scale of additional loss of RLMRec and LLMESR')
    ### languange embeddings settings
    parser.add_argument('--language_model_type', default="3small", type=str)
    parser.add_argument('--language_embs_scale', default=40, type=int)

    ### ID embeddings settings
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Number of hidden factors, i.e., ID embedding size.')
    parser.add_argument('--ID_embs_init_type', type=str, default="normal")
    ### model selection
    parser.add_argument('--model_type', type=str, default="UniSRec")
    parser.add_argument('--SR_aligement_type', type=str, default="con")
    # AlphaFuse
    #parser.add_argument('--emb_dim', type=int, default=None,)
    parser.add_argument('--null_thres', type=float, default=None,)
    parser.add_argument('--null_dim', type=int, default=64,)
    parser.add_argument('--item_frequency_flag', type=str2bool, default=False)
    parser.add_argument('--standardization', type=str2bool, default=True)
    parser.add_argument('--cover', type=str2bool, default=False)
    parser.add_argument('--ID_space', type=str, default="singular")
    parser.add_argument('--inject_space', type=str, default="singular")
    #parser.add_argument('--emb_init_type', type=str, default="normal")
    #parser.add_argument('--emb_sim_type', type=str, default="both")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    setup_seed(args.random_seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    key_words = vars(args)
    data_directory = '../ours_DiT/data/' + args.data
    model_directory = './saved/' +args.data
    
    key_words["language_embs_path"] = data_directory


    if args.model_type == "SASRec":
        model = SASRec(device, **key_words).to(device)
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

    #print(device)
    #print(next(model.item_embeddings.language_embeddings.parameters()).device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_delay_rate)
    
    # 计算参数量
    total_params, trainable_params = count_parameters(model)

    # 输出结果
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

    #for name, param in model.named_parameters():
    #    try:
    #        torch.nn.init.xavier_normal_(param.data)
    #    except:
    #        pass # just ignore those failed init layers
        
    print(key_words)

    #model.train() # enable model training

    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    train_data.reset_index(inplace=True,drop=True)
    train_dataset = SeqDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    val_data = pd.read_pickle(os.path.join(data_directory, 'val_data.df'))
    val_data.reset_index(inplace=True,drop=True)
    val_dataset = SeqDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    test_data = pd.read_pickle(os.path.join(data_directory, 'test_data.df'))
    test_data.reset_index(inplace=True,drop=True)
    test_dataset = SeqDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    best_ndcg20 = 0  # 初始化最佳验证集指标为0
    patience = 50  # 设置容忍次数
    counter = 0  # 计数器

    step = 0
    T = 0.0
    print("Loading Data Done.")
    t0 = time.time()
    val_ndcg20 = evaluate(model, val_loader, device)
    t1 = time.time() - t0
    print("\n using ",t1, "s ", "Eval Time Cost",T,"s.")
    ''''''
    for epoch in range(args.epoch):
        model.train()
        for batch in train_loader:
            
            batch_size = len(batch['seq'])
            seq = batch['seq'].to(device)
            #len_seq = batch['len_seq'].to(device)
            target = batch['next'].to(device)
            
            optimizer.zero_grad()
            
            if args.loss_type == "CE":
                loss = model.calculate_ce_loss(seq, target)
            elif args.loss_type == "BCE":
                loss = model.calculate_bce_loss(seq, target, args.neg_ratio)
            elif args.loss_type == "infoNCE":
                loss = model.calculate_infonce_loss(seq,  target, args.neg_ratio, args.temperature)
                
            if args.model_type == "RLMRec":
                if args.SR_aligement_type == 'con':
                    recon_loss = model.reconstruct_con_loss()
                elif args.SR_aligement_type == 'gen':
                    recon_loss = model.reconstruct_gen_loss()
                loss = loss + args.beta * recon_loss
            
            if args.model_type == "LLMESR":
                recon_loss = model.reg_loss(seq)
                loss = loss + args.beta * recon_loss

            loss.backward()
            optimizer.step()
            step+=1
            # print("loss in epoch {} iteration {}: {}".format(i, step, loss.item())) # expected 0.4~0.6 after init few epochs
        print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
        #if (epoch+1) % args.lr_delay_epoch == 0:
        #    scheduler.step()
        #    print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()}")
        if (epoch+1) % 50 == 0:
            _ = evaluate(model, train_loader, device)
        if (epoch+1) % 1 == 0:
            model.eval()
            print('-------------------------- EVALUATE PHRASE --------------------------')
            t0 = time.time()
            val_ndcg20 = evaluate(model, val_loader, device)
            t1 = time.time() - t0
            print("\n using ",t1, "s ", "Eval Time Cost",T,"s.")

            model.train()
            tv_ndcg20 = val_ndcg20 
            if tv_ndcg20 > best_ndcg20:
                best_ndcg20 = tv_ndcg20
                counter = 0  # 重置计数器
                # 保存模型
                print("\n best NDCG@20 is updated to ",best_ndcg20,"at epoch",epoch)
                epoch_str = f"{args.model_type}_rs{args.random_seed}_IDdim{args.hidden_dim}_Textdim{args.null_dim}_{args.lr}_{args.loss_type}.pth"
                torch.save(model.state_dict(), model_directory + epoch_str)
            else:
                counter += 1
                if counter >= patience:
                    break   # 停止训练循环
            print('----------------------------------------------------------------')
            
    model.load_state_dict(torch.load(model_directory + epoch_str))
    items_emb =model.return_item_emb()
    #with open('./item_emb/ATV_SASRec_ID64.pickle', 'wb') as f:
    #    pickle.dump(items_emb, f)
    model.eval()
    print('-------------------------- TEST RESULTS --------------------------')
    _ = evaluate(model, test_loader, device)
    print("Done.")
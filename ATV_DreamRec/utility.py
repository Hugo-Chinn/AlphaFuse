import os
import copy
import math
import numpy as np
import pandas as pd
from collections import deque
import torch.nn as nn
import torch
import torch.nn.functional as F
# import tensorflow as tf

def calculate_hit_loader(sorted_list,topk,true_items,hit_purchase,ndcg_purchase,mrr_purchase):
    true_items = true_items.tolist()
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        # print(rec_list)
        # print(true_items)
        # print('...........')
        # break
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])[0,0]
                # total_reward[i] += rewards[j]
                # if rewards[j] == r_click:
                #     hit_click[i] += 1.0
                #     ndcg_click[i] += 1.0 / np.log2(rank + 1)
                # else:
                hit_purchase[i] += 1.0
                ndcg_purchase[i] += 1.0 / np.log2(rank + 1)
                mrr_purchase[i] += 1.0/ rank

def evaluate_tg(model, diff, matrix, val_loader, w, device):

    total_purchase = 0.0
    hit_purchase=[0,0,0,0,0]
    ndcg_purchase=[0,0,0,0,0]
    mrr_purchase = [0,0,0,0,0]
    topk = [1,5,10,20,50]

    for batch in val_loader:
        seq = batch["seq"].to(device)
        len_seq = batch["len_seq"]
        target = batch["next"]

        #_, prediction = model.predict(seq, diff)
        _, prediction = model.tg_predict(seq, target, w, diff) # args.linespace
        #print(prediction.shape)
        #prediction = torch.matmul(embedding, matrix.transpose(0, 1))
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2=np.flip(topK,axis=1)
        calculate_hit_loader(sorted_list2,topk,target,hit_purchase,ndcg_purchase,mrr_purchase)
        total_purchase+=len(seq)

    hr_list = []
    ndcg_list = []
    mrr_list = []
    for i in range(len(topk)):
        hr_purchase=hit_purchase[i]/total_purchase
        ng_purchase=ndcg_purchase[i]/total_purchase
        mr_purchase=mrr_purchase[i]/total_purchase
        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase)
        mrr_list.append(mr_purchase)
    print('{:<10s} {:<10s} {:<10s} '.format("ACC","ACC","ACC"))
    print('{:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), mrr_list[0]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@'+str(topk[1]), 'HR@'+str(topk[2]), 'HR@'+str(topk[3]), 'HR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[1], (hr_list[2]), hr_list[3], hr_list[4]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('NDCG@'+str(topk[1]), 'NDCG@'+str(topk[2]), 'NDCG@'+str(topk[3]), 'NDCG@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(ndcg_list[1], (ndcg_list[2]), ndcg_list[3], ndcg_list[4]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('MRR@'+str(topk[1]), 'MRR@'+str(topk[2]), 'MRR@'+str(topk[3]), 'MRR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(mrr_list[1], (mrr_list[2]), mrr_list[3], mrr_list[4]))

    return hr_list, ndcg_list

def evaluate_lt_tg(model, diff, lt, matrix, val_loader, w, device):

    total_purchase = 0.0
    hit_purchase=[0,0,0,0,0]
    ndcg_purchase=[0,0,0,0,0]
    mrr_purchase = [0,0,0,0,0]
    topk = [1,5,10,20,50]

    for batch in val_loader:
        seq = batch["seq"].to(device)
        len_seq = batch["len_seq"]
        target = batch["next"]

        #_, prediction = model.predict(seq, diff)
        oracle, _ = model.tg_predict(seq, target, w, diff) # args.linespace
        embedding = lt(oracle)
        #print(prediction.shape)
        prediction = torch.matmul(embedding, matrix.transpose(0, 1))
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2=np.flip(topK,axis=1)
        calculate_hit_loader(sorted_list2,topk,target,hit_purchase,ndcg_purchase,mrr_purchase)
        total_purchase+=len(seq)

    hr_list = []
    ndcg_list = []
    mrr_list = []
    for i in range(len(topk)):
        hr_purchase=hit_purchase[i]/total_purchase
        ng_purchase=ndcg_purchase[i]/total_purchase
        mr_purchase=mrr_purchase[i]/total_purchase
        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase)
        mrr_list.append(mr_purchase)
    print('{:<10s} {:<10s} {:<10s} '.format("ACC","ACC","ACC"))
    print('{:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), mrr_list[0]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@'+str(topk[1]), 'HR@'+str(topk[2]), 'HR@'+str(topk[3]), 'HR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[1], (hr_list[2]), hr_list[3], hr_list[4]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('NDCG@'+str(topk[1]), 'NDCG@'+str(topk[2]), 'NDCG@'+str(topk[3]), 'NDCG@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(ndcg_list[1], (ndcg_list[2]), ndcg_list[3], ndcg_list[4]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('MRR@'+str(topk[1]), 'MRR@'+str(topk[2]), 'MRR@'+str(topk[3]), 'MRR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(mrr_list[1], (mrr_list[2]), mrr_list[3], mrr_list[4]))

    return hr_list, ndcg_list

def evaluate_lt(model, diff, lt, matrix, val_loader, device):

    total_purchase = 0.0
    hit_purchase=[0,0,0,0,0]
    ndcg_purchase=[0,0,0,0,0]
    mrr_purchase = [0,0,0,0,0]
    topk = [1,5,10,20,50]

    for batch in val_loader:
        seq = batch["seq"].to(device)
        len_seq = batch["len_seq"]
        target = batch["next"]

        #_, prediction = model.predict(seq, diff)
        oracle, _ = model.predict(seq, diff) # args.linespace
        embedding = lt(oracle)
        #print(prediction.shape)
        prediction = torch.matmul(embedding, matrix.transpose(0, 1))
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2=np.flip(topK,axis=1)
        calculate_hit_loader(sorted_list2,topk,target,hit_purchase,ndcg_purchase,mrr_purchase)
        total_purchase+=len(seq)

    hr_list = []
    ndcg_list = []
    mrr_list = []
    for i in range(len(topk)):
        hr_purchase=hit_purchase[i]/total_purchase
        ng_purchase=ndcg_purchase[i]/total_purchase
        mr_purchase=mrr_purchase[i]/total_purchase
        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase)
        mrr_list.append(mr_purchase)
    print('{:<10s} {:<10s} {:<10s} '.format("ACC","ACC","ACC"))
    print('{:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), mrr_list[0]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@'+str(topk[1]), 'HR@'+str(topk[2]), 'HR@'+str(topk[3]), 'HR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[1], (hr_list[2]), hr_list[3], hr_list[4]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('NDCG@'+str(topk[1]), 'NDCG@'+str(topk[2]), 'NDCG@'+str(topk[3]), 'NDCG@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(ndcg_list[1], (ndcg_list[2]), ndcg_list[3], ndcg_list[4]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('MRR@'+str(topk[1]), 'MRR@'+str(topk[2]), 'MRR@'+str(topk[3]), 'MRR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(mrr_list[1], (mrr_list[2]), mrr_list[3], mrr_list[4]))

    return hr_list, ndcg_list
    
def evaluate_loader(model, diff, val_loader, device):

    total_purchase = 0.0
    hit_purchase=[0,0,0,0,0]
    ndcg_purchase=[0,0,0,0,0]
    mrr_purchase = [0,0,0,0,0]
    topk = [1,5,10,20,50]

    for batch in val_loader:
        seq = batch["seq"].to(device)
        len_seq = batch["len_seq"]
        target = batch["next"].to(device)

        #_, prediction = model.predict(seq, diff)
        _, prediction = model.predict(seq, target,  diff) # args.linespace
        #print(prediction.shape)
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2=np.flip(topK,axis=1)
        calculate_hit_loader(sorted_list2,topk,target,hit_purchase,ndcg_purchase,mrr_purchase)
        total_purchase+=len(seq)

    hr_list = []
    ndcg_list = []
    mrr_list = []
    for i in range(len(topk)):
        hr_purchase=hit_purchase[i]/total_purchase
        ng_purchase=ndcg_purchase[i]/total_purchase
        mr_purchase=mrr_purchase[i]/total_purchase
        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase)
        mrr_list.append(mr_purchase)
    print('{:<10s} {:<10s} {:<10s} '.format("ACC","ACC","ACC"))
    print('{:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), mrr_list[0]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@'+str(topk[1]), 'HR@'+str(topk[2]), 'HR@'+str(topk[3]), 'HR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[1], (hr_list[2]), hr_list[3], hr_list[4]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('NDCG@'+str(topk[1]), 'NDCG@'+str(topk[2]), 'NDCG@'+str(topk[3]), 'NDCG@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(ndcg_list[1], (ndcg_list[2]), ndcg_list[3], ndcg_list[4]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('MRR@'+str(topk[1]), 'MRR@'+str(topk[2]), 'MRR@'+str(topk[3]), 'MRR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(mrr_list[1], (mrr_list[2]), mrr_list[3], mrr_list[4]))

    return hr_list, ndcg_list


def evaluate(model, diff, val_loader, device):
    # timesteps_end, int_length
    total_purchase = 0.0
    hit_purchase=[0,0,0,0,0]
    ndcg_purchase=[0,0,0,0,0]
    mrr_purchase = [0,0,0,0,0]
    topk = [1,5,10,20,50]

    for batch in val_loader:
        seq = batch["seq"].to(device)
        len_seq = batch["len_seq"]
        target = batch["next"].to(device)

        #_, prediction = model.predict(seq, target, diff, timesteps_end, int_length)
        _, prediction = model.predict(seq, target, diff)
        #print(prediction.shape)
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2=np.flip(topK,axis=1)
        calculate_hit_loader(sorted_list2,topk,target,hit_purchase,ndcg_purchase,mrr_purchase)
        total_purchase+=len(seq)

    hr_list = []
    ndcg_list = []
    mrr_list = []
    for i in range(len(topk)):
        hr_purchase=hit_purchase[i]/total_purchase
        ng_purchase=ndcg_purchase[i]/total_purchase
        mr_purchase=mrr_purchase[i]/total_purchase
        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase)
        mrr_list.append(mr_purchase)
    print('{:<10s} {:<10s} {:<10s} '.format("ACC","ACC","ACC"))
    print('{:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), mrr_list[0]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@'+str(topk[1]), 'HR@'+str(topk[2]), 'HR@'+str(topk[3]), 'HR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[1], (hr_list[2]), hr_list[3], hr_list[4]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('NDCG@'+str(topk[1]), 'NDCG@'+str(topk[2]), 'NDCG@'+str(topk[3]), 'NDCG@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(ndcg_list[1], (ndcg_list[2]), ndcg_list[3], ndcg_list[4]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('MRR@'+str(topk[1]), 'MRR@'+str(topk[2]), 'MRR@'+str(topk[3]), 'MRR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(mrr_list[1], (mrr_list[2]), mrr_list[3], mrr_list[4]))

    return ndcg_list[2]
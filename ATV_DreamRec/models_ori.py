import numpy as np
import pandas as pd
import torch
import math
from torch import nn
import torch.nn.functional as F
import os
from collections import Counter
from Modules import *

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            #nn.Linear(frequency_embedding_size, 2*hidden_size, bias=True),
            nn.Linear(frequency_embedding_size, hidden_size),
            #nn.SiLU(),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            #nn.Linear(2*hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        #embeddings = math.log(10000) / (half_dim - 1)
        #embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        #embeddings = time[:, None] * embeddings[None, :]
        #embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        half = dim // 2
        freqs = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=t.device) * -freqs)
        #freqs = torch.exp(
        #    -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        #).to(device=t.device)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        #embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_dim, ):
        super().__init__()
        self.linear = nn.Linear(3*hidden_size, out_dim)

    def forward(self, x, t, y):
        """
        x:(N, L+1, d) tensor of noisy item embedding
        t:(N, d)
        """
        y_out = torch.cat((x, y, t), dim=1)
        x_out = self.linear(y_out)
        return x_out

def linear_transformation(all_samples, trans_type, emb_std, hidden_size):
        mean = np.mean(all_samples, axis=0)
        cov = np.cov( all_samples - mean, rowvar=False)
        #cov = np.cov( word_vectors -self.mean, rowvar=False)
        U, S, VT = np.linalg.svd(cov, full_matrices=False)
        if trans_type == "PCA":
            W = U.dot(np.diag(np.sqrt(1/S)))
        elif trans_type == "avgPCA":
            W = np.sqrt(1/S).mean() * U
        elif trans_type == "ZCA":
            W = U.dot(np.dot(np.diag(np.sqrt(1/S)), U.T)) # ZCA
        elif trans_type == "U":
            W = U
        elif trans_type == "UUT":
            W = U.dot(U.T)

        word_vectors = np.dot((all_samples), W)[:,:hidden_size]
        # np.dot(np.diag(np.sqrt(1/S)), U.T
        
        #norms = np.linalg.norm(word_vectors, axis=1, keepdims=True)
        #word_vectors = word_vectors / norms * emb_std
        #word_vectors = word_vectors / norms 
        #word_vectors[np.isnan(word_vectors)] = 0
        #word_vectors = np.dot((all_samples), W)[:,:hidden_size]
        #word_vectors = np.dot((word_vectors-self.mean), self.W[:,:self.hidden_size])
        #word_vectors = np.dot((word_vectors-int_w*self.mean), int_w*self.W +(1-int_w)* np.eye(self.hidden_size))
        #word_vectors = np.dot((word_vectors), self.W)
        return mean, W, word_vectors
    
def rank_null_transformation(all_samples, threshold):
        #items_pop = np.load(os.path.join("./data/ATV", 'items_pop.npy'))
        #items_freq = (items_pop / items_pop.sum()).reshape(-1, 1)
        
        #mean = np.sum(all_samples*items_freq, axis=0)
        #cov = np.cov( (all_samples - mean)*np.sqrt(items_freq), rowvar=False)
        mean = np.mean(all_samples, axis=0)
        cov = np.cov( all_samples - mean, rowvar=False)
        #cov = np.cov( word_vectors -self.mean, rowvar=False)
        U, S, VT = np.linalg.svd(cov, full_matrices=False)

        indices_null = np.where(S <= threshold)[0]
        indices_rank = np.where(S > threshold)[0]
        rank = len(indices_rank) 

        S_null = S[indices_null]
        S_rank = S[indices_rank]
        
        W_rank = U[:, indices_rank].dot(np.diag(np.sqrt(1/S_rank)))
        
        #W_rank = U[:, indices_rank]
 
        W_null = U[:, indices_null]
        

        #word_vectors = np.dot((all_samples-mean), W_rank)
        #np.dot(np.diag(np.sqrt(1/S)), U.T)
        word_vectors = np.dot((all_samples-mean), W_rank)
        #norms = np.linalg.norm(word_vectors, axis=1, keepdims=True)
        #word_vectors = word_vectors / norms * emb_std
        #word_vectors = word_vectors / norms 
        #word_vectors[np.isnan(word_vectors)] = 0
        #word_vectors = np.dot((all_samples), W)[:,:hidden_size]
        #word_vectors = np.dot((word_vectors-self.mean), self.W[:,:self.hidden_size])
        #word_vectors = np.dot((word_vectors-int_w*self.mean), int_w*self.W +(1-int_w)* np.eye(self.hidden_size))
        #word_vectors = np.dot((word_vectors), self.W)
        return rank, mean, W_rank, W_null, word_vectors
            

class Tenc(nn.Module):
    def __init__(self, args, data_directory, hidden_size, item_num, state_size, dropout, device, emb_type, emb_std, trans_type, null_thres, num_heads=1):
        super(Tenc, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.device = device
        print(args)
        #self.item_emb = NSI(data_directory, args.emb_std, args.emb_type, args.emb_dim, args.emb_init_type, args.null_thres, args.null_dim,\
        #    args.standardization, args.cover, args.ID_space, args.inject_space)
        """
        if args.emb_dim is not None:
            self.hidden_size = args.emb_dim 
        else:
            self.hidden_size = args.hidden_factor
        """
        if "3small" in emb_type or "3large" in emb_type:
            wordvec_df = pd.read_pickle(os.path.join(data_directory, emb_type+'_emb.pickle'))
            word_vectors = np.stack(wordvec_df)
            if trans_type != "None" and "Null" not in trans_type:
                all_samples = word_vectors * emb_std
                self.mean, self.W, word_vectors = linear_transformation(all_samples, trans_type, emb_std, self.hidden_size)
            elif "Null" in trans_type:
                all_samples = word_vectors * emb_std
                self.rank, self.mean, self.W_r, self.W_null, word_vectors = rank_null_transformation(all_samples, threshold=null_thres)
                #self.rank = 0
                #word_vectors = all_samples
                self.ID_embeddings = nn.Embedding(
                    num_embeddings=item_num + 1,
                    embedding_dim=hidden_size-self.rank,
                    padding_idx=self.item_num
                )
                nn.init.normal_(self.ID_embeddings.weight, 0, 1)
                print("Rank ", self.rank)
                #print(word_vectors.shape)
                random_vector = np.random.randn(word_vectors.shape[1])  
                word_vectors = np.vstack([word_vectors, random_vector])
                self.item_embeddings = nn.Embedding.from_pretrained(torch.tensor(word_vectors,dtype=torch.float32),freeze=True,padding_idx=self.item_num)
            else:
                #print(word_vectors.mean())
                #print(word_vectors.std())
                #print(emb_std)
                word_vectors = word_vectors * emb_std
                #print(word_vectors.mean())
                #print(word_vectors.std())
        """
        elif "ID" in emb_type:
            print(emb_type)
            wordvec_df = pd.read_pickle(os.path.join(data_directory, emb_type+'_emb.pickle')).transpose(1,0)
            word_vectors = wordvec_df.detach().cpu().numpy()[:-1]
            word_vectors = word_vectors * emb_std
            if trans_type != "None":
                all_samples = word_vectors
                _, _, word_vectors = linear_transformation(all_samples, trans_type, emb_std, self.hidden_size)
        """
        
        """
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        """
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.state_size,
            embedding_dim=self.hidden_size
        )

        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(self.hidden_size)
        self.ln_2 = nn.LayerNorm(self.hidden_size)
        self.ln_3 = nn.LayerNorm(self.hidden_size)
        self.mh_attn = MultiHeadAttention(self.hidden_size, self.hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(self.hidden_size, self.hidden_size, dropout)
        
        self.step_mlp = TimestepEmbedder(self.hidden_size)

        self.diffuser = FinalLayer(self.hidden_size, self.hidden_size)
    
    def return_item_emb(self,):
        emb_rank = self.item_embeddings.weight
        emb_null = self.ID_embeddings.weight
        #return emb_rank + emb_null
        return torch.cat((emb_rank, emb_null), dim=-1)
    
    def forward(self, x, h, step):

        t = self.step_mlp(step)
        res = self.diffuser(x,t,h)
        return res

    def forward_uncon(self, x, step):
        h = self.none_embedding(torch.tensor([0]).to(self.device))
        h = torch.cat([h.view(1, self.hidden_size)]*x.shape[0], dim=0)

        t = self.step_mlp(step)
        res = self.diffuser(x,t,h)
            
        return res

    
    def cacu_x(self, x):
        x_rank = self.item_embeddings(x)
        x_null = self.ID_embeddings(x)
        #return x_rank + x_null
        return torch.cat((x_rank, x_null), dim=-1)

    
    def cacu_h(self, states, len_states, p):
        #hidden
        B, L = states.shape
        #inputs_emb_rank = self.item_embeddings(states)
        #inputs_emb_null = self.ID_embeddings(states)
        #inputs_emb = torch.cat((inputs_emb_rank, inputs_emb_null), dim=-1)
        inputs_emb = self.cacu_x(states)
        #inputs_emb = self.item_emb.inject(states)

        #inputs_emb = self.LT(inputs_emb.view(-1, self.hidden_size).reshape(B,L,-1))
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        #state_hidden = extract_axis_1(ff_out, len_states - 1)
        state_hidden = ff_out[:,-1,:]
        h = state_hidden.squeeze()

        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(self.device)

        # print(h.device, self.none_embedding(torch.tensor([0]).to(self.device)).device, mask.device)
        h = h * mask + self.none_embedding(torch.tensor([0]).to(self.device)) * (1-mask)

        return h  
    
    def predict(self, states, target, diff):
        # timesteps_end, int_length
        #hidden
        #inputs_emb_rank = self.item_embeddings(states)
        #inputs_emb_null = self.ID_embeddings(states)
        #inputs_emb = torch.cat((inputs_emb_rank, inputs_emb_null), dim=-1)

        inputs_emb = self.cacu_x(states)

        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        #state_hidden = extract_axis_1(ff_out, len_states - 1)
        state_hidden = ff_out[:,-1]
        h = state_hidden.squeeze()
        
        #item_emb = self.item_embeddings(items)
        #x = diff.sample_from_item( item_emb, self.forward, self.forward_uncon, h)
        #x = diff.sample_from_noise( self.forward, self.forward_uncon, self.cacu_x(target), h,self.hidden_size, timesteps_end, int_length)
        x = diff.sample_from_noise( self.forward, self.forward_uncon, h,self.hidden_size)
        test_item_emb = self.return_item_emb()
        #test_item_emb = self.item_emb.return_embs()
        scores = torch.matmul(x, test_item_emb[:-1].transpose(0, 1))

        return x, scores

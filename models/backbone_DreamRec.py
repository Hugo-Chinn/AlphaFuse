import os
import copy
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embedding import Item_Embedding
from models.modules import *

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_dim, input_channels = 3 ):
        super().__init__()
        self.linear = nn.Linear(input_channels*hidden_size, out_dim)

    def forward(self, x, t, y):
        """
        x:(N, L+1, d) tensor of noisy item embedding
        t:(N, d)
        """

        y_out = torch.cat((x, y, t), dim=1)
        x_out = self.linear(y_out)
        return x_out
    
class NSI(nn.Module):
    def __init__(self, data_directory, emb_std, emb_type, emb_dim, emb_init_type, null_thres, null_dim, standardization, cover, ID_space, inject_space):
        super(NSI, self).__init__()
        base_embs = self.load_base_embs(data_directory, emb_type, emb_std)
        self.emb_dim = emb_dim
        self.construct_null_space(base_embs, null_thres=null_thres, null_dim=null_dim)
        #print(standardization)
        #print(cover)
        #print(ID_space)
        #print(inject_space)
        self.inject, self.return_embs = self.init_injection(
            base_embs, 
            standardization=standardization,
            cover=cover,
            ID_space=ID_space,
            inject_space=inject_space
            )
        self.ID_embs_init(emb_init_type)
    
    def ID_embs_init(self, emb_init_type):
        if emb_init_type == "uniform":
            nn.init.uniform_(self.ID_embeddings.weight, a=0.0, b=1.0)
        elif emb_init_type == "normal":
            nn.init.normal_(self.ID_embeddings.weight, 0, 1)
        elif emb_init_type == "zero":
            nn.init.zeros_(self.ID_embeddings.weight)
        elif emb_init_type == "ortho":
            nn.init.orthogonal_(self.ID_embeddings.weight, gain=1.0)
        elif emb_init_type == "xavier":
            nn.init.xavier_uniform_(self.ID_embeddings.weight, gain=1.0)
        elif emb_init_type == "sparse":
            nn.init.sparse_(self.ID_embeddings.weight, 0.01, std=1)

        
    def load_base_embs(self,data_directory, emb_type, emb_std):
        text_embs = pd.read_pickle(os.path.join(data_directory, emb_type+'_emb.pickle'))
        self.item_num = len(text_embs)
        return np.stack(text_embs) * emb_std
    
    def construct_null_space(self, base_embs, null_thres=None, null_dim=None):
        self.mean = np.mean(base_embs, axis=0)
        cov = np.cov( base_embs - self.mean, rowvar=False)
        U, S, _ = np.linalg.svd(cov, full_matrices=False)

        if null_thres is not None:
            indices_null = np.where(S <= null_thres)[0]
            indices_rank = np.where(S > null_thres)[0]
        elif null_dim is not None:
            indices = np.arange(len(S))
            indices_null = indices[-null_dim:]
            indices_rank = indices[:-null_dim]
            
        self.nullity = len(indices_null)
        print("The Nullity is", self.nullity)
        #self.S_null = S[indices_null]
        #self.S_rank = S[indices_rank]
        self.S = S
            # U[:, indices_rank].dot(np.diag(np.sqrt(1/S_rank)))
        self.U = U
        self.U_null = torch.tensor(U[:, indices_null]).float()
        #self.U_rank = U[:, indices_rank]
        return None
    
    def init_injection(self, base_embs, standardization=False, cover=False, ID_space="singular", inject_space="singular"):

        if ID_space == "singular" and inject_space == "singular":
            P = self.U
            S = self.S
            if not cover:
                def injection(id, emb_type="both"):
                    x = self.text_embeddings(id)
                    #x_null = self.ID_embeddings(id)
                    y = x.clone()
                    if emb_type == "both":
                        x_null = self.ID_embeddings(id)
                        y[..., -self.nullity:] = x[..., -self.nullity:] + x_null
                    elif emb_type == "id":
                        x_null = self.ID_embeddings(id)
                        y[..., :-self.nullity] = 0
                        y[..., -self.nullity:] = x_null 
                    return y
                def return_embs(emb_type="both"):
                    x = self.text_embeddings.weight
                    #x_null = self.ID_embeddings.weight
                    y = x.clone()
                    if emb_type == "both":
                        x_null = self.ID_embeddings.weight
                        y[..., -self.nullity:] = x[..., -self.nullity:] + x_null
                    elif emb_type == "id":
                        x_null = self.ID_embeddings.weight
                        y[..., :-self.nullity] = 0
                        y[..., -self.nullity:] = x_null 
                    #x[..., -self.nullity:] = x[..., -self.nullity:] + x_null
                    return y
            else:
                def injection(id, emb_type="both"):
                    x = self.text_embeddings(id)
                    x_null = self.ID_embeddings(id)
                    y = x.clone()
                    y[..., -self.nullity:] = 0
                    if emb_type == "both":
                        x_null = self.ID_embeddings(id)
                        y[..., -self.nullity:] = x_null
                    elif emb_type == "id":
                        x_null = self.ID_embeddings(id)
                        y[..., :-self.nullity] = 0
                        y[..., -self.nullity:] = x_null 
                    return y
                def return_embs(emb_type="both"):
                    x = self.text_embeddings.weight
                    #x_null = self.ID_embeddings.weight
                    #x[..., -self.nullity:] = 0
                    #x[..., -self.nullity:] = x[..., -self.nullity:] + x_null
                    y = x.clone()
                    y[..., -self.nullity:] = 0
                    if emb_type == "both":
                        x_null = self.ID_embeddings.weight
                        y[..., -self.nullity:] = x_null
                    elif emb_type == "id":
                        x_null = self.ID_embeddings.weight
                        y[..., :-self.nullity] = 0
                        y[..., -self.nullity:] = x_null 
                    return y
            if standardization:
                P = P.dot(np.diag(np.sqrt(1/S)))
            else:
                P = P
            #base_embs = base_embs.dot(P)
            base_embs = (base_embs-self.mean).dot(P[:,:self.emb_dim])
            self.ID_embeddings = nn.Embedding(
                num_embeddings=self.item_num+1,
                embedding_dim=self.nullity,
            )
        elif ID_space == "euclidean"and inject_space == "singular":
            P = self.U
            S = self.S
            if not cover:
                def injection(id):
                    x = self.text_embeddings(id)
                    x_null = self.ID_embeddings(id)
                    #x_null = x_null @ self.U_null.to(x.device)
                    x_null = (x_null-torch.tensor(self.mean).to(x.device).float()) @ self.U_null.to(x.device)
                    x = x.clone()
                    x[..., -self.nullity:] = x[..., -self.nullity:] + x_null
                    return x
                def return_embs():
                    x = self.text_embeddings.weight
                    x_null = self.ID_embeddings.weight
                    #x_null = x_null @ self.U_null.to(x.device)
                    x_null = (x_null-torch.tensor(self.mean).to(x.device).float()) @ self.U_null.to(x.device)
                    x = x.clone()
                    x[..., -self.nullity:] = x[..., -self.nullity:] + x_null
                    return x
                    
            else:
                def injection(id):
                    x = self.text_embeddings(id)
                    x_null = self.ID_embeddings(id)
                    #x_null = x_null @ self.U_null.to(x.device)
                    x_null = (x_null-torch.tensor(self.mean).to(x.device).float()) @ self.U_null.to(x.device)
                    x = x.clone()
                    x[..., -self.nullity:] = 0
                    x[..., -self.nullity:] = x[..., -self.nullity:] + x_null
                    return x
                def return_embs():
                    x = self.text_embeddings.weight
                    x_null = self.ID_embeddings.weight
                    #x_null = x_null @ self.U_null.to(x.device)
                    x_null = (x_null-torch.tensor(self.mean).to(x.device).float()) @ self.U_null.to(x.device)
                    x = x.clone()
                    x[..., -self.nullity:] = 0
                    x[..., -self.nullity:] = x[..., -self.nullity:] + x_null
                    return x
            if standardization:
                P = P.dot(np.diag(np.sqrt(1/S)))
            else:
                P = P
            #base_embs = base_embs.dot(P)
            base_embs = (base_embs-self.mean).dot(P[:,:self.emb_dim])
            self.ID_embeddings = nn.Embedding(
                num_embeddings=self.item_num+1,
                embedding_dim=base_embs.shape[-1],
            )
        elif ID_space == "singular"and inject_space == "euclidean":
            P = self.U
            self.ID_embeddings = nn.Embedding(
                num_embeddings=self.item_num+1,
                embedding_dim=self.nullity,
            )
            if standardization:
                singulars = np.ones(base_embs.shape[-1])
                singulars[:-self.nullity] = np.sqrt(1/self.S[:-self.nullity])
                P = P.dot(np.dot(np.diag(singulars),P.T))
                #base_embs = base_embs.dot(P)
                base_embs = (base_embs-self.mean).dot(P) + self.mean
            def injection(id):
                x = self.text_embeddings(id)
                x_null = self.ID_embeddings(id)
                #x_null = x_null @ self.U_null.T.to(x.device)
                x_null = x_null @ self.U_null.T.to(x.device) + torch.tensor(self.mean).to(x.device).float()
                return x + x_null
            def return_embs():
                x = self.text_embeddings.weight
                x_null = self.ID_embeddings.weight
                #x_null = x_null @ self.U_null.T.to(x.device)
                x_null = x_null @ self.U_null.T.to(x.device) + torch.tensor(self.mean).to(x.device).float()
                return x + x_null
            
        elif ID_space == "euclidean"and inject_space == "euclidean":
            P = self.U
            self.ID_embeddings = nn.Embedding(
                num_embeddings=self.item_num+1,
                embedding_dim=base_embs.shape[-1],
            )
            if standardization:
                singulars = np.ones(base_embs.shape[-1])
                singulars[:-self.nullity] = np.sqrt(1/self.S[:-self.nullity])
                P = P.dot(np.dot(np.diag(singulars),P.T))
                #base_embs = base_embs.dot(P)
                base_embs = (base_embs-self.mean).dot(P) + self.mean
            self.UUT = torch.tensor(np.dot(self.U_null,self.U_null.T)).float()
            def injection(id):
                x = self.text_embeddings(id)
                x_null = self.ID_embeddings(id)
                #x_null = x_null @ self.UUT.to(x.device)
                x_null = (x_null-torch.tensor(self.mean).to(x.device).float()) @ self.UUT.to(x.device) + torch.tensor(self.mean).to(x.device).float()
                return x + x_null
            def return_embs():
                x = self.text_embeddings.weight
                x_null = self.ID_embeddings.weight
                #x_null = x_null @ self.UUT.to(x.device)
                x_null = (x_null-torch.tensor(self.mean).to(x.device).float()) @ self.UUT.to(x.device) + torch.tensor(self.mean).to(x.device).float()
                return x + x_null
        
        
        padding_vector = np.random.randn(base_embs.shape[-1])  
        base_embs = np.vstack([base_embs, padding_vector])
        self.text_embeddings = nn.Embedding.from_pretrained(torch.tensor(base_embs,dtype=torch.float32), freeze=True, padding_idx=self.item_num)
        
        return injection, return_embs
            
class DreamRec_backbone(nn.Module):
    def __init__(self, device, **key_words):
        super(DreamRec_backbone, self).__init__()
        data_statis = pd.read_pickle(os.path.join(key_words["language_embs_path"], 'data_statis.df'))  
        self.seq_len = data_statis['seq_size'][0]  
        self.item_num = data_statis['item_num'][0]

        self.device = device
        self.dropout = key_words["dropout_rate"]
        self.device = device

        #self.item_emb = NSI(data_directory, args.emb_std, args.emb_type, args.emb_dim, args.emb_init_type, args.null_thres, args.null_dim,\
        #    args.standardization, args.cover, args.ID_space, args.inject_space)
        
        self.hidden_dim = key_words["hidden_dim"]
        
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_dim,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.seq_len,
            embedding_dim=self.hidden_dim
        )
        # emb_dropout is added
        #self.LT = PWLayer(hidden_size,hidden_size, dropout)
        self.emb_dropout = nn.Dropout(self.dropout)
        self.ln_1 = nn.LayerNorm(self.hidden_dim)
        self.ln_2 = nn.LayerNorm(self.hidden_dim)
        self.ln_3 = nn.LayerNorm(self.hidden_dim)
        self.mh_attn = MultiHeadAttention(self.hidden_dim, self.hidden_dim, key_words["num_heads"], self.dropout)
        self.feed_forward = PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, self.dropout)

        self.step_mlp = TimestepEmbedder(self.hidden_dim)

        self.diffuser = FinalLayer(self.hidden_dim, self.hidden_dim)

    def embed_ID(self, x):
        #return self.item_embeddings.ID_embeddings(x)
        pass
    
    def return_item_emb(self,):
        #return self.item_embeddings.ID_embeddings.weight 
        pass
    
    def cacu_condition(self, sequences, p):
        #hidden
        B, L = sequences.shape
        inputs_emb = self.embed_ID(sequences)
        inputs_emb += self.positional_embeddings(torch.arange(self.seq_len).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(sequences, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        #state_hidden = extract_axis_1(ff_out, len_states - 1)
        condition = ff_out[:,-1].squeeze()
        # classifier-free guidance
        if p is not None:
            B, D = condition.shape[0], condition.shape[1]
            mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
            maske1d = mask1d.view(B, 1)
            mask = torch.cat([maske1d] * D, dim=1)
            mask = mask.to(self.device)
            condition = condition * mask + self.none_embedding(torch.tensor([0]).to(self.device)) * (1-mask)
        return condition  
    
    def forward(self, x, condition, step):

        t = self.step_mlp(step)
        res = self.diffuser(x,t,condition)
        return res

    def forward_uncon(self, x, step):
        condition = self.none_embedding(torch.tensor([0]).to(self.device))
        condition = torch.cat([condition.view(1, self.hidden_dim)]*x.shape[0], dim=0)

        t = self.step_mlp(step)
        res = self.diffuser(x,t,condition)
            
        return res
    
    
    def predict(self, sequences, diff):

        condition = self.cacu_condition(sequences, None)
        
        oracle_embs = diff.sample_from_noise(self.forward, self.forward_uncon, condition, self.hidden_dim)

        item_embs = self.return_item_emb()[:-1]
        scores = torch.matmul(oracle_embs, item_embs.transpose(0, 1))

        return oracle_embs, scores

### pure ID embeddings
class DreamRec(DreamRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)

        self.item_embeddings = Item_Embedding("ID", **key_words)
 
    def embed_ID(self, x):
        return self.item_embeddings.ID_embeddings(x)
    
    def return_item_emb(self,):
        return self.item_embeddings.ID_embeddings.weight 
    
class MoRec(DreamRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)

        self.item_embeddings = Item_Embedding("AP", **key_words)
        self.language_dim = self.item_embeddings.language_dim
        self.adapter = nn.Sequential(
            nn.Linear(self.language_dim, key_words['hidden_dim']),
            nn.GELU()
        )
 
 
    def embed_ID(self, x):
        language_embs = self.item_embeddings.language_embeddings(x)
        return self.adapter(language_embs)
    
    def return_item_emb(self,):
        language_embs = self.item_embeddings.language_embeddings.weight
        return self.adapter(language_embs) 

class iDreamRec(DreamRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)

        self.item_embeddings = Item_Embedding("WAP", **key_words)
        self.language_dim = self.item_embeddings.language_dim
        
    def embed_ID(self, x):
        language_embs = self.item_embeddings.language_embeddings(x)
        return language_embs
    
    def return_item_emb(self,):
        language_embs = self.item_embeddings.language_embeddings.weight
        return language_embs

class WhitenRec(DreamRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)

        self.item_embeddings = Item_Embedding("WAP", **key_words)
        self.language_dim = self.item_embeddings.language_dim
        self.adapter = nn.Sequential(
            nn.Linear(self.language_dim, key_words['hidden_dim']),
            nn.GELU()
        )
        
    def embed_ID(self, x):
        language_embs = self.item_embeddings.language_embeddings(x)
        return self.adapter(language_embs)
    
    def return_item_emb(self,):
        language_embs = self.item_embeddings.language_embeddings.weight
        return self.adapter(language_embs) 
    
class LLMInit(DreamRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        
        self.item_embeddings = Item_Embedding("SI", **key_words)
        #self.language_dim = self.item_embeddings.language_dim
 
    def embed_ID(self, x):
        return self.item_embeddings.ID_embeddings(x)
    
    def return_item_emb(self,):
        return self.item_embeddings.ID_embeddings.weight 

class RLMRec(DreamRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        
        self.item_embeddings = Item_Embedding("SR", **key_words)
        self.language_dim = self.item_embeddings.language_dim
        if key_words['SR_aligement_type'] == 'con':
            self.reconstructor = nn.Sequential(
                nn.Linear(self.language_dim, (self.language_dim + key_words['hidden_dim']) // 2),
                nn.LeakyReLU(),
                nn.Linear((self.language_dim + key_words['hidden_dim']) // 2, key_words['hidden_dim'])
            )
        elif key_words['SR_aligement_type'] == 'gen':
            self.reconstructor = nn.Sequential(
                nn.Linear(key_words['hidden_dim'], (self.language_dim + key_words['hidden_dim']) // 2),
                nn.LeakyReLU(),
                nn.Linear((self.language_dim + key_words['hidden_dim']) // 2, self.language_dim)
            )
 
    def embed_ID(self, x):
        return self.item_embeddings.ID_embeddings(x)
    
    def return_item_emb(self,):
        return self.item_embeddings.ID_embeddings.weight
    
    def reconstruct_gen_loss(self, ):
        rec_language_embs = self.reconstructor(self.return_item_emb()[:-1]) # self.return_item_emb()[-1] is the padding embedding
        language_embs = self.item_embeddings.language_embeddings.weight
        rec_language_embs = F.normalize(rec_language_embs, p=2, dim=-1)
        language_embs = F.normalize(language_embs, p=2, dim=-1)
        return 1 - (rec_language_embs * language_embs).sum() / self.item_num
    
    def reconstruct_con_loss(self, ):
        language_embs = self.item_embeddings.language_embeddings.weight
        rec_ID_embs = self.reconstructor(language_embs) # self.return_item_emb()[-1] is the padding embedding
        ID_embs = self.return_item_emb()[:-1]
        rec_ID_embs = F.normalize(rec_ID_embs, p=2, dim=-1)
        ID_embs = F.normalize(ID_embs, p=2, dim=-1)
        return 1 - (rec_ID_embs * ID_embs).sum() / self.item_num
    
class UniSRec(DreamRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        
        self.item_embeddings = Item_Embedding("AP", **key_words)
        self.language_dim = self.item_embeddings.language_dim
        self.adapter = MoEAdaptorLayer(
            8,
            [self.language_dim, key_words['hidden_dim']],
            0.2
        )
 
 
    def embed_ID(self, x):
        language_embs = self.item_embeddings.language_embeddings(x)
        return self.adapter(language_embs)
    
    def return_item_emb(self,):
        language_embs = self.item_embeddings.language_embeddings.weight
        return self.adapter(language_embs) 
     
class LLMESR(DreamRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        
        self.item_embeddings = Item_Embedding("Dual_view", **key_words)
        self.language_dim = self.item_embeddings.language_dim
        self.adapter = nn.Sequential(
            nn.Linear(self.language_dim, (self.language_dim + key_words['hidden_dim']) // 2),
            nn.Linear((self.language_dim + key_words['hidden_dim']) // 2, key_words['hidden_dim'])
        )
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=2*self.hidden_dim,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        self.diffuser = FinalLayer(self.hidden_dim, 2*self.hidden_dim, input_channels = 5)
        
        self.language2ID = Multi_CrossAttention(self.hidden_dim, self.hidden_dim, 2)
        self.ID2language = Multi_CrossAttention(self.hidden_dim, self.hidden_dim, 2)
        
        self.reg = Contrastive_Loss2()
 
    def embed_ID_text(self, x):
        language_embs = self.item_embeddings.language_embeddings(x)
        ID_embs = self.item_embeddings.ID_embeddings(x)
        return ID_embs, self.adapter(language_embs)
    
    def embed_ID(self, x):
        ID_embs, language_embs = self.embed_ID_text(x)
        return torch.cat([ID_embs, language_embs], dim=-1)
    
    def return_item_emb(self,):
        ID_embs = self.item_embeddings.ID_embeddings.weight
        language_embs = self.item_embeddings.language_embeddings.weight
        language_embs = self.adapter(language_embs)
        return torch.cat([ID_embs, language_embs], dim=-1) 
    
    def cacu_condition(self, sequences, p):
        
        inputs_id_emb, inputs_text_emb = self.embed_ID_text(sequences)
        inputs_text_emb += self.positional_embeddings(torch.arange(self.seq_len).to(self.device))
        inputs_id_emb += self.positional_embeddings(torch.arange(self.seq_len).to(self.device))
        
        text_seq = self.emb_dropout(inputs_text_emb)
        id_seq = self.emb_dropout(inputs_text_emb)
        
        cross_id_seqs = self.language2ID(text_seq, id_seq, sequences, self.item_num)
        cross_text_seqs = self.ID2language(id_seq, text_seq, sequences, self.item_num)
        cross_id_seqs = 1 * cross_id_seqs + 0 * id_seq
        cross_text_seqs = 1 * cross_text_seqs + 0 * text_seq

        mask = torch.ne(sequences, self.item_num).float().unsqueeze(-1).to(self.device)
        cross_id_seqs *= mask
        seq_normalized = self.ln_1(cross_id_seqs)
        mh_attn_out = self.mh_attn(seq_normalized, cross_id_seqs)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        id_logits = ff_out[:,-1].squeeze()
        
        #mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        cross_text_seqs *= mask
        seq_normalized = self.ln_1(cross_text_seqs)
        mh_attn_out = self.mh_attn(seq_normalized, cross_text_seqs)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        text_logits = ff_out[:,-1].squeeze()

        condition = torch.cat([id_logits, text_logits], dim=-1)

        if p is not None:
            B, D = condition.shape[0], condition.shape[1]
            mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
            maske1d = mask1d.view(B, 1)
            mask = torch.cat([maske1d] * D, dim=1)
            mask = mask.to(self.device)
            condition = condition * mask + self.none_embedding(torch.tensor([0]).to(self.device)) * (1-mask)

        return condition
    
    def forward_uncon(self, x, step):
        condition = self.none_embedding(torch.tensor([0]).to(self.device))
        condition = torch.cat([condition.view(1, 2*self.hidden_dim)]*x.shape[0], dim=0)

        t = self.step_mlp(step)
        res = self.diffuser(x,t,condition)
            
        return res
    
    def predict(self, sequences, diff):

        condition = self.cacu_condition(sequences, None)
        
        oracle_embs = diff.sample_from_noise(self.forward, self.forward_uncon, condition, 2*self.hidden_dim)

        item_embs = self.return_item_emb()[:-1]
        scores = torch.matmul(oracle_embs, item_embs.transpose(0, 1))

        return oracle_embs, scores
    
    def reg_loss(self, sequences):
        unfold_item_id = torch.masked_select(sequences, sequences!=self.item_num)
        language_emb, id_emb = self.embed_ID_text(unfold_item_id)
        reg_loss = self.reg(language_emb, id_emb)
        return reg_loss
    
class AlphaFuse(DreamRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)

        self.item_embeddings = Item_Embedding("AF", **key_words)
        #self.language_dim = self.item_embeddings.language_dim
        self.nullity = self.item_embeddings.nullity
        self.cover = key_words["cover"]
 
    def embed_ID(self, x):
        language_embs = self.item_embeddings.language_embeddings(x)
        ID_embs = self.item_embeddings.ID_embeddings(x)
        if self.cover:
            return torch.cat((language_embs, ID_embs), dim=-1)
        else:
            fuse_embs = language_embs.clone()
            fuse_embs[...,-self.nullity:] = language_embs[...,-self.nullity:] + ID_embs
        return fuse_embs
    
    def return_item_emb(self,):
        language_embs = self.item_embeddings.language_embeddings.weight
        ID_embs = self.item_embeddings.ID_embeddings.weight
        if self.cover:
            return torch.cat((language_embs, ID_embs), dim=-1)
        else:
            fuse_embs = language_embs.clone()
            fuse_embs[...,-self.nullity:] = language_embs[...,-self.nullity:] + ID_embs
        return fuse_embs
    

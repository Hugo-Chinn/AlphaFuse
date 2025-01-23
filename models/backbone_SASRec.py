import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from SASmodules import SASRec
from models.modules import *

class Item_Embedding(nn.Module):
    def __init__(self, emb_pipline, **key_words):
        super(Item_Embedding, self).__init__()
        data_statis = pd.read_pickle(os.path.join(key_words["language_embs_path"], 'data_statis.df'))  
        self.state_size = data_statis['seq_size'][0]  
        self.item_num = data_statis['item_num'][0]
        self.construct_item_embeddings(emb_pipline, **key_words)
            
    def construct_item_embeddings(self, emb_pipline, **key_words):
        if emb_pipline == "ID":
            self.init_ID_embedding(key_words["hidden_dim"], key_words["ID_embs_init_type"])
        elif emb_pipline == "SI": # semantic initialization
            self.init_ID_embedding(key_words["hidden_dim"], "language_embeddings", **key_words)
        elif emb_pipline == "SR": # semantic reconstruction
            self.init_ID_embedding(key_words["hidden_dim"], key_words["ID_embs_init_type"], **key_words)
            language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"], key_words["language_embs_scale"])
            #padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
            #language_embs = np.vstack([language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs,dtype=torch.float32),
                freeze=True,
                )
        elif emb_pipline == "Dual_view": # Dual view modeling of LLNESR
            self.init_ID_embedding(key_words["hidden_dim"], "language_embeddings", **key_words)
            language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"], key_words["language_embs_scale"])
            padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
            language_embs = np.vstack([language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs,dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
                )
        elif emb_pipline == "AP": # Adaptive Projection
            language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"], key_words["language_embs_scale"])
            padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
            language_embs = np.vstack([language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs,dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
                )
        elif emb_pipline == "WAP": # Adaptive Projection for whitened language embeddings
            key_words["item_frequency_flag"] = False
            key_words['standardization'] = True
            language_embs = self.semantic_space_decomposion( None, **key_words)
            padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
            language_embs = np.vstack([language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(language_embs,dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
                )
        elif emb_pipline == "AF": # AlphaFuse
            cliped_language_embs = self.semantic_space_decomposion( key_words["hidden_dim"], **key_words)
            padding_emb = np.random.rand(cliped_language_embs.shape[1])  # padding ID embedding
            cliped_language_embs = np.vstack([cliped_language_embs, padding_emb])
            self.language_embeddings = nn.Embedding.from_pretrained(
                torch.tensor(cliped_language_embs,dtype=torch.float32),
                freeze=True,
                padding_idx=self.item_num
                )
            self.init_ID_embedding(self.nullity, key_words["ID_embs_init_type"])
            #self.init_ID_embedding(self.nullity, "zeros")        
        
    def load_language_embeddings(self, directory, language_model_type, scale):
        language_embs = pd.read_pickle(os.path.join(directory, language_model_type + '_emb.pickle'))
        self.item_num = len(language_embs)
        self.language_dim = len(language_embs[0])
        return np.stack(language_embs) * scale
    
    def init_ID_embedding(self, ID_dim, init_type, **key_words):
        if init_type == "language_embeddings":
            language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"], key_words["language_embs_scale"])
            if self.language_dim == ID_dim:
                padding_emb = np.random.rand(language_embs.shape[1])  # padding ID embedding
                language_embs = np.vstack([language_embs, padding_emb])
                #language_embs = np.vstack([language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.item_num
                    )
            else:
                clipped_language_embs = self.semantic_space_decomposion(ID_dim, **key_words)
                padding_emb = np.random.rand(clipped_language_embs.shape[1])  # padding ID embedding
                clipped_language_embs = np.vstack([clipped_language_embs, padding_emb])
                #language_embs = np.vstack([language_embs, padding_emb])
                self.ID_embeddings = nn.Embedding.from_pretrained(
                    torch.tensor(clipped_language_embs, dtype=torch.float32),
                    freeze=False,
                    padding_idx=self.item_num
                    )
        else:
            self.ID_embeddings = nn.Embedding(
                num_embeddings=self.item_num+1,
                embedding_dim=ID_dim,
            )
            if init_type == "uniform":
                nn.init.uniform_(self.ID_embeddings.weight, a=0.0, b=1.0)
            elif init_type == "normal":
                nn.init.normal_(self.ID_embeddings.weight, 0, 1)
            elif init_type == "zeros":
                nn.init.zeros_(self.ID_embeddings.weight)
            elif init_type == "ortho":
                nn.init.orthogonal_(self.ID_embeddings.weight, gain=1.0)
            elif init_type == "xavier":
                nn.init.xavier_uniform_(self.ID_embeddings.weight, gain=1.0)
            elif init_type == "sparse":
                nn.init.sparse_(self.ID_embeddings.weight, 0.01, std=1)
            else:
                raise NotImplementedError("This kind of init for ID embeddings is not implemented yet.")
                
    def semantic_space_decomposion(self, clipped_dim, **key_words):
        language_embs = self.load_language_embeddings(key_words["language_embs_path"], key_words["language_model_type"], key_words["language_embs_scale"])
        if not key_words["item_frequency_flag"]:
            # The default item distribution is a uniform distribution.
            self.language_mean = np.mean(language_embs, axis=0)
            cov = np.cov( language_embs - self.language_mean, rowvar=False)
        else:
            items_pop = np.load(os.path.join(key_words["language_embs_path"], 'items_pop.npy'))
            items_freq_scale = 1.0 / items_pop.sum()
            items_freq = (items_pop*items_freq_scale).reshape(-1, 1)
            self.language_mean = np.sum(language_embs*items_freq, axis=0)
            cov = np.cov( (language_embs - self.language_mean)*np.sqrt(items_freq), rowvar=False)
            #raise NotImplementedError("Custom item distribution is not implemented yet.")
        U, S, _ = np.linalg.svd(cov, full_matrices=False)
        
        if key_words["null_thres"] is not None:
            indices_null = np.where(S <= key_words["null_thres"])[0]
            self.nullity = len(indices_null)
        elif key_words["null_dim"] is not None:
            self.nullity = key_words["null_dim"]
        #print("The Nullity is", self.nullity)
        #self.squared_singular_values = S
        #self.language_bases = U
        if clipped_dim is None:
            clipped_dim = self.language_dim
        if key_words["cover"]:
            clipped_dim = clipped_dim - self.nullity
        Projection_matrix = U[...,:clipped_dim]
        
        if key_words['standardization']:
            Diagnals = np.sqrt(1/S)[:clipped_dim]
            Projection_matrix = Projection_matrix.dot(np.diag(Diagnals)) # V_{\lamda} into V_1
        clipped_language_embs = (language_embs-self.language_mean).dot(Projection_matrix)
        return clipped_language_embs

class SASRec_backbone(nn.Module):
    def __init__(self, device, **key_words):
        super(SASRec_backbone, self).__init__()
        
        data_statis = pd.read_pickle(os.path.join(key_words["language_embs_path"], 'data_statis.df'))  
        self.seq_len = data_statis['seq_size'][0]  
        self.item_num = data_statis['item_num'][0]
        #self.item_embeddings = Item_Embedding("ID", **key_words)
        #self.item_num = item_num
        #self.seq_len = seq_len
        
        self.dropout = key_words["dropout_rate"]
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        #self.language_dim = self.item_embeddings.language_dim
        self.hidden_dim = key_words["hidden_dim"]

        self.positional_embeddings = nn.Embedding(
            num_embeddings=self.seq_len,
            embedding_dim=self.hidden_dim
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(self.dropout)
        self.ln_1 = nn.LayerNorm(self.hidden_dim)
        self.ln_2 = nn.LayerNorm(self.hidden_dim)
        self.ln_3 = nn.LayerNorm(self.hidden_dim)
        self.mh_attn = MultiHeadAttention(self.hidden_dim, self.hidden_dim, key_words["num_heads"], self.dropout)
        self.feed_forward = PositionwiseFeedForward(self.hidden_dim, self.hidden_dim, self.dropout)
        #self.s_fc = nn.Linear(self.hidden_size, self.item_num)
        # self.ac_func = nn.ReLU()

    def embed_ID(self, x):
        #return self.item_embeddings.ID_embeddings(x)
        pass
    
    def return_item_emb(self,):
        #return self.item_embeddings.ID_embeddings.weight 
        pass
    
    def forward(self, sequences):
 
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
        logits = ff_out[:,-1].squeeze()
        return logits
    
    def predict(self, sequences):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        state_hidden = self.forward(sequences)
        item_embs = self.return_item_emb() 
        scores = torch.matmul(state_hidden, item_embs[:-1].transpose(0, 1))  # (B,|I|)
        return scores

    def calculate_ce_loss(self, sequences, target):
        seq_output = self.forward(sequences)
        item_embs = self.return_item_emb() # (|I|,d)
        #item_embs = self.item_emb.return_embs()
        logits = torch.matmul(seq_output, item_embs[:-1].transpose(0, 1))
        loss = self.ce_loss(logits, target)
        return loss
    
    def calculate_bce_loss(self, sequences, target, neg_ratio, emb_type="both"):
        
        # negative sampling
        # 生成与正样本对应的负样本目标张量
        # 首先生成所有可能的负样本
        #sequences_set = set(sequences.view(-1).tolist())
        batch_size = target.shape[0]
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        #expanded_sequences = sequences.view(batch_size, -1, 1).expand(batch_size, sequences.shape[1], neg_ratio).cpu()
        #mask_target = neg_samples == expanded_target
        #mask_sequences = (neg_samples.unsqueeze(1).expand(-1, sequences.shape[1], -1) == expanded_sequences).any(dim=1)
        #mask = mask_target | mask_sequences
        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
            #mask_target = neg_samples == expanded_target
            #mask_sequences = (neg_samples.unsqueeze(1).expand(-1, sequences.shape[1], -1) == expanded_sequences).any(dim=1)
            #mask = mask_target | mask_sequences
        target_neg = neg_samples.to(target.device)

        #pos_embs = self.item_embeddings(target)
        pos_embs = self.embed_ID(target)
        neg_embs = self.embed_ID(target_neg)
        
        log_feats = self.forward(sequences)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats.unsqueeze(1) * neg_embs).sum(dim=-1)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape, device=self.device)
        loss = self.bce_loss(pos_logits, pos_labels)
        loss += self.bce_loss(neg_logits, neg_labels)

        return loss
    
    def calculate_infonce_loss(self, sequences, target, neg_ratio, temperature, emb_type="both"):
        
        # negative sampling
        # 生成与正样本对应的负样本目标张量
        # 首先生成所有可能的负样本
        #sequences_set = set(sequences.view(-1).tolist())
        batch_size = target.shape[0]
        neg_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
        expanded_target = target.view(batch_size, 1).expand(batch_size, neg_ratio).cpu()
        expanded_sequences = sequences.view(batch_size, -1, 1).expand(batch_size, sequences.shape[1], neg_ratio).cpu()
        #mask_target = neg_samples == expanded_target
        #mask_sequences = (neg_samples.unsqueeze(1).expand(-1, sequences.shape[1], -1) == expanded_sequences).any(dim=1)
        #mask = mask_target | mask_sequences
        mask = neg_samples == expanded_target
        while mask.any():
            new_samples = torch.randint(0, self.item_num, (batch_size, neg_ratio))
            neg_samples = torch.where(mask, new_samples, neg_samples)
            mask = neg_samples == expanded_target
            #mask_target = neg_samples == expanded_target
            #mask_sequences = (neg_samples.unsqueeze(1).expand(-1, sequences.shape[1], -1) == expanded_sequences).any(dim=1)
            #mask = mask_target | mask_sequences
        target_neg = neg_samples.to(target.device)

        #pos_embs = self.item_embeddings(target)
        pos_embs = self.embed_ID(target)
        neg_embs = self.embed_ID(target_neg)
        log_feats = self.forward(sequences)
        
        log_feats = F.normalize(log_feats, p=2, dim=-1)
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs = F.normalize(neg_embs, p=2, dim=-1)
        #normed_log_feats = log_feats / torch.sqrt(1e-8 + log_feats.square().sum(-1, keepdim=True))
        #normed_pos_embs = pos_embs / torch.sqrt(1e-8 + pos_embs.square().sum(-1, keepdim=True))
        #normed_neg_embs = neg_embs / torch.sqrt(1e-8 + neg_embs.square().sum(-1, keepdim=True))
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1, keepdim=True)
        neg_logits = torch.bmm(neg_embs, log_feats.unsqueeze(-1)).squeeze(-1)
        
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits /= temperature
        
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)  # (batch_size,)
        loss = F.cross_entropy(logits, labels)
        return loss

### pure ID embeddings
class SASRec(SASRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)

        self.item_embeddings = Item_Embedding("ID", **key_words)
 
    def embed_ID(self, x):
        return self.item_embeddings.ID_embeddings(x)
    
    def return_item_emb(self,):
        return self.item_embeddings.ID_embeddings.weight 
    
class MoRec(SASRec_backbone):
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
    
class WhitenRec(SASRec_backbone):
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
    
class LLMInit(SASRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        
        self.item_embeddings = Item_Embedding("SI", **key_words)
        #self.language_dim = self.item_embeddings.language_dim
 
    def embed_ID(self, x):
        return self.item_embeddings.ID_embeddings(x)
    
    def return_item_emb(self,):
        return self.item_embeddings.ID_embeddings.weight 

class RLMRec(SASRec_backbone):
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
    
class UniSRec(SASRec_backbone):
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
     
class LLMESR(SASRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)
        
        self.item_embeddings = Item_Embedding("Dual_view", **key_words)
        self.language_dim = self.item_embeddings.language_dim
        self.adapter = nn.Sequential(
            nn.Linear(self.language_dim, int(self.language_dim / 2)),
            nn.Linear(int(self.language_dim / 2), key_words['hidden_dim'])
        )
        
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
    
    def forward(self, sequences):
        
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

        log_feats = torch.cat([id_logits, text_logits], dim=-1)

        return log_feats
    
    def reg_loss(self, sequences):
        unfold_item_id = torch.masked_select(sequences, sequences!=self.item_num)
        language_emb, id_emb = self.embed_ID_text(unfold_item_id)
        reg_loss = self.reg(language_emb, id_emb)
        return reg_loss
    
class AlphaFuse(SASRec_backbone):
    def __init__(self, device, **key_words):
        super().__init__(device, **key_words)

        self.item_embeddings = Item_Embedding("AF", **key_words)
        #self.language_dim = self.item_embeddings.language_dim
        self.nullity = self.item_embeddings.nullity
        self.cover = key_words["cover"]
 
    def embed_ID(self, x):
        language_embs = self.item_embeddings.language_embeddings(x)
        #fuse_embs = language_embs.clone()
        ID_embs = self.item_embeddings.ID_embeddings(x)
        if self.cover:
            return torch.cat((language_embs, ID_embs), dim=-1)
        else:
            fuse_embs = language_embs.clone()
            fuse_embs[...,-self.nullity:] = language_embs[...,-self.nullity:] + ID_embs
        return fuse_embs
    
    def return_item_emb(self,):
        language_embs = self.item_embeddings.language_embeddings.weight
        #fuse_embs = language_embs.clone()
        ID_embs = self.item_embeddings.ID_embeddings.weight
        if self.cover:
            return torch.cat((language_embs, ID_embs), dim=-1)
        else:
            fuse_embs = language_embs.clone()
            fuse_embs[...,-self.nullity:] = language_embs[...,-self.nullity:] + ID_embs
        return fuse_embs
    
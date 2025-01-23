import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

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
            print()
        Projection_matrix = U[...,:clipped_dim]
        
        if key_words['standardization']:
            Diagnals = np.sqrt(1/S)[:clipped_dim]
            Projection_matrix = Projection_matrix.dot(np.diag(Diagnals)) # V_{\lamda} into V_1
        clipped_language_embs = (language_embs-self.language_mean).dot(Projection_matrix)
        return clipped_language_embs


class AlphaFuse_embs(nn.Module):
    def __init__(self, data_directory, emb_std, emb_type, emb_dim, emb_init_type, null_thres, null_dim, standardization, cover, ID_space, inject_space):
        super(AlphaFuse_embs, self).__init__()
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
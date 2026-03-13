
import pickle
import tqdm
import torch
import naskit as nsk
import math
import random
import numpy as np
from dataclasses import dataclass
from .dictionary import mono_pair_dict, dimer_pair_dict
from typing import Iterable

# используется для проверки сиквенсов в msa
class SequenceError(ValueError):
    pass

class AlinaDataset:
    
    def __init__(self, 
                 nas, 
                 dimer_embeddings: bool,
                 with_adjacency: bool = True
                ):
        
        self.nas = nas
        self.X = [None]*len(nas)
        self.dimer_embeddings = dimer_embeddings
        self.with_adjacency = with_adjacency

        self.seq2matrix_func = self.dimer_seq2matrix if self.dimer_embeddings else self.mono_seq2matrix
        self.cache_dtype = torch.uint16 if self.dimer_embeddings else torch.uint8
    
    
    def __len__(self):
        return len(self.nas)

    
    def __getitem__(self, n):
        na = self.nas[n]
        
        if self.X[n] is not None:
            x = self.X[n].to(torch.int32)
        else:
            x = self.seq2matrix_func(na)    
            self.X[n] = x.to(self.cache_dtype)
        
        y = torch.FloatTensor(na.get_adjacency()) if self.with_adjacency else None
        return x, y

    
    @staticmethod
    def dimer_seq2matrix(na):
        leng = len(na)
        M = torch.zeros((leng, leng), dtype=torch.int32)
        for n in range(leng):
            for p in range(n-1):
                fx = na[n]
                fy = na[p]
    
                fx1 = ''
                fy1 = ''
              
                if n<leng-1:
                    fx1 = na[n+1]
                if p<leng-1:
                    fy1 = na[p+1]
                    
                M[n][p] = dimer_pair_dict[fx+fx1+'/'+fy1+fy]
                M[p][n] = dimer_pair_dict[fy+fy1+'/'+fx1+fx]
        return M

    
    @staticmethod
    def mono_seq2matrix(na):
        leng = len(na)
        M = torch.zeros((leng, leng), dtype=torch.int32)
        for n in range(leng):
            for p in range(n-1):
                fx = na[n]
                fy = na[p]
    
                M[n][p] = mono_pair_dict[fx+fy]
                M[p][n] = mono_pair_dict[fx+fy]
        return M

    
    def precache(self):
        for i in tqdm.tqdm(range(len(self))):
            _ = self[i]

    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({"nas":self.nas, "X":self.X, "dimer_embeddings":self.dimer_embeddings}, f)


    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        ds = cls(data["nas"], data["dimer_embeddings"])
        ds.X = data["X"]
        return ds


    def __add__(self, other):
        self.nas += other.nas
        self.X += other.X
        return self
        

def make_collate(max_len: int, center_pad: bool):
    def collate_fn(dps):
        with_adjacency = dps[0][1] is not None
        
        X = torch.zeros((len(dps), max_len, max_len), dtype=torch.int32)
        Y = torch.zeros((len(dps), max_len, max_len), dtype=torch.float32) if with_adjacency else None
        L, Sl = [], []
        
        for i, (x, y) in enumerate(dps):
            n = x.shape[0]

            left = (max_len - n)//2 if center_pad else 0
            until = (left+n) if center_pad else n

            X[i, left:until, left:until] = x
            if with_adjacency:
                Y[i, left:until, left:until] = y

            L.append(left)
            Sl.append(n)
        
        return X, Y, L, Sl
    
    return collate_fn

@dataclass
class ComplexData:
    msa:           torch.Tensor
    true_adj:      torch.Tensor
    best_coev_adj: torch.Tensor
    coev_adjs:     torch.Tensor
    offset:        tuple
    

####  MSAClassifier ####
class MSAClassifierDataset:
    def __init__(self, 
                 data: Iterable[nsk.NucleicAcid],
                 mode: str,
                 msa_sample_size: float = 1.,
                 ss_augment_range: tuple[float, float] = (0.2, 0.4),
                 augment_helix_len_range: tuple[int, int] = (2,4),
                 
                 dimer_embeddings: bool = False,
                 center_pad: bool = False,
                 max_len: int = 256):

        assert (mode in ['augment', 'train', 'valid']), f"unknown 'mode' value, expected: 'augment', 'train', 'valid'"
        self.data = data
        #self.augment_struct = augment_struct

        is_cached = isinstance(data[0], ComplexData)
        if is_cached:
            self._cache = True
        else:
            self._cache = False
            # check up
            for i, na in enumerate(data):
                ls = []
                assert "msa" in na.meta.keys(), f"dp #{i}: meta must contain 'msa' field"
                for j, seq in enumerate(na.meta['msa']):
                    check_seq(seq, max_len, i, j)
                    ls.append(len(seq))
                    
                if mode=='valid':
                    assert "coev_struct" in na.meta.keys(), f"dp #{i}: 'valid' mode, but meta doesn't contain 'coev_struct' field"
                    ls.append(len(na.meta['coev_struct']))
                if mode=='train':
                    assert "coev_dbns" in na.meta.keys(), f"dp #{i}: 'train' mode, but meta doesn't contain 'coev_dbns' field"
                    ls = ls + [len(dbn) for dbn in na.meta['coev_dbns']]
                    
                ls.append(len(na.struct))
                assert len(set(ls)) == 1, f"dp #{i}, here are seq/ss(s) with different lengths"

        self.mode = mode
        self.msa_sample_size = msa_sample_size
        
        self.ss_augment_range = ss_augment_range
        self.augment_helix_len_range = augment_helix_len_range

        self.dimer_embeddings = dimer_embeddings
        self.center_pad = center_pad
        self.max_len = max_len

        self.seq2matrix_func = self.dimer_seq2matrix if self.dimer_embeddings else self.mono_seq2matrix
        self.cache_dtype = torch.uint16 if self.dimer_embeddings else torch.uint8

        self.dp_adjs_inds = torch.zeros(len(data), dtype=torch.int32)
    
    
    def __len__(self):
        return len(self.data)

    @staticmethod
    def dimer_seq2matrix(na):
        
        leng = len(na)
        M = torch.zeros((leng, leng), dtype=torch.int32)
        for n in range(leng):
            for p in range(n-1):
                fx = na[n]
                fy = na[p]
    
                fx1 = ''
                fy1 = ''
              
                if n<leng-1:
                    fx1 = na[n+1]
                if p<leng-1:
                    fy1 = na[p+1]
                    
                M[n][p] = dimer_pair_dict[fx+fx1+'/'+fy1+fy]
                M[p][n] = dimer_pair_dict[fy+fy1+'/'+fx1+fx]
        return M

    
    @staticmethod
    def mono_seq2matrix(na):
        
        leng = len(na)
        M = torch.zeros((leng, leng), dtype=torch.int32)
        for n in range(leng):
            for p in range(n-1):
                fx = na[n]
                fy = na[p]
    
                M[n][p] = mono_pair_dict[fx+fy]
                M[p][n] = mono_pair_dict[fx+fy]
        return M

    def get_msa_tensor(self, msa_list: Iterable[str]):

        msa = [self.seq2matrix_func(seq).to(self.cache_dtype) for seq in msa_list]
        X = torch.zeros((len(msa), self.max_len, self.max_len), dtype=torch.int32)
        
        n = msa[0].shape[0]
        left = (self.max_len - n)//2 if self.center_pad else 0
        until = (left+n) if self.center_pad else n
        
        for i, x in enumerate(msa):
            X[i, left:until, left:until] = x

        return X, left, until

    def get_coev_adjs_tensor(self, coev_dbns, coev_struct, L, Sl):

        coev_adjs_list = []
        for dbn in coev_dbns:
            if dbn == coev_struct:
                continue
            coev_adjs_list.append( torch.FloatTensor(nsk.NA(dbn).get_adjacency()) )
            
        coev_adjs_tensor = torch.zeros((len(coev_adjs_list), self.max_len, self.max_len),
                                       dtype=torch.int32)
        for i, adj in enumerate(coev_adjs_list):
            coev_adjs_tensor[i, L:Sl, L:Sl] = adj
            
        return coev_adjs_tensor
        
        
    # def augment_ss(self,
    #                na,
    #                patience = 5):
        
    #     n_max_pairs = len(na) // 2
    #     n_pairs = len(na.pairs)
        
    #     freq = self._rng.uniform(*self.ss_augment_range)
    #     seed = int(self._rng.integers(1e6))
    
    #     n_add_pairs = int(np.floor(n_pairs*freq))
    #     n_add_pairs = max(1, n_add_pairs)
    
    #     n_pairs = min(n_max_pairs, n_pairs + n_add_pairs)
        
    #     max_compl_ratio = 2*n_pairs/len(na)

    #     true_struct = na.struct
    #     na = nsk.algo.generate_ss(na,
    #                               min_helix_size=self.augment_helix_len_range[0],
    #                               max_helix_size=self.augment_helix_len_range[0],
    #                               max_compl_ratio=max_compl_ratio,
    #                               patience=patience, seed = seed)
        
    #     compl_ratio = 2*len(na.pairs)/len(na)
    #     if(compl_ratio<max_compl_ratio):
    #         na = nsk.algo.generate_ss(na,
    #                                   min_helix_size=1,
    #                                   max_helix_size=1,
    #                                   max_compl_ratio=max_compl_ratio,
    #                                   patience=patience, seed = seed)

    #     aug_struct = na.struct
    #     na.struct = true_struct
        
    #     return aug_struct
    
    def make_msa_sample(self, msa: torch.Tensor):

        n = msa.size(0)
        k = int(math.ceil(n*self.msa_sample_size))
        assert k<=n

        inds = torch.randperm(n)[:k]
        
        return msa[inds,:,:]

        
    def increment_data_point_ind(self, key: int):
        dp = self.data[key]
        lim = dp.coev_adjs.size(0)
        
        curr_ind = self.dp_adjs_inds[key]
        next_ind = (curr_ind+1) % lim
        if next_ind==0:
            adjs = dp.coev_adjs
            inds = torch.randperm(lim)
            self.data[key].coev_adjs = adjs[inds]
            
        self.dp_adjs_inds[key] = next_ind

    def make_data_point(self, key: int):
        
        na = self.data[key]

        msa, L, Sl = self.get_msa_tensor(na.meta['msa'])
        
        Y = None
        if na.struct is not None:
            Y = torch.zeros((self.max_len, self.max_len), dtype=torch.float32)
            Y[L:Sl, L:Sl] = torch.FloatTensor(na.get_adjacency())
            
        coev_struct_dbn, coev_struct_adj = None, None
        if 'coev_struct' in na.meta.keys():
            coev_struct_dbn = na.meta['coev_struct']
            adj = torch.FloatTensor(nsk.NA(coev_struct_dbn).get_adjacency())
            coev_struct_adj = torch.zeros((self.max_len, self.max_len), dtype=torch.float32)
            coev_struct_adj[L:Sl, L:Sl] = adj

        coev_adjs = None
        if 'coev_dbns' in na.meta.keys():
            coev_adjs = self.get_coev_adjs_tensor(na.meta['coev_dbns'], coev_struct_dbn, L, Sl)

        return ComplexData(msa, Y, coev_struct_adj, coev_adjs, (L,Sl))
        
        
    def __getitem__(self, key: int):

        if not isinstance(self.data[key], ComplexData):
            return self.data[key]
            
        dp = self.data[key]
        
        y     = dp.true_adj
        msa   = self.make_msa_sample(dp.msa)
        L, Sl = dp.offset
        
        if self.mode == 'augment':
            # заглушка
            adj = torch.zeros((self.max_len, self.max_len), dtype=torch.float32)
        elif self.mode == 'valid':
            adj = dp.best_coev_adj
        elif self.mode == 'train':
            adj = dp.coev_adjs[self.dp_adjs_inds[key].item()].squeeze()
            self.increment_data_point_ind(key)
    
        return msa, adj, L, Sl, y

    def precache(self):
        if not self._cache:
            for i in tqdm.tqdm(range(len(self))):
                self.data[i] = self.make_data_point(i)
            self._cache = True
        else:
            print("Dataset is already cached!")

    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({"data":self.data,
                         "mode":self.mode,
                         "msa_sample_size":self.msa_sample_size,
                         "ss_augment_range":self.ss_augment_range,
                         "augment_helix_len_range":self.augment_helix_len_range,
                         "dimer_embeddings":self.dimer_embeddings,
                         "center_pad":self.center_pad,
                         "max_len":self.max_len}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        ds = cls(**data)
        return ds

def check_seq(seq, max_len, i, j):
    rn = set(seq) - {'A', 'U', 'G', 'C', 'N'}
    if len(rn)!=0:
        raise SequenceError(f'msa seq #{j} in dp #{i} contains unknown symbols: {tuple(rn)}')
    if len(seq)>max_len or len(seq)==0:
        raise SequenceError(f'msa seq #{j} in dp #{i} out of range (0, 256], got {len(seq)}')
        
    

def identity_collate(dps):
    assert len(dps) == 1
    return dps[0]









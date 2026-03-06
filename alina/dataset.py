import pickle
import tqdm
import torch
import naskit as nsk
import math
import numpy as np
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

####  MSAClassifier ####
class MSAClassifierDataset:
    def __init__(self, 
                 data: Iterable[nsk.NucleicAcid],
                 augment_struct: bool,
                 msa_sample_size: float = 1.,
                 ss_augment_range: tuple[float, float] = (0.2, 0.4),
                 augment_helix_len_range: tuple[int, int] = (2,4),
                 
                 cache: bool = True,
                 dimer_embeddings: bool = False,
                 center_pad: bool = False,
                 max_len: int = 256,
                 seed: int = 42
                ):
            
        self.data = data
        self.augment_struct = augment_struct
        
        # check up
        for i, na in enumerate(data):
            is_cached = isinstance(na.meta['msa'], torch.Tensor)
            if is_cached:
                continue

            ls = []
            for j, seq in enumerate(na.meta['msa']):
                check_seq(seq, max_len, i, j)
                ls.append(len(seq))
                
            if not self.augment_struct:
                assert "coev_struct" in na.meta.keys(), f"dp #{i}: augment_struct==False, but meta doesn't contain 'coev_struct' field"
                ls.append(len(na.meta['coev_struct']))
            
            ls.append(len(na.struct))
            assert len(set(ls)) == 1, f"dp #{i}, here are seq/ss with different length"

            self.data[i].meta['cached'] = False
                
        self.msa_sample_size = msa_sample_size
        
        self.ss_augment_range = ss_augment_range
        self.augment_helix_len_range = augment_helix_len_range

        self._cache = cache
        self.dimer_embeddings = dimer_embeddings
        self.center_pad = center_pad
        self.max_len = max_len
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.seq2matrix_func = self.dimer_seq2matrix if self.dimer_embeddings else self.mono_seq2matrix
        self.cache_dtype = torch.uint16 if self.dimer_embeddings else torch.uint8
    
    
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

    def get_msa_tensor(self, na):

        msa = [self.seq2matrix_func(seq).to(self.cache_dtype) for seq in na.meta['msa']]
        X = torch.zeros((len(msa), self.max_len, self.max_len), dtype=torch.int32)
        
        n = msa[0].shape[0]
        left = (self.max_len - n)//2 if self.center_pad else 0
        until = (left+n) if self.center_pad else n
        
        for i, x in enumerate(msa):
            X[i, left:until, left:until] = x

        return X, left, until

    def get_adj_tensor(self, na):

        L, Sl = na.meta['offset']
        
        if self.augment_struct:
            #print(na)
            ss = self.augment_ss(na)
            adj = nsk.NA(ss).get_adjacency()
        else:
            adj = nsk.NA(na.meta['coev_struct']).get_adjacency()
            
        adj = torch.FloatTensor(adj)
        
        A = torch.zeros((self.max_len, self.max_len), dtype=torch.float32)
        A[L:Sl, L:Sl] = adj
        
        return A
        
        
    def augment_ss(self,
                   na,
                   patience = 5):
        
        n_max_pairs = len(na) // 2
        n_pairs = len(na.pairs)
        
        freq = self._rng.uniform(*self.ss_augment_range)
        seed = int(self._rng.integers(1e6))
    
        n_add_pairs = int(np.floor(n_pairs*freq))
        n_add_pairs = max(1, n_add_pairs)
    
        n_pairs = min(n_max_pairs, n_pairs + n_add_pairs)
        
        max_compl_ratio = 2*n_pairs/len(na)

        true_struct = na.struct
        na = nsk.algo.generate_ss(na,
                                  min_helix_size=self.augment_helix_len_range[0],
                                  max_helix_size=self.augment_helix_len_range[0],
                                  max_compl_ratio=max_compl_ratio,
                                  patience=patience, seed = seed)
        
        compl_ratio = 2*len(na.pairs)/len(na)
        if(compl_ratio<max_compl_ratio):
            na = nsk.algo.generate_ss(na,
                                      min_helix_size=1,
                                      max_helix_size=1,
                                      max_compl_ratio=max_compl_ratio,
                                      patience=patience, seed = seed)

        aug_struct = na.struct
        na.struct = true_struct
        
        return aug_struct

    def prepare_dp(self, key):
        
        dp = self.data[key]
        
        if not dp.meta['cached']:
            X, L, Sl = self.get_msa_tensor(dp)
            dp.meta['msa'] = X
            dp.meta['offset'] = (L, Sl)
            
            if dp.struct is not None:
                y =  torch.FloatTensor(dp.get_adjacency())
                Y = torch.zeros((self.max_len, self.max_len), dtype=torch.float32)
                Y[L:Sl, L:Sl] = y
                dp.meta['true_struct'] = Y
            else:
                dp.meta['true_struct'] = None
                
            dp.meta['cached'] = True
               
        return dp

    def make_msa_sample(self, X):

        n = X.shape[0]
        k = int(math.ceil(n*self.msa_sample_size))
        assert k<=n

        inds = self._rng.choice(n, size=k, replace=False)
        inds = torch.as_tensor(inds, device=X.device)
        
        X = X[inds,:,:]

        return X
        
    def __getitem__(self, key: int):

        is_cached = self.data[key].meta['cached']
        if is_cached:
            dp = self.data[key]
        else:
            dp = self.prepare_dp(key)
            if self._cache:
                self.data[key] = dp
                
        X = dp.meta['msa']
        X = self.make_msa_sample(X) # sample msa
        
        adj = self.get_adj_tensor(dp)
        
        L, Sl = dp.meta['offset']

        y = dp.meta['true_struct']
    
        return X, adj, L, Sl, y

    def precache(self):
        for i in tqdm.tqdm(range(len(self))):
            _ = self[i]

    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({"data":self.data,
                         "augment_struct":self.augment_struct,
                         "msa_sample_size":self.msa_sample_size,
                         "ss_augment_range":self.ss_augment_range,
                         "augment_helix_len_range":self.augment_helix_len_range,
                         "cache":self._cache,
                         "dimer_embeddings":self.dimer_embeddings,
                         "center_pad":self.center_pad,
                         "max_len":self.max_len,
                         "seed":self.seed}, f)

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









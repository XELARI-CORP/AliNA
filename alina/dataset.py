import pickle
import tqdm
import torch
from .dictionary import mono_pair_dict, dimer_pair_dict



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

    
    def __len__(self):
        return len(self.nas)

    
    def __getitem__(self, n):
        if self.dimer_embeddings:
            seq2matrix_func = self.dimer_seq2matrix
            cache_dtype = torch.uint16
        else:
            seq2matrix_func = self.mono_seq2matrix
            cache_dtype = torch.uint8
            
        na = self.nas[n]
        
        if self.X[n] is not None:
            x = self.X[n].to(torch.int32)
        else:
            x = seq2matrix_func(na)    
            self.X[n] = x.to(cache_dtype)
        
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
        X = torch.zeros((len(dps), max_len, max_len), dtype=torch.int32)
        Y = torch.zeros((len(dps), max_len, max_len), dtype=torch.float32)
        L, Sl = [], []
        
        for i, (x, y) in enumerate(dps):
            n = x.shape[0]

            if center_pad:
                left = (max_len - n)// 2
                X[i, left:(left+n), left:(left+n)] = x
                Y[i, left:(left+n), left:(left+n)] = y
            else:
                left = 0
                X[i, :n, :n] = x
                Y[i, :n, :n] = y

            L.append(left)
            Sl.append(n)
        
        return X, Y, L, Sl
    
    return collate_fn











import torch
import torch.nn as nn
import torch.nn.functional as F


# Self-Atten
### I just set the default value for d_model=2 that help for caculating 
class SelfAttention(nn.Module):
    def __init__(self, d_model=2, row_dim=0, col_dim = 1):
        
        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        
        self.row_dim = row_dim
        self.col_dim = col_dim
        
    def forward(self, token_encodings):
        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)
        
        ### Compute similarities scores: (q, K.T)
        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        
        ### Scale the similarities by dividing by sqrt (k.col_dim)
        scaled_sims = sims / torch.tensor(k.size(self.col_dim) ** .5)
        
        ### Apply Softmax to determine what is percent of each token'value
        atten_percents = F.softmax(scaled_sims, dim=self.col_dim)
        
        ## Scale the values by their associated percentages and add them up
        atten_scores = torch.matmul(atten_percents, v)
        return atten_scores


encodings_matrix = torch.tensor([[1.16, 0.23],
                                 [0.57, 1.36],
                                 [4.41, -2.16]])


selfAttention = SelfAttention(d_model=2,
                               row_dim=0,
                               col_dim=1)



print(selfAttention(encodings_matrix))
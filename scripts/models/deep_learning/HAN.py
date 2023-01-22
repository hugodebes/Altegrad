import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class AttentionWithContext(nn.Module):
    """
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    """
    
    def __init__(self, input_shape, return_coefficients=False, bias=True):
        super(AttentionWithContext, self).__init__()
        self.return_coefficients = return_coefficients

        self.W = nn.Linear(input_shape, input_shape, bias=bias)
        self.tanh = nn.Tanh()
        self.u = nn.Linear(input_shape, 1, bias=False)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.W.weight.data.uniform_(-initrange, initrange)
        self.W.bias.data.uniform_(-initrange, initrange)
        self.u.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz):
        # do not pass the mask to the next layers
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
    
    def forward(self, x, mask=None):
        uit = self.W(x) 
        uit = self.tanh(uit)
        ait = self.u(uit)
        a = torch.exp(ait)
        
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            a = a*mask.double()
        
        eps = 1e-9
        a = a / (torch.sum(a, axis=1, keepdim=True) + eps)
        weighted_input =  torch.sum(a*x,axis=1,keepdim=True) 
        if self.return_coefficients:
            return  [weighted_input, a]
        else:
            return  weighted_input


class AttentionBiGRU(nn.Module):
    def __init__(self, input_shape, n_units, id_to_aa, dropout=0):
        super(AttentionBiGRU, self).__init__()
        self.embedding = nn.Embedding(len(id_to_aa) + mfw_idx,
                                      d, 
                                      padding_idx=0)
        self.dropout = nn.Dropout(drop_rate)
        self.gru = nn.GRU(input_size=d,  
                          hidden_size=n_units,
                          num_layers=1,
                          bias=True,
                          batch_first=True,
                          bidirectional=True)
        self.attention = AttentionWithContext(n_units*2, #biderectional so hidden from left to right and from right to left 
                                              return_coefficients=True)


    def forward(self, part_ints):
        part_wv = self.embedding(part_ints)
        part_wv_dr = self.dropout(part_wv)
        part_wa, _ = self.gru(part_wv_dr) # RNN layer
        part_att_vec, aa_att_coeffs = self.attention(part_wa)
        part_att_vec_dr = self.dropout(part_att_vec)     
        return spart_att_vec_dr, aa_att_coeffs


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  
        part_att_vec_dr, aa_att_coeffs = self.module(x_reshape)
        # We have to reshape the output
        if self.batch_first:
            part_att_vec_dr = part_att_vec_dr.contiguous().view(x.size(0), -1, part_att_vec_dr.size(-1)) 
            aa_att_coeffs = aa_att_coeffs.contiguous().view(x.size(0), -1, aa_att_coeffs.size(-1)) 
        else:
            part_att_vec_dr = part_att_vec_dr.view(-1, x.size(1), part_att_vec_dr.size(-1))
            aa_att_coeffs = aa_att_coeffs.view(-1, x.size(1), aa_att_coeffs.size(-1)) 
        return part_att_vec_dr, aa_att_coeffs  


class HAN(nn.Module):
    def __init__(self, input_shape, n_units, id_to_aa, dropout=0):
        super(HAN, self).__init__()
        self.encoder = AttentionBiGRU(input_shape, n_units, id_to_aa, dropout)
        self.timeDistributed = TimeDistributed(self.encoder, True)
        self.dropout = nn.Dropout(drop_rate)
        self.gru = nn.GRU(input_size=n_units*2,
                          hidden_size=n_units,
                          num_layers=1,
                          bias=True,
                          batch_first=True,
                          bidirectional=True)
        self.attention = AttentionWithContext(n_units*2, 
                                              return_coefficients=True)
        self.lin_out = nn.Linear(n_units*2,  
                                 18)
        self.preds = nn.Softmax()

    def forward(self, part_ints):
        part_att_vecs_dr, aa_att_coeffs = self.timeDistributed(part_ints) #time distributed
        part_sa, _ = self.gru(part_att_vecs_dr)
        part_att_vec, part_att_coeffs = self.attention(part_sa)
        part_att_vec_dr = self.dropout(part_att_vec)
        part_att_vec_dr = self.lin_out(part_att_vec_dr)
        return self.preds(part_att_vec_dr), aa_att_coeffs, part_att_coeffs    




import torch
import torch.nn as nn
import torch.nn.functional as F 

config = {
    'activation_dim':768,
    'dict_dim':16384,
    'l1_coeff':3e-4,

}

class JumpReluAutoEncoder(nn.Module):

    def __init__(self,cfg):
        self.activation_dim = cfg['activation_dim']
        self.dict_dim = cfg['dict_dim']

        self.W_enc = nn.Parameter(torch.empty(self.activation_dim,self.dict_dim))
        self.b_enc = nn.Parameter(torch.empty(self.dict_dim))
        self.W_dec = nn.Parameter(torch.empty(self.dict_dim,self.activation_dim))
        self.b_dec = nn.Parameter(torch.empty(self.activation_dim))
        self.threshold = nn.Parameter(torch.empty(self.dict_dim))

        self.W_enc.data = t.randn_like(self.W_enc)
        self.W_enc.data = self.W_enc / self.W_enc.norm(dim=0, keepdim=True)
        self.W_dec.data = self.W_enc.data.clone().T

    def encode(self,x):

        x = x - self.b_dec
        pre_jump = x @ self.W_enc + self.b_enc
        f = nn.ReLU()(pre_jump * (pre_jump > threshold))
        f = f * self.W_dec.norm(dim=1)   # Google is bitch

        return f 

    def decode(self,f):

        f = f / self.W_dec.norm(dim=1)    # Google is bitch
        return f@self.W_dec + self.b_dec

    def forward(self,x):
        acts = self.encode(x)
        x_reconstruct = self.decode(acts)
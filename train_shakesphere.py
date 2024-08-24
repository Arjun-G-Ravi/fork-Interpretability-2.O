import tiktoken
import torch
import torch.nn as nn
from config import GPTConfig
from model import GPT
from inference import inference
from dataloader import DataLoaderLite

if __name__ == "__main__": 

   if torch.cuda.is_available():
      device = 'cuda'
   else:
      device = 'cpu'

   print(device)
   model = GPT(GPTConfig())
   model.to(device)

   train_loader = DataLoaderLite(B=4,T=32)
   optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
   
   for i in range(5):
    x,y = train_loader.next_batch()
    x,y = x.to(device),y.to(device)
    optimizer.zero_grad()
    logits,loss = model(x,y)
    print(i,":",loss.item())   
    loss.backward()
    optimizer.step()

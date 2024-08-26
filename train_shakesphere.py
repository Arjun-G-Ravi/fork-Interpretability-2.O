import tiktoken
import torch
import torch.nn as nn
from config import GPTConfig
from model import GPT
from inference import inference
from dataloader import DataLoaderLite
import time

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps

    elif it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


if __name__ == "__main__": 

   if torch.cuda.is_available():
      device = 'cuda'
   else:
      device = 'cpu'

   print(device)
   model = GPT(GPTConfig(vocab_size=50304))
   model.to(device)
   
   model = torch.compile(model)

   train_loader = DataLoaderLite(B=16,T=1024)
   torch.set_float32_matmul_precision('high')
   optimizer = model.configure_optimizers(weight_decay=0.1,learning_rate=6e-4,device=device)
   
   for step in range(max_steps):

      t0 = time.time()
      x,y = train_loader.next_batch()
      x,y = x.to(device),y.to(device)
      optimizer.zero_grad()

      with torch.autocast(device_type=device,dtype=torch.bfloat16) :
         logits,loss = model(x,y)   

      loss.backward()
      norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      lr = get_lr(step)

      for param_group in optimizer.param_groups:
        param_group['lr'] = lr  

      optimizer.step()
      torch.cuda.synchronize()

      t1 = time.time()
      dt = t1 - t0
      tokens_processed = train_loader.B * train_loader.T
      tokens_per_sec = tokens_processed/dt
      print(f"step {step:4d} | loss: {loss.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    
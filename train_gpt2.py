from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
from hellaswag import iterate_examples, render_example, get_most_likely_row

#----------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024 # maximum sequence length
    vocab_size: int = 50257 # 50000 bpe merges + 256 bytes token + 1 <|endoftext|> token # TODO: watch tokenizer video
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

#----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0 # scale the weights to account for accumulation on residual path

        # regularization
        self.n_heads = config.n_head
        self.n_embd = config.n_embd

        # bias for causal self-attention
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        # att = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # use Flash Attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

#----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # gelu activation function: https://arxiv.org/abs/1606.08415
        # reason: https://github.com/pytorch/pytorch/issues/39853
        # gelu fixes the relu's dead zone problem
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x

#----------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

#----------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # weights for token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # weights for positional embeddings
            h = nn.ModuleList([Block(self.config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** 0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    @classmethod
    def from_pretrained(cls, model_type: str):
        """Loads pretrained GPT-2 weights from hugging face"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt-2: {model_type}")

        # n_layer, n_head and n_embd are determined by the model type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768), # 124M parameters
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024), # 350M parameters
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280), # 774M parameters
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600), # 1558M parameters
        }[model_type]
        
        config_args["vocab_size"] = 50257 # always 50257 for GPT model
        config_args["block_size"] = 1024 # always 1024 for GPT model
        model = GPT(GPTConfig(**config_args))
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] # discard this mask / buffer, not a param

        # init hugging face / transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        assert len(sd_keys_hf) == len(sd_keys), f"mismatch: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def forward(self, idx, targets = None): # input is token indices
        # idx is of shape (B, T)
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx) # (B, T, C)
        pos_emb = self.transformer.wpe(pos) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)

        # forward the transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # forward the final layer norm and the classification head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print("using fused AdamW: ", use_fused)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

#----------------------------------------------------------------------------

import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # load the shards
        data_dir = "edu_fineweb10B"
        shards = os.listdir(data_dir)
        shards = [os.path.join(data_dir, shard) for shard in shards if shard.startswith(f"edufineweb_{split}")]
        shards.sort()
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        
        self.current_shard = 0
        self.tokens = load_tokens(shards[self.current_shard])

    
        # state
        self.current_position = self.B * self.T * self.process_rank

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T * self.num_processes
        
        if self.current_position + B * T * self.num_processes + 1 > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])

        return x, y
    

#----------------------------------------------------------------------------
# run training loop
import os
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import tiktoken


# NOTE: check how Tero do this in his code
# set up distributed data parallel
# torchrun command sets the env variables RANK, WORLD_SIZE, and LOCAL_RANK
ddp = int(os.environ.get("RANK", -1)) != -1 
print(f"ddp: {ddp}")
if ddp:
    assert torch.cuda.is_available(), "cuda is not available for distributed training"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will also do logging, checkpointing, etc.
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    master_process = True


#----------------------------------------------------------------------------

# TODO: add mps on apple. need to also change pyproject.toml
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288
B = 64 # micro batch size # TODO: reduce if necessary
T = 1024 # sequence length
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"calculated gradient accumulation steps: {grad_accum_steps}")


enc = tiktoken.get_encoding("gpt2")
train_dataloader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_dataloader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

# Read https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf
# to see difference between float32, tensorfloat32 and bfloat16
# Although i'm still not sure why speed up is not that much. (maybe memory-bound?)
torch.set_float32_matmul_precision("high")
model = GPT(GPTConfig()) # can change vocab size here to be "nice number"
model.to(device)

use_compile = False # TODO: fix torch.compile interferes with HellaSwag eval and Generation
if use_compile:
    model = torch.compile(model) # huge speed up from this op


if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

#----------------------------------------------------------------------------

# use cosine decay lr similar to gpt-3 paper
# NOTE: follow the paper's suggestion to use 6e-4 as max_lr for 124M model
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
def get_lr(step):
    # 1. linear warmup for warmup_iters steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # 2. if step > max_steps, use min_lr
    if step > max_steps:
        return min_lr
    
    # 3. in between, use cosine decay down to min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

#----------------------------------------------------------------------------

# optimize
# NOTE: hyperparameters are from GPT-3 paper
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device) # TODO: experiment with Muon here

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass

for step in range(max_steps):
    t0 = time.time()

    # validation
    if step % 250 == 0 or step == max_steps - 1:
        model.eval()
        val_dataloader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for micro_step in range(val_loss_steps):
                x, y = val_dataloader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
    

    # evaluate on HellaSwag
    if (step % 250 == 0 or step == max_steps - 1) and not use_compile:
        num_correct_norm = 0
        num_total = 0

        for i, example in enumerate(iterate_examples("val")):
            if i % ddp_world_size != ddp_rank:
                continue
            
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, _ = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            
            num_correct_norm += (pred_norm == label)
            num_total += 1

        if ddp:
            num_correct_norm = torch.tensor(num_correct_norm, device=device, dtype=torch.long)
            num_total = torch.tensor(num_total, device=device, dtype=torch.long)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            num_correct_norm = num_correct_norm.item()
            num_total = num_total.item()
        
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
            
            if step > 0 and (step % 5000 == 0 or step == max_steps - 1):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)


    # once in a while generate from the model (except step 0, which is noise)
    if step > 0 and (step % 250 == 0 or step == max_steps - 1):
        model.eval()
        max_return_sequences = 4
        max_seq_length = 32

        tokens = enc.encode("Hello, I'm a language model")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(max_return_sequences, 1)
        xgen = tokens.to(device)

        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)

        while xgen.size(1) < max_seq_length:
            with torch.no_grad():
                logits, _ = model(xgen)
                # get the logits for the last position
                logits = logits[:, -1, :] # (B, vocab_size)

                # apply softmax to get the probabilities
                probs = F.softmax(logits, dim=-1) # (B, vocab_size)
                # do top-k sampling
                topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
                # select a token form top-k probabilities
                idx_next = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng) # (B, 1)
                # gather the corresponding token indices
                xcol = torch.gather(topk_indices, dim=-1, index=idx_next) # (B, 1)
                # append the sampled token to the running sequence
                xgen = torch.cat((xgen, xcol), dim=1) # (B, T+1)
        
        for i in range(max_return_sequences):
            tokens = xgen[i, :max_seq_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


    # training
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        
        # NOTE: read autocast documentation for more details
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        
        loss = loss / grad_accum_steps # scale the loss by the number of gradient accumulation steps
        
        # only sync gradients on the last micro_step
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss_accum += loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
    # clip the global norm of the gradients to 1.0 (gpt-3 paper)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    torch.cuda.synchronize() # wait for the gpu to finish work
    t1 = time.time()
    dt = (t1 - t0) * 1000 # time difference in milliseconds
    tokens_per_sec = (train_dataloader.B * train_dataloader.T) * grad_accum_steps / (t1 - t0)

    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tokens/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()

#----------------------------------------------------------------------------
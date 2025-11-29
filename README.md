# gpt 2

2 years ago I wrote a deep dive on [From Transformers to ChatGPT](https://hoanhle.github.io/blog/2023/chatgpt/), and now I want to refresh some of my knowledge by reproducing gpt-2 from scratch.

I will slowly introduce algorithmic improvements from [gpt-oss](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the?hide_intro_popup=true), and a few ideas from vision / graphics world (e.g: magnitude preserving and weight constraints from [edm2](https://arxiv.org/abs/2312.02696) paper, which is somewhat similar to [Thinking Machine's blog](https://thinkingmachines.ai/blog/modular-manifolds/)).

## setup

To train this on multi-gpu (8xH100) cloud, I went with [Prime Intellect](https://www.primeintellect.ai/) and [Datacrunch](https://verda.com/). At 
the time when I was writing this, Prime Intellect did not provide spot for more than 1 gpu, and Datacrunch did and was incredible cheap (~7.18e/hour). 

Setup was quite easy by following instructions on both infras. After launching the instance (which takes about 5 minutes) and `ssh` (nit: use `ssh -A` to use agent forwarding which allow remote server to "borrow" local ssh keys), do

```bashrc
# 1. Update system & install basic tools
apt-get update && apt-get install -y git python3-pip python3-venv tmux htop # or btop

# 2. Install uv 
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Note: surprising `btop` does not work well with multi-gpu instance on my current setup and produce this error. I will figure out why later.

```bash
root@quiet-snow-welcomes-ice-01:~# btop
ERROR: Exception in runner thread -> Cpu:: -> graphs, clock, meter : basic_string::at: __n (which is 0) >= this->size() (which is 0)
```

Then clone this repository and install the dependencies:

```python
git clone https://github.com/hoanhle/gpt2

uv sync --extra gpu # if u have nvidia gpu
uv sync --extra cpu

source .venv/bin/activate
``` 

## reproducing gpt 2

first get the [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset

```
python3 fineweb_edu.py
```

then train

```
torchrun --nnodes=1 --nproc-per-node=$NUM_GPU_IN_INSTANCE train_gpt2.py
```

NOTE: torchrun also works with a single gpu setup, but tiny bit slower. Can also do
`python3 train_gpt2.py` on locally.

I used `rsync` to get the logs file locally, might not be best practice though, but works fine for now

```bash
while true; do
    echo "🔄 Syncing data from H100..."
    rsync -avz --update root@<IP_ADDRESS>:~/gpt2/logs/ ./logs/ 
    sleep 360
done
```

## citations

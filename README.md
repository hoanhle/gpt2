# gpt 2

2 years ago I wrote a deep dive on [From Transformers to ChatGPT](https://hoanhle.github.io/blog/2023/chatgpt/), and now I want to refresh some of my knowledge by reproducing gpt-2 from scratch.

## setup

Clone this repository and install the dependencies:

```python
uv sync --extra gpu # if you have nvidia gpu
uv sync --extra cpu
```
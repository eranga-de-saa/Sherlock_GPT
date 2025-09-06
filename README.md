# Sherlock-GPT — Encoder Transformer on the Sherlock Holmes Collection

A compact project that trains an **Encoder-only Transformer** on the *Sherlock Holmes — Complete Collection* dataset from Kaggle.  
This repo demonstrates:
- Data preparation
- A clean encoder-transformer architecture for strong contextual representations

**Dataset:** [Sherlock Holmes Complete Collection (Kaggle)](https://www.kaggle.com/datasets/bengiefigueroa/sherlock-holmes-complete-collection)

**Learning resource inspiration:** Huge thanks to [Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY) for his "Let's build GPT — from scratch" talk and materials that helped shape this project.

---

## Highlights

- **Architecture:** Encoder-only Transformer (stack of self-attention + feed-forward blocks).  

- **Dataset:** Sherlock Holmes text collection (stories & novels). Source: Kaggle.

- **Learning resources:** Based on transformer intuition from Karpathy’s tutorials.

---


## Architecture (Encoder Transformer)

This project uses an encoder-only Transformer characterized by:

- **Input:** tokenized text sequences (length *L*)
- **Embedding layer:** token embeddings + positional encodings (learned or sinusoidal)
- **N encoder blocks:**
  - Multi-head self-attention (no causal mask for pure encoder reps)
  - LayerNorm
  - Position-wise Feed-Forward network
  - Residual connections around attention and FFN
- **Output heads:** depending on task
  - Masked Language Modeling (MLM) head — softmax over vocab
  - Classification head — pooled encoder outputs + linear
  - Retrieval / embedding output — normalized sequence vector

### Minimal block (pseudocode)
```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln1  = nn.LayerNorm(d_model)
        self.ffn  = FeedForward(d_model, d_ff)
        self.ln2  = nn.LayerNorm(d_model)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
```


## Acknowledgements

- **Dataset:** [Sherlock Holmes Complete Collection (Kaggle)](https://www.kaggle.com/datasets/bengiefigueroa/sherlock-holmes-complete-collection)
- **Inspiration:** [Andrej Karpathy’s "Let’s build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY)

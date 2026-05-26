import torch
from model import LayerNormalization

batch_size = 1
seq_len = 2
d_model = 128
x = torch.randn(batch_size, seq_len, d_model)

layer_norm = LayerNormalization(d_model)
x_norm = layer_norm(x)

print(f"X stats before forward")
print(x.mean(-1, keepdim=True))
print(x.std(-1, keepdim=True))

print(f"X stats after forward")
print(x_norm.mean(-1, keepdim=True))
print(x_norm.std(-1, keepdim=True))

print("As you can see, after layer normalization the mean tends to zero and the variance to one across the d_model"
      "dimension.")

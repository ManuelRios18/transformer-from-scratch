import torch
import matplotlib.pyplot as plt
from model import PositionalEncoding

d_model = 400
seq_len = 500
drop_out = 0.0

positional_encoding = PositionalEncoding(d_model=d_model, seq_len=seq_len, drop_out=drop_out)
x = torch.zeros(1, seq_len, d_model)
pos_encoding = positional_encoding(x).squeeze(0).numpy()
plt.title("Positional Encodings")
plt.xlabel("Embeddings dimensions")
plt.ylabel("Position")
plt.imshow(pos_encoding, cmap="RdPu")
plt.show()
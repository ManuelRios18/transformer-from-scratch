import torch
from model import InputEmbedding

vocab_size = 100
embed_size = 512
batch_size = 10
sequence_length = 20
x = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))

input_embedding = InputEmbedding(vocab_size=vocab_size, d_model=embed_size)
embeddings = input_embedding(x)
print(f"The input shape is {x.shape}")
print(f"The embedding shape is {embeddings.shape}")
print(f"What happened? We had {batch_size} sequences of length {sequence_length}.\nThe embedding layer transformed it "
      f"into {batch_size} matrices with {sequence_length} rows (one for each element of the sequence)\n"
      f"and {embed_size} columns which correspond to the embeddings dimension.")

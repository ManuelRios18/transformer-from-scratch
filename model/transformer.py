import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = d_model
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This forward method takes an input tensor of shape (batch_size, seq_len) and gets it corresponding embedding
        for each element in the sequence.

        For instance,

        Imagine an input tensor of shape (2, 10). This is two sequences of 10 elements each.
        Also consider an embedding of size 100.
        Then the output will be a tensor of shape (2, 10, 100).
        This is two matrices of shape (10, 100) being each row a 100 dimensional vector for each element of the
        sequence.

        The method simply uses the ´nn.Embedding´ method and then multiplies it the resulting vector by the embedding
        dimension as proposed in the original paper (Section 3.4)
        But why?
        Basically because as the embedding size grows, its variance decreases. And given that the initialized vectors
        have zero mean, their values will then be close to zero. This is a problem because then we need to add these
        vectors to the positional embeddings vectors. If they magnitudes differ much, the embedding will get ´forgotten´

        :param x: The input tensor containing a batch of index sequences.
        :return: The embeddings associated to each of the input indices.
        """
        return self.embedding(x) * math.sqrt(self.embed_size)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, drop_out: float = 0.1):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be an even number to split between sin and cos."
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=drop_out)

        # We need to unsqueeze to add an extra dimension to broadcasted operation work down the road.
        # Then positions will be a column vector of shape (seq_len, 1)
        positions = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)

        # In the paper the denominator is presented like: (10000)^{\fraq{2*i}{d_model}}
        # But computing element-wise powers is not gpu friendly. Therefore, we use the Log-Space trick A = exp(ln(A))
        # Thus,
        # \begin{aligned}
        #     10000^{\frac{2i}{d_{\text{model}}}} &= e^{\ln\left(10000^{\frac{2i}{d_{\text{model}}}}\right)} \\
        #     &= e^{\frac{2i}{d_{\text{model}}} \ln(10000)} \\
        #     \frac{1}{10000^{\frac{2i}{d_{\text{model}}}}} &= e^{-2i \frac{\ln(10000)}{d_{\text{model}}}}
        # \end{aligned}
        # We are skipping the element-wise power. Now we rely only on logarithms, exponents and multiplications
        # Which are highly optimized operations. And we just have to implement:
        # e^{-2i \frac{\ln(10000)}{d_{\text{model}}}}

        inv_denominator = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        # Initialize the full PE matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Compute the internal matrix product using broadcasting
        # Shape: (seq_len, 1) * (d_model / 2,) -> (seq_len, d_model / 2) This is an outer product!
        angle_rates = positions * inv_denominator

        # Assign to even indices and odd indices
        pe[:, 0::2] = torch.sin(angle_rates)
        pe[:, 1::2] = torch.cos(angle_rates)

        # Unsqueeze to add batch dimension: shape (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as a buffer so the attribute is stored when saving the model.
        self.register_buffer(name='pe', tensor=pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This forward method simple add the positional encoding to the input.
        It also passes the tensor through a dropout layer.

        :param x: Input tensor containing a batch of embedded sequences. Size (batch_size, seq_len, d_model)
        :return: The same tensor plus the positional encoding.
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, d_model: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        # Layer Normalization parameters. One parameter for each dimension.
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This forward method implements Layer Normalization.
        The inputs are of shape (batch_size, seq_len, d_model).
        Computing mean and std using the -1 axis tell Pytorch to compute the mean ACROSS the last dimension.
        In this case, this dimension is d_model.
        In this case, if gamma=1 and beta=0, after normalization, if we compute the mean and the
        std for each word/token, they should be 0 and 1 respectively.
        Basically what we do here is that we ensure that each word/token has zero mean and unit variance, and then we
        scale (gamma) or move (beta) each feature individually.
        :param x: Input tensor of shape (batch_size, seq_len, d_model)
        :return: Output tensor of shape (batch_size, seq_len, d_model)
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False) # unbiased False guarantees dividing by N.
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


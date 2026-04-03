from __future__ import annotations

import math

import torch
from torch import nn


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        hidden_dim = int(d_model * expansion)
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + x


class MultiHeadSelfAttentionModule(nn.Module):
    """Paper-faithful custom MHA shape:
    d_model can differ from num_heads * d_k, matching the paper equations.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int = 64,
        d_v: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.d_k = int(d_k)
        self.d_v = int(d_v)
        self.norm = nn.LayerNorm(d_model)
        self.query_proj = nn.Linear(d_model, self.num_heads * self.d_k, bias=False)
        self.key_proj = nn.Linear(d_model, self.num_heads * self.d_k, bias=False)
        self.value_proj = nn.Linear(d_model, self.num_heads * self.d_v, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        batch_size, seq_len, _ = x.shape

        q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_v)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        return residual + attn_output


class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding="same",
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return residual + x


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        d_k: int = 64,
        d_v: int = 64,
        ffm_expansion: int = 4,
        attn_dropout: float = 0.2,
        ffm_dropout: float = 0.2,
        conv_kernel_size: int = 31,
        conv_dropout: float = 0.2,
        use_final_norm: bool = False,
    ) -> None:
        super().__init__()
        self.ffm = FeedForwardModule(d_model=d_model, expansion=ffm_expansion, dropout=ffm_dropout)
        self.attn = MultiHeadSelfAttentionModule(
            d_model=d_model,
            num_heads=num_heads,
            d_k=d_k,
            d_v=d_v,
            dropout=attn_dropout,
        )
        self.conv = ConvolutionModule(d_model=d_model, kernel_size=conv_kernel_size, dropout=conv_dropout)
        self.final_norm = nn.LayerNorm(d_model) if use_final_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffm(x)
        x = self.attn(x)
        x = self.conv(x)
        return self.final_norm(x)


class FOMLNRegressor(nn.Module):
    """Closer reproduction of the paper's meta-learner.

    Input: (B, W, F=15)
    Conv1d over time with in_channels=15, out_channels=10, kernel_size=10, stride=1
    -> sequence of N Conformer blocks
    -> flatten
    -> fully connected regressor
    """

    def __init__(
        self,
        input_dim: int = 15,
        window_size: int = 30,
        conv_channels: int = 10,
        conv_kernel_size: int = 10,
        conv_stride: int = 1,
        d_model: int | None = 512,
        num_heads: int = 8,
        num_blocks: int = 1,
        d_k: int = 64,
        d_v: int = 64,
        ffm_expansion: int = 4,
        attn_dropout: float = 0.2,
        ffm_dropout: float = 0.2,
        conformer_conv_kernel_size: int = 31,
        conformer_conv_dropout: float = 0.2,
        head_dropout: float = 0.0,
        use_final_norm: bool = False,
    ) -> None:
        super().__init__()
        d_model = int(conv_channels if d_model is None else d_model)
        self.window_size = int(window_size)
        self.conv_kernel_size = int(conv_kernel_size)
        self.conv_stride = int(conv_stride)
        self.frontend = nn.Conv1d(
            input_dim,
            conv_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=0,
        )
        self.frontend_activation = nn.SiLU()
        self.channel_proj = nn.Identity() if conv_channels == d_model else nn.Linear(conv_channels, d_model)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_k=d_k,
                    d_v=d_v,
                    ffm_expansion=ffm_expansion,
                    attn_dropout=attn_dropout,
                    ffm_dropout=ffm_dropout,
                    conv_kernel_size=conformer_conv_kernel_size,
                    conv_dropout=conformer_conv_dropout,
                    use_final_norm=use_final_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        output_length = (self.window_size - self.conv_kernel_size) // self.conv_stride + 1
        if output_length <= 0:
            raise ValueError("Convolution output length must be positive; adjust window_size/kernel_size/stride.")
        self.output_length = int(output_length)
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(self.output_length * d_model, 1)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.frontend(x)
        x = self.frontend_activation(x)
        x = x.transpose(1, 2)
        x = self.channel_proj(x)
        for block in self.blocks:
            x = block(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        sequence_features = self.encode_sequence(x)
        return torch.flatten(sequence_features, start_dim=1)

    def predict_from_features(self, features: torch.Tensor) -> torch.Tensor:
        predictions = self.head(self.head_dropout(features))
        return predictions.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_from_features(self.forward_features(x))

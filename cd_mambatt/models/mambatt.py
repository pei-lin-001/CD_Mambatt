from __future__ import annotations

import math

import torch
from mamba_ssm import Mamba
from torch import nn

from cd_mambatt.models.dd_mamba import DDMambaBlock


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term[: pe[:, 0, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0)]


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)


class ResidualMambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        norm_mode: str,
        inner_dropout: float,
    ) -> None:
        super().__init__()
        if norm_mode not in {"pre", "post"}:
            raise ValueError("norm_mode must be 'pre' or 'post'")
        self.norm_mode = norm_mode
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=False,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(inner_dropout)
        self.dropout2 = nn.Dropout(inner_dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_mode == "pre":
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
            x = x + self.dropout1(attn_out)
            x = x + self.dropout2(self.ffn(self.norm2(x)))
            return x

        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout1(attn_out))
        return self.norm2(x + self.dropout2(self.ffn(x)))


class TorchTransformerStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float,
        norm_mode: str,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=False,
            norm_first=(norm_mode == "pre"),
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MambAttRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 21,
        d_model: int = 21,
        d_state: int = 16,
        d_conv: int = 8,
        expand: int = 2,
        num_mamba_layers: int = 1,
        num_transformer_layers: int = 3,
        num_heads: int = 7,
        dropout: float = 0.5,
        dim_feedforward: int = 2048,
        transformer_impl: str = "custom",
        transformer_norm_mode: str = "pre",
        transformer_inner_dropout: float = 0.0,
        mamba_block_mode: str = "bare",
        spd_gate_init_bias: float = -2.0,
        spd_predictor_mode: str = "shared_head",
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if transformer_impl not in {"custom", "torch"}:
            raise ValueError("transformer_impl must be 'custom' or 'torch'")
        if mamba_block_mode not in {"bare", "prenorm_residual", "dd_spd"}:
            raise ValueError("mamba_block_mode must be 'bare', 'prenorm_residual', or 'dd_spd'")
        if spd_predictor_mode not in {"shared_head", "decomposed_residual", "shared_aux_residual"}:
            raise ValueError(
                "spd_predictor_mode must be 'shared_head', 'decomposed_residual', or 'shared_aux_residual'"
            )

        self.input_proj = nn.Identity() if input_dim == d_model else nn.Linear(input_dim, d_model)
        if mamba_block_mode == "bare":
            self.mamba_blocks = nn.ModuleList(
                [
                    MambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    )
                    for _ in range(num_mamba_layers)
                ]
            )
        elif mamba_block_mode == "prenorm_residual":
            self.mamba_blocks = nn.ModuleList(
                [
                    ResidualMambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    )
                    for _ in range(num_mamba_layers)
                ]
            )
        else:
            self.mamba_blocks = nn.ModuleList(
                [
                    DDMambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        spd_gate_init_bias=spd_gate_init_bias,
                    )
                    for _ in range(num_mamba_layers)
                ]
            )
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        self.transformer_impl = transformer_impl
        self.mamba_block_mode = mamba_block_mode
        self.spd_predictor_mode = spd_predictor_mode
        if transformer_impl == "custom":
            self.transformer_blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=d_model,
                        num_heads=num_heads,
                        dim_feedforward=dim_feedforward,
                        norm_mode=transformer_norm_mode,
                        inner_dropout=transformer_inner_dropout,
                    )
                    for _ in range(num_transformer_layers)
                ]
            )
            self.transformer_encoder = None
        else:
            self.transformer_blocks = None
            self.transformer_encoder = TorchTransformerStack(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                num_layers=num_transformer_layers,
                dropout=dropout,
                norm_mode=transformer_norm_mode,
            )
        self.output_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)
        if self.mamba_block_mode == "dd_spd" and self.spd_predictor_mode in {"decomposed_residual", "shared_aux_residual"}:
            self.inv_head = nn.Linear(d_model, 1)
            self.spec_head = nn.Linear(d_model, 1)
            with torch.no_grad():
                self.inv_head.weight.copy_(self.head.weight)
                self.inv_head.bias.copy_(self.head.bias)
                nn.init.zeros_(self.spec_head.weight)
                nn.init.zeros_(self.spec_head.bias)
        else:
            self.inv_head = None
            self.spec_head = None

    def _decode_sequence(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.transpose(0, 1)
        hidden = self.positional_encoding(hidden)
        if self.transformer_impl == "custom":
            for block in self.transformer_blocks:
                hidden = block(hidden)
        else:
            hidden = self.transformer_encoder(hidden)
        return hidden[-1]

    def _encode_internal(self, x: torch.Tensor, *, return_aux: bool) -> torch.Tensor | dict[str, torch.Tensor]:
        if x.device.type != "cuda":
            raise RuntimeError("MambAttRegressor requires CUDA because the installed Mamba kernels are GPU-only.")

        hidden = self.input_proj(x)
        last_block_aux: dict[str, torch.Tensor] | None = None
        for block in self.mamba_blocks:
            if return_aux and isinstance(block, DDMambaBlock):
                hidden, last_block_aux = block(hidden, return_aux=True)
            else:
                hidden = block(hidden)
        pre_transformer_hidden = hidden
        features = self._decode_sequence(pre_transformer_hidden)
        if not return_aux:
            return features

        domain_sequence = pre_transformer_hidden if last_block_aux is None else last_block_aux["inv_sequence"]
        outputs: dict[str, torch.Tensor] = {
            "features": features,
            "domain_features": domain_sequence.mean(dim=1),
            "pre_transformer_features": pre_transformer_hidden[:, -1, :],
        }
        if last_block_aux is not None:
            invariant_features = self._decode_sequence(last_block_aux["inv_sequence"])
            outputs["invariant_features"] = invariant_features
            outputs["specific_features"] = features - invariant_features
            outputs["gate_sequence"] = last_block_aux["gate_sequence"]
            outputs["gate_mean"] = last_block_aux["gate_mean"]
        return outputs

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode_internal(x, return_aux=False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def forward_features_with_aux(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self._encode_internal(x, return_aux=True)
        if not isinstance(outputs, dict):
            raise TypeError("forward_features_with_aux expected a dictionary of tensors")
        return outputs

    def predict_from_features(self, features: torch.Tensor) -> torch.Tensor:
        prediction = self.head(self.output_dropout(features))
        return prediction.squeeze(-1)

    def predict_from_output_dict(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = outputs["features"]
        prediction_shared = self.predict_from_features(features)
        if self.inv_head is None or self.spec_head is None:
            return {
                "prediction": prediction_shared,
                "prediction_shared": prediction_shared,
                "prediction_inv": prediction_shared,
                "prediction_spec": prediction_shared.new_zeros(prediction_shared.shape),
                "prediction_decomposed": prediction_shared,
            }

        invariant_features = outputs["invariant_features"]
        specific_features = outputs["specific_features"]
        prediction_inv = self.inv_head(self.output_dropout(invariant_features)).squeeze(-1)
        prediction_spec = self.spec_head(self.output_dropout(specific_features)).squeeze(-1)
        prediction_decomposed = prediction_inv + prediction_spec
        if self.spd_predictor_mode == "decomposed_residual":
            prediction = prediction_decomposed
        elif self.spd_predictor_mode == "shared_aux_residual":
            prediction = prediction_shared
        else:
            prediction = prediction_shared
        return {
            "prediction": prediction,
            "prediction_shared": prediction_shared,
            "prediction_inv": prediction_inv,
            "prediction_spec": prediction_spec,
            "prediction_decomposed": prediction_decomposed,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inv_head is not None and self.spec_head is not None:
            outputs = self.forward_features_with_aux(x)
            predictions = self.predict_from_output_dict(outputs)
            return predictions["prediction"]
        features = self.forward_features(x)
        return self.predict_from_features(features)

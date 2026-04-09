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
        spd_scan_mode: str = "mixed",
        spd_gate_mode: str = "token",
        spd_gate_scheme: str = "shared",
        spd_predictor_mode: str = "shared_head",
        domain_conditioned_gate: bool = False,
        frontend_adapter_mode: str = "none",
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if transformer_impl not in {"custom", "torch"}:
            raise ValueError("transformer_impl must be 'custom' or 'torch'")
        if mamba_block_mode not in {"bare", "prenorm_residual", "dd_spd"}:
            raise ValueError("mamba_block_mode must be 'bare', 'prenorm_residual', or 'dd_spd'")
        if spd_scan_mode not in {"mixed", "dual_state"}:
            raise ValueError("spd_scan_mode must be 'mixed' or 'dual_state'")
        if spd_gate_mode not in {"token", "window"}:
            raise ValueError("spd_gate_mode must be 'token' or 'window'")
        if spd_gate_scheme not in {"shared", "dt_bc"}:
            raise ValueError("spd_gate_scheme must be 'shared' or 'dt_bc'")
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
                        spd_scan_mode=spd_scan_mode,
                        spd_gate_mode=spd_gate_mode,
                        spd_gate_scheme=spd_gate_scheme,
                        domain_conditioned_gate=domain_conditioned_gate,
                        frontend_adapter_mode=frontend_adapter_mode,
                    )
                    for _ in range(num_mamba_layers)
                ]
            )
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        self.transformer_impl = transformer_impl
        self.mamba_block_mode = mamba_block_mode
        self.spd_scan_mode = spd_scan_mode
        self.spd_gate_mode = spd_gate_mode
        self.spd_gate_scheme = spd_gate_scheme
        self.spd_predictor_mode = spd_predictor_mode
        self.frontend_adapter_mode = frontend_adapter_mode
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

    def _forward_mamba_sequence(
        self,
        x: torch.Tensor,
        *,
        return_aux: bool,
        domain_label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if x.device.type != "cuda":
            raise RuntimeError("MambAttRegressor requires CUDA because the installed Mamba kernels are GPU-only.")

        hidden = self.input_proj(x)
        last_block_aux: dict[str, torch.Tensor] | None = None
        for block in self.mamba_blocks:
            if return_aux and isinstance(block, DDMambaBlock):
                hidden, last_block_aux = block(hidden, return_aux=True, domain_label=domain_label)
            elif isinstance(block, DDMambaBlock):
                hidden = block(hidden, domain_label=domain_label)
            else:
                hidden = block(hidden)
        return hidden, last_block_aux

    def _decode_sequence(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.transpose(0, 1)
        hidden = self.positional_encoding(hidden)
        if self.transformer_impl == "custom":
            for block in self.transformer_blocks:
                hidden = block(hidden)
        else:
            hidden = self.transformer_encoder(hidden)
        return hidden[-1]

    def _encode_internal(self, x: torch.Tensor, *, return_aux: bool, domain_label: torch.Tensor | None = None) -> torch.Tensor | dict[str, torch.Tensor]:
        hidden, last_block_aux = self._forward_mamba_sequence(x, return_aux=return_aux, domain_label=domain_label)
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
            if "conv_sequence" in last_block_aux:
                outputs["frontend_features"] = last_block_aux["conv_sequence"].mean(dim=1)
                outputs["frontend_last"] = last_block_aux["conv_sequence"][:, -1, :]
            outputs["gate_sequence"] = last_block_aux["gate_sequence"]
            outputs["gate_mean"] = last_block_aux["gate_mean"]
            if "gate_dt_sequence" in last_block_aux:
                outputs["gate_dt_sequence"] = last_block_aux["gate_dt_sequence"]
                outputs["gate_dt_mean"] = last_block_aux["gate_dt_mean"]
            if "gate_bc_sequence" in last_block_aux:
                outputs["gate_bc_sequence"] = last_block_aux["gate_bc_sequence"]
                outputs["gate_bc_mean"] = last_block_aux["gate_bc_mean"]
        return outputs

    def encode(self, x: torch.Tensor, *, domain_label: torch.Tensor | None = None) -> torch.Tensor:
        return self._encode_internal(x, return_aux=False, domain_label=domain_label)

    def forward_mamba_sequence(self, x: torch.Tensor, *, domain_label: torch.Tensor | None = None) -> torch.Tensor:
        hidden, _ = self._forward_mamba_sequence(x, return_aux=False, domain_label=domain_label)
        return hidden

    def forward_features(self, x: torch.Tensor, *, domain_label: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode(x, domain_label=domain_label)

    def forward_features_with_aux(self, x: torch.Tensor, *, domain_label: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        outputs = self._encode_internal(x, return_aux=True, domain_label=domain_label)
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

    def forward(self, x: torch.Tensor, *, domain_label: torch.Tensor | None = None) -> torch.Tensor:
        if self.inv_head is not None and self.spec_head is not None:
            outputs = self.forward_features_with_aux(x, domain_label=domain_label)
            predictions = self.predict_from_output_dict(outputs)
            return predictions["prediction"]
        features = self.forward_features(x, domain_label=domain_label)
        return self.predict_from_features(features)

    @staticmethod
    def extract_encoder_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prefixes = ("input_proj.", "mamba_blocks.")
        return {key: value for key, value in state_dict.items() if key.startswith(prefixes)}


class DualPathMambAttRegressor(nn.Module):
    """Dual-path architecture: Mamba+Attention path (temporal) + Attention-only path (domain-invariant).

    The attention-only path bypasses Mamba entirely, avoiding the hidden-state
    domain drift that accumulates through SSM recurrence.  Both paths share the
    same Transformer decoder weights.  An adaptive gate fuses the two paths,
    learning to shift reliance toward the drift-free attention features under
    domain shift.

    Cross-domain training should apply MMD alignment on the attention-only path
    features (``features_attn``) while leaving the full MambAtt path unconstrained.
    """

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
        fusion_hidden_dim: int | None = None,
        fusion_gate_init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.input_proj = nn.Identity() if input_dim == d_model else nn.Linear(input_dim, d_model)

        # Mamba blocks (Path A only)
        if mamba_block_mode == "bare":
            self.mamba_blocks = nn.ModuleList(
                [MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                 for _ in range(num_mamba_layers)]
            )
        elif mamba_block_mode == "prenorm_residual":
            self.mamba_blocks = nn.ModuleList(
                [ResidualMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                 for _ in range(num_mamba_layers)]
            )
        else:
            raise ValueError(f"DualPathMambAttRegressor supports 'bare' or 'prenorm_residual' mamba_block_mode, got '{mamba_block_mode}'")

        # Shared Transformer decoder + PE
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        if transformer_impl == "custom":
            self.transformer_blocks = nn.ModuleList(
                [TransformerBlock(
                    d_model=d_model, num_heads=num_heads,
                    dim_feedforward=dim_feedforward, norm_mode=transformer_norm_mode,
                    inner_dropout=transformer_inner_dropout,
                ) for _ in range(num_transformer_layers)]
            )
            self.transformer_encoder = None
        else:
            self.transformer_blocks = None
            self.transformer_encoder = TorchTransformerStack(
                d_model=d_model, num_heads=num_heads,
                dim_feedforward=dim_feedforward, num_layers=num_transformer_layers,
                dropout=dropout, norm_mode=transformer_norm_mode,
            )

        # Fusion gate
        gate_hidden = fusion_hidden_dim or d_model
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        with torch.no_grad():
            self.fusion_gate[-1].bias.fill_(fusion_gate_init_bias)

        self.output_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def _decode_sequence(self, hidden: torch.Tensor) -> torch.Tensor:
        """Shared Transformer decoder: (B, T, D) → (D,) last-timestep features."""
        hidden = hidden.transpose(0, 1)
        hidden = self.positional_encoding(hidden)
        if self.transformer_blocks is not None:
            for block in self.transformer_blocks:
                hidden = block(hidden)
        else:
            hidden = self.transformer_encoder(hidden)
        return hidden[-1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns fused features."""
        outputs = self.forward_features_with_aux(x)
        return outputs["features"]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def forward_features_with_aux(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.device.type != "cuda":
            raise RuntimeError("DualPathMambAttRegressor requires CUDA (Mamba kernels are GPU-only).")

        proj = self.input_proj(x)

        # Path A: Full MambAtt (Mamba → Transformer)
        h_mamba = proj
        for block in self.mamba_blocks:
            h_mamba = block(h_mamba)
        features_mamba = self._decode_sequence(h_mamba)

        # Path B: Attention-only (bypass Mamba → Transformer)
        features_attn = self._decode_sequence(proj)

        # Adaptive fusion
        gate = torch.sigmoid(self.fusion_gate(torch.cat([features_mamba, features_attn], dim=-1)))
        features = gate * features_mamba + (1.0 - gate) * features_attn

        return {
            "features": features,
            "features_mamba": features_mamba,
            "features_attn": features_attn,
            "domain_features": features_attn,
            "pre_transformer_features": features,
            "gate_mean": gate.mean(),
            "gate": gate,
        }

    def predict_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.output_dropout(features)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode(x)
        return self.predict_from_features(features)

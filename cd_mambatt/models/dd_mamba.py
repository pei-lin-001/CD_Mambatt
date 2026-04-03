from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from torch import nn


class DDMambaBlock(nn.Module):
    """SPD-style Mamba block with invariant/specific projector branches.

    This is the minimal SPD v0 implementation:

    - keep the standard Mamba `in_proj`, depthwise conv, `A_log`, `D`, `out_proj`
    - split selectivity generation into invariant and specific branches
    - fuse the specific branch through a learnable token-wise gate
    - expose the invariant output sequence for domain-adversarial supervision

    Notes:
    - we intentionally use the explicit non-fused path because we need access to
      the internal selectivity generation tensors.
    - the specific branch is zero-initialized so the block starts close to the
      original Mamba behavior.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        *,
        spd_gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        base = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )

        self.d_model = base.d_model
        self.d_state = base.d_state
        self.d_conv = base.d_conv
        self.expand = base.expand
        self.d_inner = base.d_inner
        self.dt_rank = base.dt_rank

        self.in_proj = base.in_proj
        self.conv1d = base.conv1d
        self.activation = base.activation
        self.act = base.act
        self.A_log = base.A_log
        self.D = base.D
        self.out_proj = base.out_proj

        self.x_proj_inv = base.x_proj
        self.dt_proj_inv = base.dt_proj
        self.x_proj_spec = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj_spec = nn.Linear(self.dt_rank, self.d_inner, bias=False)
        self.gate_proj = nn.Linear(self.d_inner, 1, bias=True)

        # Use near-zero (not strict zero) init for spec branch so the gate
        # receives a gradient signal from the very first step.  With exact zeros
        # the gate gradient is also zero (chain rule: dL/dgate ∝ spec_output = 0),
        # which delays gate learning until spec weights drift away from zero
        # through other loss terms.
        nn.init.normal_(self.x_proj_spec.weight, std=1e-4)
        nn.init.normal_(self.dt_proj_spec.weight, std=1e-4)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, spd_gate_init_bias)

    def _project_selectivity(
        self,
        x: torch.Tensor,
        *,
        batch: int,
        seqlen: int,
    ) -> dict[str, torch.Tensor]:
        x_flat = rearrange(x, "b d l -> (b l) d")

        gate = torch.sigmoid(self.gate_proj(x_flat))

        inv_proj = self.x_proj_inv(x_flat)
        spec_proj = self.x_proj_spec(x_flat)
        dt_inv_raw, B_inv_raw, C_inv_raw = torch.split(
            inv_proj,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt_spec_raw, B_spec_raw, C_spec_raw = torch.split(
            spec_proj,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        B_combined_raw = B_inv_raw + gate * B_spec_raw
        C_combined_raw = C_inv_raw + gate * C_spec_raw

        dt_inv_pre = F.linear(dt_inv_raw, self.dt_proj_inv.weight)
        dt_spec_pre = F.linear(dt_spec_raw, self.dt_proj_spec.weight)
        dt_combined_pre = dt_inv_pre + gate * dt_spec_pre

        return {
            "gate": rearrange(gate, "(b l) 1 -> b l 1", b=batch, l=seqlen),
            "dt_inv": rearrange(dt_inv_pre, "(b l) d -> b d l", b=batch, l=seqlen),
            "dt_combined": rearrange(dt_combined_pre, "(b l) d -> b d l", b=batch, l=seqlen),
            "B_inv": rearrange(B_inv_raw, "(b l) n -> b n l", b=batch, l=seqlen).contiguous(),
            "C_inv": rearrange(C_inv_raw, "(b l) n -> b n l", b=batch, l=seqlen).contiguous(),
            "B_combined": rearrange(B_combined_raw, "(b l) n -> b n l", b=batch, l=seqlen).contiguous(),
            "C_combined": rearrange(C_combined_raw, "(b l) n -> b n l", b=batch, l=seqlen).contiguous(),
        }

    def _run_selective_scan(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        A = -torch.exp(self.A_log.float())
        # delta_bias from the inv branch is shared across both the combined and
        # inv-only scan paths.  It acts as a learned initialization offset for
        # the discretization step and is not domain-specific.
        delta_bias = None if self.dt_proj_inv.bias is None else self.dt_proj_inv.bias.float()
        if x.is_cuda:
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=delta_bias,
                delta_softplus=True,
            )
        else:
            y = selective_scan_ref(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=delta_bias,
                delta_softplus=True,
            )
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, seqlen, _ = hidden_states.shape

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :seqlen])

        selectivity = self._project_selectivity(x, batch=batch, seqlen=seqlen)
        combined_out = self._run_selective_scan(
            x,
            z,
            selectivity["dt_combined"],
            selectivity["B_combined"],
            selectivity["C_combined"],
        )

        if not return_aux:
            return combined_out

        invariant_out = self._run_selective_scan(
            x,
            z,
            selectivity["dt_inv"],
            selectivity["B_inv"],
            selectivity["C_inv"],
        )
        aux = {
            "inv_sequence": invariant_out,
            "gate_sequence": selectivity["gate"],
            "gate_mean": selectivity["gate"].mean(),
        }
        return combined_out, aux

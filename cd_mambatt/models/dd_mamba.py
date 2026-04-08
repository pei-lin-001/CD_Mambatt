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
        spd_scan_mode: str = "mixed",
        spd_gate_mode: str = "token",
        spd_gate_scheme: str = "shared",
    ) -> None:
        super().__init__()
        if spd_scan_mode not in {"mixed", "dual_state"}:
            raise ValueError("spd_scan_mode must be 'mixed' or 'dual_state'")
        if spd_gate_mode not in {"token", "window"}:
            raise ValueError("spd_gate_mode must be 'token' or 'window'")
        if spd_gate_scheme not in {"shared", "dt_bc"}:
            raise ValueError("spd_gate_scheme must be 'shared' or 'dt_bc'")
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
        self.spd_scan_mode = spd_scan_mode
        self.spd_gate_mode = spd_gate_mode
        self.spd_gate_scheme = spd_gate_scheme

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
        if self.spd_gate_scheme == "shared":
            self.gate_proj = nn.Linear(self.d_inner, 1, bias=True)
            self.gate_proj_dt = None
            self.gate_proj_bc = None
        else:
            self.gate_proj = None
            self.gate_proj_dt = nn.Linear(self.d_inner, 1, bias=True)
            self.gate_proj_bc = nn.Linear(self.d_inner, 1, bias=True)

        # Use near-zero (not strict zero) init for spec branch so the gate
        # receives a gradient signal from the very first step.  With exact zeros
        # the gate gradient is also zero (chain rule: dL/dgate ∝ spec_output = 0),
        # which delays gate learning until spec weights drift away from zero
        # through other loss terms.
        nn.init.normal_(self.x_proj_spec.weight, std=1e-4)
        nn.init.normal_(self.dt_proj_spec.weight, std=1e-4)
        if self.gate_proj is not None:
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, spd_gate_init_bias)
        if self.gate_proj_dt is not None:
            nn.init.zeros_(self.gate_proj_dt.weight)
            nn.init.constant_(self.gate_proj_dt.bias, spd_gate_init_bias)
        if self.gate_proj_bc is not None:
            nn.init.zeros_(self.gate_proj_bc.weight)
            nn.init.constant_(self.gate_proj_bc.bias, spd_gate_init_bias)

    def _compute_gate(
        self,
        x: torch.Tensor,
        proj: nn.Linear,
        *,
        batch: int,
        seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_flat = rearrange(x, "b d l -> (b l) d")
        if self.spd_gate_mode == "window":
            x_window = x.mean(dim=2)
            gate = torch.sigmoid(proj(x_window)).unsqueeze(1).expand(-1, seqlen, -1)
            gate_flat = rearrange(gate, "b l 1 -> (b l) 1")
        else:
            gate_flat = torch.sigmoid(proj(x_flat))
            gate = rearrange(gate_flat, "(b l) 1 -> b l 1", b=batch, l=seqlen)
        return gate, gate_flat

    def _project_selectivity(
        self,
        x: torch.Tensor,
        *,
        batch: int,
        seqlen: int,
    ) -> dict[str, torch.Tensor]:
        x_flat = rearrange(x, "b d l -> (b l) d")

        if self.spd_gate_scheme == "shared":
            gate_dt, gate_dt_flat = self._compute_gate(x, self.gate_proj, batch=batch, seqlen=seqlen)
            gate_bc, gate_bc_flat = gate_dt, gate_dt_flat
        else:
            gate_dt, gate_dt_flat = self._compute_gate(x, self.gate_proj_dt, batch=batch, seqlen=seqlen)
            gate_bc, gate_bc_flat = self._compute_gate(x, self.gate_proj_bc, batch=batch, seqlen=seqlen)
        gate = 0.5 * (gate_dt + gate_bc)

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

        B_combined_raw = B_inv_raw + gate_bc_flat * B_spec_raw
        C_combined_raw = C_inv_raw + gate_bc_flat * C_spec_raw

        dt_inv_pre = F.linear(dt_inv_raw, self.dt_proj_inv.weight)
        dt_spec_pre = F.linear(dt_spec_raw, self.dt_proj_spec.weight)
        dt_combined_pre = dt_inv_pre + gate_dt_flat * dt_spec_pre

        return {
            "gate": gate,
            "gate_scan": rearrange(gate, "b l 1 -> b 1 l"),
            "gate_dt": gate_dt,
            "gate_dt_scan": rearrange(gate_dt, "b l 1 -> b 1 l"),
            "gate_bc": gate_bc,
            "gate_bc_scan": rearrange(gate_bc, "b l 1 -> b 1 l"),
            "dt_inv": rearrange(dt_inv_pre, "(b l) d -> b d l", b=batch, l=seqlen),
            "dt_spec": rearrange(dt_spec_pre, "(b l) d -> b d l", b=batch, l=seqlen),
            "dt_combined": rearrange(dt_combined_pre, "(b l) d -> b d l", b=batch, l=seqlen),
            "B_inv": rearrange(B_inv_raw, "(b l) n -> b n l", b=batch, l=seqlen).contiguous(),
            "C_inv": rearrange(C_inv_raw, "(b l) n -> b n l", b=batch, l=seqlen).contiguous(),
            "B_spec": rearrange(B_spec_raw, "(b l) n -> b n l", b=batch, l=seqlen).contiguous(),
            "C_spec": rearrange(C_spec_raw, "(b l) n -> b n l", b=batch, l=seqlen).contiguous(),
            "B_combined": rearrange(B_combined_raw, "(b l) n -> b n l", b=batch, l=seqlen).contiguous(),
            "C_combined": rearrange(C_combined_raw, "(b l) n -> b n l", b=batch, l=seqlen).contiguous(),
        }

    def _run_selective_scan_core(
        self,
        x: torch.Tensor,
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
                None,
                z=None,
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
                None,
                z=None,
                delta_bias=delta_bias,
                delta_softplus=True,
            )
        return y

    def _finalize_scan_output(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        dtype_in = x.dtype
        if self.D is not None:
            y = y + x * rearrange(self.D.float(), "d -> d 1")
        y = y * F.silu(z)
        y = y.to(dtype=dtype_in)
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)

    def _run_selective_scan(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        y = self._run_selective_scan_core(x, dt, B, C)
        return self._finalize_scan_output(y, x, z)

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
        if self.spd_scan_mode == "dual_state":
            invariant_core = self._run_selective_scan_core(
                x,
                selectivity["dt_inv"],
                selectivity["B_inv"],
                selectivity["C_inv"],
            )
            if self.spd_gate_scheme == "dt_bc":
                specific_core = self._run_selective_scan_core(
                    x,
                    selectivity["gate_dt_scan"] * selectivity["dt_spec"],
                    selectivity["gate_bc_scan"] * selectivity["B_spec"],
                    selectivity["gate_bc_scan"] * selectivity["C_spec"],
                )
                combined_core = invariant_core + specific_core
            else:
                specific_core = self._run_selective_scan_core(
                    x,
                    selectivity["dt_spec"],
                    selectivity["B_spec"],
                    selectivity["C_spec"],
                )
                combined_core = invariant_core + selectivity["gate_scan"] * specific_core
            combined_out = self._finalize_scan_output(combined_core, x, z)
            invariant_out = self._finalize_scan_output(invariant_core, x, z)
            specific_out = self._finalize_scan_output(specific_core, x, z)
        else:
            combined_out = self._run_selective_scan(
                x,
                z,
                selectivity["dt_combined"],
                selectivity["B_combined"],
                selectivity["C_combined"],
            )
            invariant_out = self._run_selective_scan(
                x,
                z,
                selectivity["dt_inv"],
                selectivity["B_inv"],
                selectivity["C_inv"],
            )
            specific_out = None

        if not return_aux:
            return combined_out

        aux = {
            "inv_sequence": invariant_out,
            "gate_sequence": selectivity["gate"],
            "gate_mean": selectivity["gate"].mean(),
            "gate_dt_sequence": selectivity["gate_dt"],
            "gate_dt_mean": selectivity["gate_dt"].mean(),
            "gate_bc_sequence": selectivity["gate_bc"],
            "gate_bc_mean": selectivity["gate_bc"].mean(),
            "conv_sequence": rearrange(x, "b d l -> b l d"),
        }
        if specific_out is not None:
            aux["spec_sequence"] = specific_out
        return combined_out, aux

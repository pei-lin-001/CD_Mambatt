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
        domain_conditioned_gate: bool = False,
        frontend_adapter_mode: str = "none",
    ) -> None:
        super().__init__()
        if spd_scan_mode not in {"mixed", "dual_state"}:
            raise ValueError("spd_scan_mode must be 'mixed' or 'dual_state'")
        if spd_gate_mode not in {"token", "window"}:
            raise ValueError("spd_gate_mode must be 'token' or 'window'")
        if spd_gate_scheme not in {"shared", "dt_bc"}:
            raise ValueError("spd_gate_scheme must be 'shared' or 'dt_bc'")
        if frontend_adapter_mode not in {"none", "target_affine", "target_residual"}:
            raise ValueError("frontend_adapter_mode must be 'none', 'target_affine', or 'target_residual'")
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
        self.domain_conditioned_gate = domain_conditioned_gate
        self.frontend_adapter_mode = frontend_adapter_mode

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

        if self.domain_conditioned_gate:
            self.domain_gate_shift = nn.Parameter(torch.zeros(1))
            if self.spd_gate_scheme == "dt_bc":
                self.domain_gate_shift_dt = nn.Parameter(torch.zeros(1))
                self.domain_gate_shift_bc = nn.Parameter(torch.zeros(1))
            else:
                self.domain_gate_shift_dt = None
                self.domain_gate_shift_bc = None
        else:
            self.domain_gate_shift = None
            self.domain_gate_shift_dt = None
            self.domain_gate_shift_bc = None

        if self.frontend_adapter_mode == "target_affine":
            self.frontend_target_scale = nn.Parameter(torch.zeros(self.d_inner))
            self.frontend_target_bias = nn.Parameter(torch.zeros(self.d_inner))
            self.frontend_target_adapter = None
        elif self.frontend_adapter_mode == "target_residual":
            self.frontend_target_scale = None
            self.frontend_target_bias = None
            self.frontend_target_adapter = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=1, bias=True)
            nn.init.zeros_(self.frontend_target_adapter.weight)
            nn.init.zeros_(self.frontend_target_adapter.bias)
        else:
            self.frontend_target_scale = None
            self.frontend_target_bias = None
            self.frontend_target_adapter = None

    def _compute_gate(
        self,
        x: torch.Tensor,
        proj: nn.Linear,
        *,
        batch: int,
        seqlen: int,
        domain_shift: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_flat = rearrange(x, "b d l -> (b l) d")
        if self.spd_gate_mode == "window":
            x_window = x.mean(dim=2)
            pre_gate = proj(x_window)
            if domain_shift is not None:
                pre_gate = pre_gate + domain_shift
            gate = torch.sigmoid(pre_gate).unsqueeze(1).expand(-1, seqlen, -1)
            gate_flat = rearrange(gate, "b l 1 -> (b l) 1")
        else:
            pre_gate = proj(x_flat)
            if domain_shift is not None:
                if domain_shift.shape[0] == 1:
                    shift_expanded = domain_shift.expand(batch * seqlen, -1)
                else:
                    shift_expanded = domain_shift.repeat_interleave(seqlen, dim=0)
                pre_gate = pre_gate + shift_expanded
            gate_flat = torch.sigmoid(pre_gate)
            gate = rearrange(gate_flat, "(b l) 1 -> b l 1", b=batch, l=seqlen)
        return gate, gate_flat

    def _project_selectivity(
        self,
        x_inv: torch.Tensor,
        x_spec: torch.Tensor,
        *,
        batch: int,
        seqlen: int,
        domain_label: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x_gate = x_spec
        x_inv_flat = rearrange(x_inv, "b d l -> (b l) d")
        x_spec_flat = rearrange(x_spec, "b d l -> (b l) d")

        shift_dt: torch.Tensor | None = None
        shift_bc: torch.Tensor | None = None
        if self.domain_conditioned_gate and domain_label is not None:
            unique_labels = domain_label.unique()
            if unique_labels.numel() == 1:
                flag = unique_labels[0].float()
                if self.spd_gate_scheme == "dt_bc" and self.domain_gate_shift_dt is not None:
                    shift_dt = (self.domain_gate_shift_dt * flag).unsqueeze(0)
                    shift_bc = (self.domain_gate_shift_bc * flag).unsqueeze(0)
                else:
                    shift_dt = (self.domain_gate_shift * flag).unsqueeze(0)
                    shift_bc = shift_dt
            else:
                if self.spd_gate_scheme == "dt_bc" and self.domain_gate_shift_dt is not None:
                    shift_dt = (self.domain_gate_shift_dt * domain_label.float()).unsqueeze(1)
                    shift_bc = (self.domain_gate_shift_bc * domain_label.float()).unsqueeze(1)
                else:
                    per_sample = self.domain_gate_shift * domain_label.float()
                    shift_dt = per_sample.unsqueeze(1)
                    shift_bc = shift_dt

        if self.spd_gate_scheme == "shared":
            gate_dt, gate_dt_flat = self._compute_gate(x_gate, self.gate_proj, batch=batch, seqlen=seqlen, domain_shift=shift_dt)
            gate_bc, gate_bc_flat = gate_dt, gate_dt_flat
        else:
            gate_dt, gate_dt_flat = self._compute_gate(x_gate, self.gate_proj_dt, batch=batch, seqlen=seqlen, domain_shift=shift_dt)
            gate_bc, gate_bc_flat = self._compute_gate(x_gate, self.gate_proj_bc, batch=batch, seqlen=seqlen, domain_shift=shift_bc)
        gate = 0.5 * (gate_dt + gate_bc)

        inv_proj = self.x_proj_inv(x_inv_flat)
        spec_proj = self.x_proj_spec(x_spec_flat)
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

    def _apply_frontend_adapter(
        self,
        x: torch.Tensor,
        domain_label: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.frontend_adapter_mode == "none" or domain_label is None:
            return x
        domain_mask = domain_label.float().view(-1, 1, 1)
        if self.frontend_adapter_mode == "target_affine":
            scale = self.frontend_target_scale.view(1, -1, 1)
            bias = self.frontend_target_bias.view(1, -1, 1)
            return x * (1.0 + domain_mask * scale) + domain_mask * bias
        if self.frontend_target_adapter is None:
            return x
        return x + domain_mask * self.frontend_target_adapter(x)

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
        domain_label: torch.Tensor | None = None,
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
        x_shared = self.act(self.conv1d(x)[..., :seqlen])
        x_combined = self._apply_frontend_adapter(x_shared, domain_label)

        selectivity = self._project_selectivity(
            x_shared,
            x_combined,
            batch=batch,
            seqlen=seqlen,
            domain_label=domain_label,
        )
        if self.spd_scan_mode == "dual_state":
            invariant_core = self._run_selective_scan_core(
                x_shared,
                selectivity["dt_inv"],
                selectivity["B_inv"],
                selectivity["C_inv"],
            )
            if self.spd_gate_scheme == "dt_bc":
                specific_core = self._run_selective_scan_core(
                    x_combined,
                    selectivity["gate_dt_scan"] * selectivity["dt_spec"],
                    selectivity["gate_bc_scan"] * selectivity["B_spec"],
                    selectivity["gate_bc_scan"] * selectivity["C_spec"],
                )
                combined_core = invariant_core + specific_core
            else:
                specific_core = self._run_selective_scan_core(
                    x_combined,
                    selectivity["dt_spec"],
                    selectivity["B_spec"],
                    selectivity["C_spec"],
                )
                combined_core = invariant_core + selectivity["gate_scan"] * specific_core
            combined_out = self._finalize_scan_output(combined_core, x_combined, z)
            invariant_out = self._finalize_scan_output(invariant_core, x_shared, z)
            specific_out = self._finalize_scan_output(specific_core, x_combined, z)
        else:
            combined_out = self._run_selective_scan(
                x_combined,
                z,
                selectivity["dt_combined"],
                selectivity["B_combined"],
                selectivity["C_combined"],
            )
            invariant_out = self._run_selective_scan(
                x_shared,
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
            "conv_sequence": rearrange(x_shared, "b d l -> b l d"),
            "combined_conv_sequence": rearrange(x_combined, "b d l -> b l d"),
        }
        if specific_out is not None:
            aux["spec_sequence"] = specific_out
        return combined_out, aux

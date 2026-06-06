# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn as _selective_scan_cuda,
        mamba_inner_fn,
        bimamba_inner_fn,
        mamba_inner_fn_no_out_proj,
    )
except ImportError:
    _selective_scan_cuda, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as _mamba_chunk_scan_cuda
except ImportError:
    _mamba_chunk_scan_cuda = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def _expand_group_param(param: Tensor, target_channels: int) -> Tensor:
    """Expand grouped B/C scan parameters to per-channel parameters."""
    if param.dim() == 3:
        return param.unsqueeze(1).expand(-1, target_channels, -1, -1)
    if param.dim() != 4:
        raise ValueError(f"Expected scan parameter with 3 or 4 dims, got {param.shape}.")
    groups = param.shape[1]
    if groups == target_channels:
        return param
    if target_channels % groups != 0:
        raise ValueError(
            f"Cannot expand grouped scan parameter from {groups} groups "
            f"to {target_channels} channels."
        )
    return param.repeat_interleave(target_channels // groups, dim=1)


def _selective_scan_ref(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor] = None,
    z: Optional[Tensor] = None,
    delta_bias: Optional[Tensor] = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    """Dependency-free selective scan reference path.

    This is slower than the CUDA/Triton implementation but keeps the repository
    runnable when `mamba_ssm` is not installed.
    """
    batch, channels, seqlen = u.shape
    state_dim = A.shape[-1]
    if B is None or C is None:
        raise ValueError("The reference selective scan requires explicit B and C tensors.")

    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1).to(dtype=delta.dtype, device=delta.device)
    if delta_softplus:
        delta = F.softplus(delta)

    B = _expand_group_param(B, channels).to(dtype=u.dtype, device=u.device)
    C = _expand_group_param(C, channels).to(dtype=u.dtype, device=u.device)
    A = A.to(dtype=torch.float32, device=u.device)
    D = D.to(dtype=u.dtype, device=u.device) if D is not None else None

    state = torch.zeros(batch, channels, state_dim, dtype=torch.float32, device=u.device)
    outputs = []
    u_float = u.float()
    delta_float = delta.float()
    B_float = B.float()
    C_float = C.float()

    for step in range(seqlen):
        delta_t = delta_float[:, :, step]
        u_t = u_float[:, :, step]
        dA = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
        dB = delta_t.unsqueeze(-1) * B_float[:, :, :, step]
        state = state * dA + u_t.unsqueeze(-1) * dB
        y = torch.sum(state * C_float[:, :, :, step], dim=-1).to(dtype=u.dtype)
        if D is not None:
            y = y + u[:, :, step] * D.view(1, -1)
        if z is not None:
            y = y * F.silu(z[:, :, step])
        outputs.append(y)

    out = torch.stack(outputs, dim=-1)
    if return_last_state:
        return out, state
    return out


def selective_scan_fn(*args, **kwargs):
    if _selective_scan_cuda is not None:
        return _selective_scan_cuda(*args, **kwargs)
    return _selective_scan_ref(*args, **kwargs)


def _mamba_chunk_scan_combined_ref(
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    chunk_size: int = 256,
    D: Optional[Tensor] = None,
    z: Optional[Tensor] = None,
    seq_idx: Optional[Tensor] = None,
    initial_states: Optional[Tensor] = None,
    **kwargs,
):
    """Small PyTorch fallback for the Mamba2-style chunk scan used in this repo."""
    batch, seqlen, heads, head_dim = x.shape
    state_dim = B.shape[-1]
    groups = B.shape[2]
    if heads % groups != 0:
        raise ValueError(f"Cannot expand {groups} B/C groups to {heads} heads.")
    B = B.repeat_interleave(heads // groups, dim=2)
    C = C.repeat_interleave(heads // groups, dim=2)

    if initial_states is None:
        state = torch.zeros(batch, heads, head_dim, state_dim, dtype=torch.float32, device=x.device)
    else:
        state = initial_states.float().clone()

    A = A.float().to(x.device)
    if A.dim() == 1:
        A = A.view(1, heads, 1, 1)
    else:
        A = A.view(1, heads, 1, state_dim)
    D = D.to(dtype=x.dtype, device=x.device) if D is not None else None

    outputs = []
    x_float = x.float()
    dt_float = dt.float()
    B_float = B.float()
    C_float = C.float()
    for step in range(seqlen):
        dt_t = dt_float[:, step, :]
        dA = torch.exp(dt_t[:, :, None, None] * A)
        dB = dt_t[:, :, None, None] * B_float[:, step, :, None, :]
        state = state * dA + x_float[:, step, :, :, None] * dB
        y = torch.sum(state * C_float[:, step, :, None, :], dim=-1).to(dtype=x.dtype)
        if D is not None:
            y = y + D.view(1, heads, 1) * x[:, step, :, :]
        if z is not None:
            y = y * F.silu(z[:, step, :, :])
        outputs.append(y)
    return torch.stack(outputs, dim=1)


def mamba_chunk_scan_combined(*args, **kwargs):
    if _mamba_chunk_scan_cuda is not None:
        return _mamba_chunk_scan_cuda(*args, **kwargs)
    return _mamba_chunk_scan_combined_ref(*args, **kwargs)


class NCSPIPromptGenerator(nn.Module):
    """Non-Causal Semantic Prompt Injection router.

    It builds a decoupled prompt pool U = W_exp @ W_shr, routes a global
    sequence descriptor to a sparse prompt selection, and returns a point-wise
    prompt that is injected into the Mamba readout matrix C.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        num_prompts: int = 16,
        rank: int = 48,
        temperature: float = 1.0,
        hard: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.hard = hard
        self.exclusive = nn.Parameter(torch.empty(num_prompts, rank))
        self.shared = nn.Parameter(torch.empty(rank, d_state))
        hidden = max(d_model // 2, rank)
        self.router = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_prompts),
        )
        nn.init.normal_(self.exclusive, std=0.02)
        nn.init.orthogonal_(self.shared)

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch, seqlen, _ = hidden_states.shape
        pooled = hidden_states.mean(dim=1)
        logits = self.router(pooled)
        tau = max(float(self.temperature), 1e-6)
        if self.training:
            weights = F.gumbel_softmax(logits, tau=tau, hard=self.hard, dim=-1)
        else:
            weights = torch.softmax(logits / tau, dim=-1)
        prompt_pool = self.exclusive @ self.shared
        prompt = weights @ prompt_pool
        return prompt.unsqueeze(1).expand(batch, seqlen, -1)


class CustomMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        use_nc_spi=False,
        nc_spi_num_prompts=16,
        nc_spi_rank=48,
        nc_spi_temperature=1.0,
        nc_spi_hard=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank in ["auto", None] else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.nc_spi = NCSPIPromptGenerator(
            d_model=d_model,
            d_state=d_state,
            num_prompts=nc_spi_num_prompts,
            rank=nc_spi_rank,
            temperature=nc_spi_temperature,
            hard=nc_spi_hard,
        ) if use_nc_spi else None

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize dt projection
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        # Bidirectional Mamba parameters.
        if bimamba_type == "v2":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
            self.D_b._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def _run_scan_branch(
        self,
        xz: Tensor,
        seqlen: int,
        conv1d: nn.Conv1d,
        x_proj: nn.Linear,
        dt_proj: nn.Linear,
        A: Tensor,
        D: Tensor,
        prompt: Optional[Tensor] = None,
        conv_state: Optional[Tensor] = None,
        ssm_state: Optional[Tensor] = None,
    ):
        x, z = xz.chunk(2, dim=1)
        if conv_state is not None:
            conv_state.copy_(x[:, :, -self.d_conv:])
        if causal_conv1d_fn is None:
            x = self.act(conv1d(x)[..., :seqlen])
        else:
            x = causal_conv1d_fn(
                x,
                rearrange(conv1d.weight, "d 1 w -> d w"),
                conv1d.bias,
                self.activation,
            )

        x_dbl = x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        if prompt is not None:
            C = C + prompt.permute(0, 2, 1).to(dtype=C.dtype, device=C.device)

        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            D.float(),
            z=z,
            delta_bias=dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        return y

    def forward(self, hidden_states, inference_params=None, prompt=None):
        """
        hidden_states: (B, L, D)
        prompt: (B, L, d_state) or None, prompt matrix P
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        if prompt is None and self.nc_spi is not None:
            prompt = self.nc_spi(hidden_states)

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # Input projection.
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        can_use_fast_path = (
            self.use_fast_path
            and inference_params is None
            and prompt is None
            and (
                (self.bimamba_type == "v2" and mamba_inner_fn_no_out_proj is not None)
                or (self.bimamba_type != "v2" and mamba_inner_fn is not None)
            )
        )
        if can_use_fast_path:
            if self.bimamba_type == "v2":
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,
                    None,
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                out = F.linear(
                    rearrange(out + out_b.flip([-1]), "b d l -> b l d"),
                    self.out_proj.weight,
                    self.out_proj.bias
                )
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,
                    None,
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            y = self._run_scan_branch(
                xz=xz,
                seqlen=seqlen,
                conv1d=self.conv1d,
                x_proj=self.x_proj,
                dt_proj=self.dt_proj,
                A=A,
                D=self.D,
                prompt=prompt,
                conv_state=conv_state,
                ssm_state=ssm_state,
            )
            if self.bimamba_type == "v2":
                A_b = -torch.exp(self.A_b_log.float())
                prompt_b = prompt.flip(1) if prompt is not None else None
                y_b = self._run_scan_branch(
                    xz=xz.flip(-1),
                    seqlen=seqlen,
                    conv1d=self.conv1d_b,
                    x_proj=self.x_proj_b,
                    dt_proj=self.dt_proj_b,
                    A=A_b,
                    D=self.D_b,
                    prompt=prompt_b,
                ).flip(-1)
                y = y + y_b
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)

        # Convolution step.
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        A = -torch.exp(self.A_log.float())

        # SSM step. Prompt injection is skipped here because this path is used for inference.
        if selective_state_update is None:
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class CustomMamba2(CustomMamba):
    """Compatibility wrapper for call sites that previously used Mamba2.

    The repository's MedMamba blocks only rely on the common sequence module
    interface. This wrapper accepts the Mamba2-style keyword arguments and maps
    them to the local CustomMamba implementation so imports no longer need the
    external `mamba_ssm` package.
    """

    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=None,
        chunk_size=256,
        use_mem_eff_path=False,
        A_init_range=(1, 16),
        **kwargs,
    ):
        kwargs.pop("head_channel", None)
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **kwargs,
        )


Mamba = CustomMamba
Mamba2 = CustomMamba2

__all__ = [
    "CustomMamba",
    "CustomMamba2",
    "Mamba",
    "Mamba2",
    "NCSPIPromptGenerator",
    "selective_scan_fn",
    "mamba_chunk_scan_combined",
]

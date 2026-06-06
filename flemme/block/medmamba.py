from .common import *
from flemme.CustomMamba import (
    Mamba,
    Mamba2,
    NCSPIPromptGenerator,
    selective_scan_fn,
)
from .pcd_utils import gather_features
from einops import repeat


def _normalize_scores(scores):
    mean = scores.mean(dim=-1, keepdim=True)
    std = scores.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (scores - mean) / std


def _gather_points(x, index):
    if index.dim() == 2:
        return torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    return torch.gather(
        x.unsqueeze(1).expand(-1, index.shape[1], -1, -1),
        2,
        index.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]),
    )


def xsort(xyz):
    return torch.argsort(xyz[:, :, 0], dim=-1)


def ysort(xyz):
    return torch.argsort(xyz[:, :, 1], dim=-1)


def zsort(xyz):
    return torch.argsort(xyz[:, :, 2], dim=-1)


def centersort(xyz):
    center = xyz.mean(dim=1, keepdim=True)
    return torch.argsort(torch.norm(xyz - center, dim=-1), dim=-1)


def nonsort(xyz):
    return torch.arange(xyz.shape[1], device=xyz.device).unsqueeze(0).expand(xyz.shape[0], -1)


def _morton_codes(xyz, bits=10):
    xyz_min = xyz.amin(dim=1, keepdim=True)
    xyz_max = xyz.amax(dim=1, keepdim=True)
    xyz_norm = (xyz - xyz_min) / (xyz_max - xyz_min).clamp_min(1e-6)
    quantized = torch.clamp((xyz_norm * ((1 << bits) - 1)).long(), 0, (1 << bits) - 1)
    x, y, z = quantized.unbind(dim=-1)
    code = torch.zeros_like(x)
    for bit in range(bits):
        code = code | (((x >> bit) & 1) << (3 * bit + 2))
        code = code | (((y >> bit) & 1) << (3 * bit + 1))
        code = code | (((z >> bit) & 1) << (3 * bit))
    return code


def z_order_sort(xyz):
    return torch.argsort(_morton_codes(xyz), dim=-1)


def resort(sorted_index):
    return torch.argsort(sorted_index, dim=-1)


_DEFORMABLE3D_CACHE = {
    "knn_idx": None,
    "knn_idx_multiscale": None,
    "feat": None,
    "xyz": None,
}


def set_deformable3d_cache(knn_idx=None, knn_idx_multiscale=None, feat=None, xyz=None, radius_idx=None):
    if knn_idx is not None and knn_idx_multiscale is None:
        knn_idx_multiscale = [knn_idx]
    _DEFORMABLE3D_CACHE["knn_idx"] = knn_idx_multiscale[0] if knn_idx_multiscale else knn_idx
    _DEFORMABLE3D_CACHE["knn_idx_multiscale"] = knn_idx_multiscale
    _DEFORMABLE3D_CACHE["feat"] = feat
    _DEFORMABLE3D_CACHE["xyz"] = xyz


class Deformable3DPathTrans(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, score):
        _, indices = torch.topk(score, k=score.shape[-1], dim=-1, largest=False)
        x_gathered = torch.gather(x, 2, indices.unsqueeze(1).expand(-1, x.shape[1], -1)).contiguous()
        ctx.save_for_backward(x, score, indices)
        return x_gathered.permute(0, 2, 1).contiguous(), indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        x, score, indices = ctx.saved_tensors
        grad_x = torch.zeros_like(x)
        grad_x.scatter_add_(2, indices.unsqueeze(1).expand(-1, x.shape[1], -1), grad_output.permute(0, 2, 1))
        grad_score = (grad_output.permute(0, 2, 1) - grad_x).mean(dim=1).view_as(score)
        return grad_x, grad_score


class StaticScanner(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, xyz):
        if self.mode == "x_order":
            return xsort(xyz)
        if self.mode == "y_order":
            return ysort(xyz)
        if self.mode == "z_order":
            return z_order_sort(xyz)
        if self.mode == "z_axis":
            return zsort(xyz)
        if self.mode == "morton" or self.mode == "z_curve":
            return z_order_sort(xyz)
        if self.mode == "center_dist":
            return centersort(xyz)
        if self.mode == "nonsort":
            return nonsort(xyz)
        raise ValueError(f"Unsupported scan strategy: {self.mode}")


class MultiScaleDeformableScanner(nn.Module):
    def __init__(self, hidden_dim=64, alpha=0.2, num_scales=3, use_feat=True):
        super().__init__()
        self.alpha = alpha
        self.num_scales = num_scales
        self.use_feat = use_feat
        self.proj_vec = nn.Parameter(torch.randn(3))
        in_dim = 2 + 2 * num_scales
        self.offset_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _base_position(self, xyz):
        order = z_order_sort(xyz)
        base_pos = torch.zeros_like(xyz[:, :, 0])
        pos = torch.linspace(0.0, 1.0, xyz.shape[1], device=xyz.device, dtype=xyz.dtype)
        base_pos.scatter_(1, order, pos.unsqueeze(0).expand(xyz.shape[0], -1))
        return base_pos

    def _scale_stats(self, xyz, feat, knn_idx):
        neigh_xyz = _gather_points(xyz, knn_idx)
        mu = torch.norm(neigh_xyz - xyz.unsqueeze(2), dim=-1).mean(dim=-1)
        mu = _normalize_scores(mu)
        if self.use_feat and feat is not None and feat.shape[:2] == xyz.shape[:2]:
            feat_norm = F.normalize(feat, dim=-1)
            neigh_feat = _gather_points(feat_norm, knn_idx)
            eta = (feat_norm.unsqueeze(2) * neigh_feat).sum(dim=-1).mean(dim=-1)
            eta = _normalize_scores(eta)
        else:
            eta = torch.zeros_like(mu)
        return mu, eta

    def forward(self, xyz):
        batch, length, _ = xyz.shape
        base_pos = self._base_position(xyz)
        direction = self.proj_vec / self.proj_vec.norm().clamp_min(1e-6)
        centered = xyz - xyz.mean(dim=1, keepdim=True)
        global_response = _normalize_scores(torch.einsum("blc,c->bl", centered, direction))

        knn_list = _DEFORMABLE3D_CACHE.get("knn_idx_multiscale")
        feat = _DEFORMABLE3D_CACHE.get("feat")
        stats = []
        if not knn_list:
            knn_list = []
        for scale_id in range(self.num_scales):
            if scale_id < len(knn_list) and knn_list[scale_id] is not None:
                mu, eta = self._scale_stats(xyz, feat, knn_list[scale_id].long())
            else:
                mu = torch.zeros(batch, length, device=xyz.device, dtype=xyz.dtype)
                eta = torch.zeros_like(mu)
            stats.extend([mu, eta])

        score_input = torch.stack([base_pos, global_response] + stats, dim=-1)
        delta = self.offset_mlp(score_input).squeeze(-1)
        score = base_pos + self.alpha * torch.tanh(delta)
        return torch.argsort(score, dim=-1), score


def get_scanners(scanners):
    if scanners is None:
        return nn.ModuleList()
    if isinstance(scanners, (str, dict)):
        scanners = [scanners]
    modules = []
    for scanner in scanners:
        if isinstance(scanner, dict):
            config = scanner.copy()
            name = config.pop("name")
        else:
            config = {}
            name = scanner
        if name in ["mds", "gass_mds", "deformable3d_learnable", "deformable3d_multiscale"]:
            modules.append(MultiScaleDeformableScanner(**config))
        else:
            modules.append(StaticScanner(name))
    return nn.ModuleList(modules)


def _scan_features(features, sorted_index_list):
    sorted_features = []
    length = features.shape[-1]
    for item in sorted_index_list:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            indices, score = item
            indices = indices.clamp(0, length - 1)
            if score is not None:
                sorted_feature, _ = Deformable3DPathTrans.apply(features, score)
                sorted_feature = sorted_feature.permute(0, 2, 1).contiguous()
            else:
                sorted_feature = gather_features(features, index=indices, channel_dim=1, gather_dim=-1)
        else:
            indices = item.clamp(0, length - 1)
            sorted_feature = gather_features(features, index=indices, channel_dim=1, gather_dim=-1)
        sorted_features.append(sorted_feature)
    return torch.stack(sorted_features, dim=1)


def _scan_project(features, sorted_index_list):
    recovered = []
    for scan_id, item in enumerate(sorted_index_list):
        indices = item[0] if isinstance(item, (tuple, list)) and len(item) == 2 else item
        recovered.append(
            gather_features(features[:, scan_id], index=resort(indices), channel_dim=1, gather_dim=-1)
        )
    return sum(recovered)


class PathConditionedRPE(nn.Module):
    def __init__(self, d_model, max_distance=2.0, num_freqs=10):
        super().__init__()
        self.max_distance = max_distance
        self.num_freqs = num_freqs
        self.register_buffer("freq_bands", 2.0 ** torch.arange(num_freqs).float())
        in_dim = (1 + 3 + 1) * 2 * num_freqs
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def _fourier(self, x):
        scaled = x.unsqueeze(-1) * self.freq_bands.view(*([1] * x.dim()), -1) * torch.pi
        return torch.stack([torch.sin(scaled), torch.cos(scaled)], dim=-1).flatten(-3)

    def forward(self, xyz_sorted):
        rel = torch.zeros_like(xyz_sorted)
        if xyz_sorted.shape[1] > 1:
            rel[:, 1:] = xyz_sorted[:, 1:] - xyz_sorted[:, :-1]
            rel[:, 0] = rel[:, 1]
        length = torch.norm(rel, dim=-1, keepdim=True)
        direction = rel / length.clamp_min(1e-6)
        curvature = torch.zeros_like(length)
        if xyz_sorted.shape[1] > 2:
            curvature[:, 1:] = torch.norm(direction[:, 1:] - direction[:, :-1], dim=-1, keepdim=True)
            curvature[:, 0] = curvature[:, 1]
        encoded = torch.cat(
            [
                self._fourier(length / (self.max_distance + 1e-6)),
                self._fourier(direction),
                self._fourier(curvature),
            ],
            dim=-1,
        )
        return self.proj(encoded)


class MedMambaBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel=None,
        time_channel=0,
        state_channel=64,
        conv_kernel_size=4,
        inner_factor=2,
        head_channel=64,
        conv_bias=True,
        bias=False,
        chunk_size=256,
        dt_min=0.001,
        A_init_range=(1, 16),
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_rank=None,
        dt_scale=1.0,
        mlp_hidden_ratios=[4.0],
        dropout=None,
        mamba="Mamba",
        activation="relu",
        norm="batch",
        num_norm_groups=-1,
        time_injection="gate_bias",
        condition_channel=0,
        condition_injection="gate_bias",
        condition_first=False,
        use_nc_spi=False,
        nc_spi_num_prompts=16,
        nc_spi_rank=48,
        nc_spi_temperature=1.0,
        nc_spi_hard=False,
        **kwargs,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel or in_channel
        MambaClass = Mamba2 if mamba == "Mamba2" else Mamba
        self.mamba = MambaClass(
            d_model=in_channel,
            d_state=state_channel,
            d_conv=conv_kernel_size,
            expand=int(inner_factor),
            bias=bias,
            conv_bias=conv_bias,
            dt_max=dt_max,
            dt_min=dt_min,
            dt_rank=dt_rank,
            dt_init_floor=dt_init_floor,
            dt_scale=dt_scale,
            use_fast_path=False,
            use_nc_spi=use_nc_spi,
            nc_spi_num_prompts=nc_spi_num_prompts,
            nc_spi_rank=nc_spi_rank,
            nc_spi_temperature=nc_spi_temperature,
            nc_spi_hard=nc_spi_hard,
        )
        self.norm1 = NormBlock(*(get_norm(norm, in_channel, 1, num_norm_groups) + (-1,)))
        self.norm2 = NormBlock(*(get_norm(norm, in_channel, 1, num_norm_groups) + (-1,)))
        hidden_channels = [int(in_channel * ratio) for ratio in mlp_hidden_ratios]
        self.mlp = MultiLayerPerceptionBlock(
            in_channel=in_channel,
            out_channel=self.out_channel,
            hidden_channels=hidden_channels,
            activation=activation,
            dropout=dropout,
        )
        self.dense = nn.Linear(in_channel, self.out_channel) if in_channel != self.out_channel else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout is not None and dropout > 0 else nn.Identity()
        self.cinj = None
        if time_channel > 0 or condition_channel > 0:
            self.cinj = ContextInjectionBlock(
                time_channel=time_channel,
                condition_channel=condition_channel,
                out_channel=in_channel,
                time_injection=time_injection,
                condition_injection=condition_injection,
                channel_dim=-1,
                condition_first=condition_first,
            )

    def forward(self, x, t=None, c=None):
        x = x + self.dropout(self.mamba(self.norm1(x)))
        if self.cinj:
            x = self.cinj(x, t, c)
        return self.dense(x) + self.mlp(self.norm2(x))

    @staticmethod
    def is_sequence_modeling():
        return True


class MedMambaNonFFNBlock(NormBlock):
    def __init__(
        self,
        in_channel,
        out_channel=None,
        time_channel=0,
        state_channel=64,
        conv_kernel_size=4,
        inner_factor=2,
        head_channel=64,
        conv_bias=True,
        bias=False,
        chunk_size=256,
        dt_min=0.001,
        A_init_range=(1, 16),
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_rank=None,
        dt_scale=1.0,
        dropout=None,
        mamba="Mamba",
        activation="relu",
        norm="batch",
        num_norm_groups=-1,
        time_injection="gate_bias",
        condition_channel=0,
        condition_injection="gate_bias",
        condition_first=False,
        mlp_hidden_ratios=None,
        use_nc_spi=False,
        nc_spi_num_prompts=16,
        nc_spi_rank=48,
        nc_spi_temperature=1.0,
        nc_spi_hard=False,
        **kwargs,
    ):
        super().__init__(_channel_dim=-1)
        self.in_channel = in_channel
        self.out_channel = out_channel or in_channel
        MambaClass = Mamba2 if mamba == "Mamba2" else Mamba
        self.mamba = MambaClass(
            d_model=in_channel,
            d_state=state_channel,
            d_conv=conv_kernel_size,
            expand=int(inner_factor),
            bias=bias,
            conv_bias=conv_bias,
            dt_max=dt_max,
            dt_min=dt_min,
            dt_rank=dt_rank,
            dt_init_floor=dt_init_floor,
            dt_scale=dt_scale,
            use_fast_path=False,
            use_nc_spi=use_nc_spi,
            nc_spi_num_prompts=nc_spi_num_prompts,
            nc_spi_rank=nc_spi_rank,
            nc_spi_temperature=nc_spi_temperature,
            nc_spi_hard=nc_spi_hard,
        )
        self.norm, self.norm_type = get_norm(norm, in_channel, 1, num_norm_groups)
        self.act = get_act(activation)
        self.dense = nn.Linear(in_channel, self.out_channel) if in_channel != self.out_channel else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout is not None and dropout > 0 else nn.Identity()
        self.cinj = None
        if time_channel > 0 or condition_channel > 0:
            self.cinj = ContextInjectionBlock(
                time_channel=time_channel,
                condition_channel=condition_channel,
                out_channel=self.out_channel,
                time_injection=time_injection,
                condition_injection=condition_injection,
                channel_dim=-1,
                condition_first=condition_first,
            )

    def forward(self, x, t=None, c=None):
        res = x + self.dropout(self.mamba(x))
        x = self.dense(self.act(self.normalize(res)))
        if self.cinj:
            x = self.cinj(x, t, c)
        return x

    @staticmethod
    def is_sequence_modeling():
        return True


class LGSSMBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        num_scan,
        out_channel=None,
        time_channel=0,
        state_channel=None,
        conv_kernel_size=4,
        inner_factor=2.0,
        dt_rank=None,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        mlp_hidden_ratios=[4.0],
        dropout=0.0,
        conv_bias=True,
        bias=False,
        activation="silu",
        norm=None,
        num_norm_groups=0,
        time_injection="gate_bias",
        condition_channel=0,
        condition_injection="gate_bias",
        condition_first=False,
        use_rpe=False,
        rpe_max_distance=2.0,
        rpe_num_freqs=10,
        use_nc_spi=False,
        nc_spi_num_prompts=16,
        nc_spi_rank=48,
        nc_spi_temperature=1.0,
        nc_spi_hard=False,
        lgssm_block_size=48,
        with_ffn=False,
        **kwargs,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel or in_channel
        self.state_channel = state_channel or max(16, math.ceil(in_channel / 6))
        self.inner_channel = int(inner_factor * in_channel)
        self.dt_rank = dt_rank or math.ceil(in_channel / 16)
        self.num_scan = num_scan
        self.block_size = lgssm_block_size
        self.with_ffn = with_ffn
        self.in_proj = nn.Linear(in_channel, self.inner_channel * 2, bias=bias)
        self.conv = nn.Conv1d(
            self.inner_channel,
            self.inner_channel,
            groups=self.inner_channel,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            bias=conv_bias,
        )
        self.act = get_act(activation)
        self.norm1 = NormBlock(*(get_norm(norm, in_channel, 1, num_norm_groups) + (-1,)))
        self.norm2 = NormBlock(*(get_norm(norm, in_channel, 1, num_norm_groups) + (-1,)))
        self.out_proj = nn.Linear(self.inner_channel, in_channel, bias=bias)
        self.dense = nn.Linear(in_channel, self.out_channel) if in_channel != self.out_channel else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout is not None and dropout > 0 else nn.Identity()
        hidden_channels = [int(in_channel * ratio) for ratio in mlp_hidden_ratios]
        self.mlp = MultiLayerPerceptionBlock(
            in_channel=in_channel,
            out_channel=self.out_channel,
            hidden_channels=hidden_channels,
            activation=activation,
            dropout=dropout,
        ) if with_ffn else None
        self.cinj = None
        if time_channel > 0 or condition_channel > 0:
            self.cinj = ContextInjectionBlock(
                time_channel=time_channel,
                condition_channel=condition_channel,
                out_channel=in_channel,
                time_injection=time_injection,
                condition_injection=condition_injection,
                channel_dim=-1,
                condition_first=condition_first,
            )
        self.rpe = PathConditionedRPE(self.inner_channel, rpe_max_distance, rpe_num_freqs) if use_rpe else None
        self.nc_spi = NCSPIPromptGenerator(
            d_model=in_channel,
            d_state=self.state_channel,
            num_prompts=nc_spi_num_prompts,
            rank=nc_spi_rank,
            temperature=nc_spi_temperature,
            hard=nc_spi_hard,
        ) if use_nc_spi else None
        self._init_ssm_parameters("local", dt_init, dt_scale, dt_min, dt_max, dt_init_floor)
        self._init_ssm_parameters("global", dt_init, dt_scale, dt_min, dt_max, dt_init_floor)

    def _dt_init(self, dt_init, dt_scale, dt_min, dt_max, dt_init_floor):
        dt_proj = nn.Linear(self.dt_rank, self.inner_channel, bias=True)
        std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -std, std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(self.inner_channel) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    def _init_ssm_parameters(self, prefix, dt_init, dt_scale, dt_min, dt_max, dt_init_floor):
        x_proj = tuple(
            nn.Linear(self.inner_channel, self.dt_rank + self.state_channel * 2, bias=False)
            for _ in range(self.num_scan)
        )
        dt_proj = tuple(
            self._dt_init(dt_init, dt_scale, dt_min, dt_max, dt_init_floor)
            for _ in range(self.num_scan)
        )
        setattr(self, f"{prefix}_x_proj_weight", nn.Parameter(torch.stack([layer.weight for layer in x_proj], dim=0)))
        setattr(self, f"{prefix}_dt_proj_weight", nn.Parameter(torch.stack([layer.weight for layer in dt_proj], dim=0)))
        setattr(self, f"{prefix}_dt_proj_bias", nn.Parameter(torch.stack([layer.bias for layer in dt_proj], dim=0)))
        A = repeat(
            torch.arange(1, self.state_channel + 1, dtype=torch.float32),
            "n -> d n",
            d=self.inner_channel,
        )
        A_log = repeat(torch.log(A), "d n -> k d n", k=self.num_scan).flatten(0, 1)
        D = repeat(torch.ones(self.inner_channel), "d -> k d", k=self.num_scan).flatten(0, 1)
        A_param = nn.Parameter(A_log)
        D_param = nn.Parameter(D)
        A_param._no_weight_decay = True
        D_param._no_weight_decay = True
        setattr(self, f"{prefix}_A_logs", A_param)
        setattr(self, f"{prefix}_Ds", D_param)

    def _ssm(self, xs, prefix, nc_prompt=None):
        batch, scans, channels, length = xs.shape
        x_proj_weight = getattr(self, f"{prefix}_x_proj_weight")
        dt_proj_weight = getattr(self, f"{prefix}_dt_proj_weight")
        dt_proj_bias = getattr(self, f"{prefix}_dt_proj_bias")
        A_logs = getattr(self, f"{prefix}_A_logs")
        Ds = getattr(self, f"{prefix}_Ds")
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.state_channel, self.state_channel], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_proj_weight)
        if nc_prompt is not None:
            Cs = Cs + nc_prompt.permute(0, 2, 1).unsqueeze(1).to(dtype=Cs.dtype, device=Cs.device)
        out = selective_scan_fn(
            xs.float().reshape(batch, -1, length),
            dts.contiguous().float().reshape(batch, -1, length),
            -torch.exp(A_logs.float()).view(-1, self.state_channel),
            Bs.float(),
            Cs.float(),
            Ds.float().view(-1),
            z=None,
            delta_bias=dt_proj_bias.float().view(-1),
            delta_softplus=True,
            return_last_state=False,
        )
        return out.view(batch, scans, channels, length)

    def _local_ssm(self, xs, nc_prompt):
        batch, scans, channels, length = xs.shape
        pad = (-length) % self.block_size
        if pad:
            xs = F.pad(xs, (0, pad))
            if nc_prompt is not None:
                nc_prompt = F.pad(nc_prompt, (0, 0, 0, pad))
        padded_length = xs.shape[-1]
        block_count = padded_length // self.block_size
        xs_blocks = xs.view(batch, scans, channels, block_count, self.block_size)
        xs_blocks = xs_blocks.permute(0, 3, 1, 2, 4).reshape(batch * block_count, scans, channels, self.block_size)
        prompt_blocks = None
        if nc_prompt is not None:
            prompt_blocks = nc_prompt.view(batch, block_count, self.block_size, self.state_channel)
            prompt_blocks = prompt_blocks.reshape(batch * block_count, self.block_size, self.state_channel)
        out = self._ssm(xs_blocks, "local", prompt_blocks)
        out = out.view(batch, block_count, scans, channels, self.block_size)
        out = out.permute(0, 2, 3, 1, 4).reshape(batch, scans, channels, padded_length)
        return out[..., :length], padded_length, block_count

    def _global_ssm(self, local_out, padded_length, block_count):
        batch, scans, channels, length = local_out.shape
        pad = padded_length - length
        if pad:
            local_padded = F.pad(local_out, (0, pad))
        else:
            local_padded = local_out
        blocks = local_padded.view(batch, scans, channels, block_count, self.block_size)
        block_tokens = blocks.mean(dim=-1)
        block_context = self._ssm(block_tokens, "global", None)
        block_context = block_context.unsqueeze(-1).expand(-1, -1, -1, -1, self.block_size)
        block_context = block_context.reshape(batch, scans, channels, padded_length)
        return block_context[..., :length]

    def _mamba(self, x, sorted_index_list):
        batch, length, _ = x.shape
        nc_prompt = self.nc_spi(x) if self.nc_spi is not None else None
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        x_inner = channel_recover(x_inner)
        x_inner = self.act(self.conv(x_inner)[..., :length])
        xyz = _DEFORMABLE3D_CACHE.get("xyz")
        sorted_features = []
        for item in sorted_index_list:
            indices, score = item if isinstance(item, (tuple, list)) and len(item) == 2 else (item, None)
            if score is not None:
                x_sorted, _ = Deformable3DPathTrans.apply(x_inner, score)
                x_sorted = x_sorted.permute(0, 2, 1).contiguous()
            else:
                x_sorted = gather_features(x_inner, index=indices, channel_dim=1, gather_dim=-1)
            if self.rpe is not None and xyz is not None:
                xyz_sorted = _gather_points(xyz, indices)
                x_sorted = x_sorted + channel_recover(self.rpe(xyz_sorted))
            sorted_features.append(x_sorted)
        xs = torch.stack(sorted_features, dim=1)
        local_out, padded_length, block_count = self._local_ssm(xs, nc_prompt)
        global_out = self._global_ssm(local_out, padded_length, block_count)
        y = _scan_project(local_out + global_out, sorted_index_list)
        y = channel_transfer(y) * self.act(z)
        return self.out_proj(y)

    def forward(self, x, sorted_index_list, t=None, c=None):
        if len(sorted_index_list) != self.num_scan:
            raise ValueError(f"Expected {self.num_scan} scans, got {len(sorted_index_list)}.")
        if self.with_ffn:
            x = x + self.dropout(self._mamba(self.norm1(x), sorted_index_list))
            if self.cinj:
                x = self.cinj(x, t, c)
            return self.dense(x) + self.mlp(self.norm2(x))
        res = x + self.dropout(self._mamba(self.norm1(x), sorted_index_list))
        if self.cinj:
            res = self.cinj(res, t, c)
        return self.dense(res)

    @staticmethod
    def is_sequence_modeling():
        return True


class MedMambaScanBlock(LGSSMBlock):
    def __init__(self, *args, **kwargs):
        kwargs["with_ffn"] = True
        super().__init__(*args, **kwargs)


class MedMambaScan2Block(LGSSMBlock):
    pass


def get_medmamba_scan_block(name, **kwargs):
    if name in ["medmamba", "lg_ssm_ffn"]:
        return partial(MedMambaScanBlock, **kwargs)
    if name in ["lg_ssm", "lg_ssm_non_ffn", "medmamba_non_ffn", "medmamba2", "medmamba2_non_ffn"]:
        return partial(LGSSMBlock, **kwargs)
    logger.error(f"Unsupported MedMamba scan block: {name}")
    exit(1)

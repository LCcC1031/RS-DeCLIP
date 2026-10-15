"""Five-query V2 reader, global-guided aggregation, and CGVF."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _logit(p: float) -> float:
    p = float(max(1e-06, min(1.0 - 1e-06, p)))
    return math.log(p / (1.0 - p))

class GlobalConditionedP2QueryPool(nn.Module):

    def __init__(self, num_queries: int, channels: int, global_dim: int, num_heads: int, global_scale: float, spatial_sigma: float, anchor_max_offset: float, overlap_margin: float):
        super().__init__()
        if num_queries != 5:
            raise ValueError('Invalid tensor configuration.')
        if num_heads < 1 or channels % num_heads != 0:
            raise ValueError('channels must be divisible by num_heads.')
        if spatial_sigma <= 0:
            raise ValueError('Invalid tensor configuration.')
        if not 0.0 <= anchor_max_offset < 0.15:
            raise ValueError('Invalid tensor configuration.')
        if not 0.0 <= overlap_margin < 1.0:
            raise ValueError('Invalid tensor configuration.')
        self.num_queries = int(num_queries)
        self.num_heads = int(num_heads)
        self.head_dim = channels // num_heads
        self.global_scale = float(global_scale)
        self.spatial_sigma = float(spatial_sigma)
        self.anchor_max_offset = float(anchor_max_offset)
        self.overlap_margin = float(overlap_margin)
        self.query_tokens = nn.Parameter(torch.empty(num_queries, channels))
        nn.init.normal_(self.query_tokens, std=0.02)
        initial_points = torch.tensor([[0.5, 0.5], [0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]], dtype=torch.float32)
        self.register_buffer('anchor_centers', initial_points)
        self.anchor_offsets = nn.Parameter(torch.zeros_like(initial_points))
        self.query_norm = nn.LayerNorm(channels)
        self.global_proj = nn.Linear(global_dim, channels)
        self.global_norm = nn.LayerNorm(channels)
        self.key_norm = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.out_norm = nn.LayerNorm(channels)

    def forward(self, p2: torch.Tensor, z_global: torch.Tensor):
        batch, channels, height, width = p2.shape
        if z_global.shape[0] != batch:
            raise ValueError('P2 and global features must have the same batch size.')
        spatial = p2.flatten(2).transpose(1, 2)
        query = self.query_norm(self.query_tokens).unsqueeze(0)
        global_hint = self.global_norm(self.global_proj(z_global)).unsqueeze(1)
        query = query + self.global_scale * global_hint
        q = self.q_proj(query).reshape(batch, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(self.key_norm(spatial)).reshape(batch, height * width, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(spatial).reshape(batch, height * width, self.num_heads, self.head_dim).transpose(1, 2)
        y = (torch.arange(height, device=p2.device, dtype=p2.dtype) + 0.5) / height
        x = (torch.arange(width, device=p2.device, dtype=p2.dtype) + 0.5) / width
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        positions = torch.stack((yy, xx), dim=-1).reshape(1, 1, height * width, 2)
        anchors = self.anchor_centers + self.anchor_max_offset * torch.tanh(self.anchor_offsets)
        anchors = anchors.to(dtype=p2.dtype).reshape(1, self.num_queries, 1, 2)
        distance_sq = (positions - anchors).square().sum(dim=-1)
        spatial_bias = -distance_sq / (2.0 * self.spatial_sigma ** 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.head_dim ** (-0.5)
        attention = F.softmax(logits + spatial_bias.unsqueeze(1), dim=-1)
        mean_attention = attention.mean(dim=1)
        unit_attention = F.normalize(mean_attention, p=2, dim=-1, eps=1e-08)
        overlap = torch.bmm(unit_attention, unit_attention.transpose(1, 2))
        off_diagonal = ~torch.eye(self.num_queries, device=p2.device, dtype=torch.bool).unsqueeze(0)
        overlap_pairs = overlap.masked_select(off_diagonal)
        overlap_loss = F.relu(overlap_pairs - self.overlap_margin).square().mean()
        overlap_mean = overlap_pairs.mean().detach()
        gathered = torch.matmul(attention, v)
        gathered = gathered.transpose(1, 2).reshape(batch, self.num_queries, channels)
        return (self.out_norm(self.out_proj(gathered)), overlap_loss, overlap_mean)

def multi_positive_contrastive_loss(scores_t2i: torch.Tensor, pos_mask: torch.Tensor, logit_scale: torch.Tensor, t2i_weight: float=1.0):
    logits = scores_t2i * logit_scale
    labels_t2i = pos_mask.float().argmax(dim=1)
    loss_t2i = F.cross_entropy(logits, labels_t2i)
    logits_i2t = logits.t()
    pos_i2t = pos_mask.t()
    log_den = torch.logsumexp(logits_i2t, dim=1)
    masked_pos = logits_i2t.masked_fill(~pos_i2t, -1000000000.0)
    log_pos = torch.logsumexp(masked_pos, dim=1)
    loss_i2t = -(log_pos - log_den).mean()
    return ((loss_i2t + float(t2i_weight) * loss_t2i) / (1.0 + float(t2i_weight)), loss_i2t.detach(), loss_t2i.detach())

def diversity_loss(local_tokens: torch.Tensor):
    sim = torch.bmm(local_tokens, local_tokens.transpose(1, 2))
    n = sim.shape[1]
    if n <= 1:
        return torch.tensor(0.0, device=local_tokens.device, dtype=local_tokens.dtype)
    eye = torch.eye(n, device=sim.device, dtype=torch.bool).unsqueeze(0)
    offdiag = sim.masked_select(~eye)
    return offdiag.pow(2).mean()


class P2QueryCGVF(nn.Module):
    """Read P2 and return normalized local tokens plus one retrieval descriptor.

    p2: [B,256,H,W]; z_global: [B,512].
    The five anchors are soft spatial preferences, not hard crops.
    """

    def __init__(self, dim=512, channels=256, dropout=0.10):
        super().__init__()
        self.query_pool = GlobalConditionedP2QueryPool(
            num_queries=5, channels=channels, global_dim=dim, num_heads=4,
            global_scale=0.15, spatial_sigma=0.35,
            anchor_max_offset=0.10, overlap_margin=0.60,
        )
        self.local_proj = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.fusion_refine = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim),
        )
        self.cvgf_gate_alpha_logit = nn.Parameter(torch.tensor(_logit(0.20 / 0.50), dtype=torch.float32))
        self.cvgf_gate_c0 = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.cvgf_gate_tau_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.07 - 1e-3) - 1.0), dtype=torch.float32))

    def get_cgvf_gate_params(self):
        alpha = 0.50 * torch.sigmoid(self.cvgf_gate_alpha_logit)
        tau = F.softplus(self.cvgf_gate_tau_raw) + 1e-3
        return alpha, self.cvgf_gate_c0, tau

    def forward(self, p2, z_global):
        z_global = F.normalize(z_global.float(), dim=-1, eps=1e-8)
        raw_tokens, overlap_loss, overlap_mean = self.query_pool(p2, z_global)
        local_tokens = F.normalize(self.local_proj(raw_tokens).float(), dim=-1, eps=1e-8)
        sim_lg = (local_tokens * z_global.unsqueeze(1)).sum(dim=-1).clamp(-1.0, 1.0)
        weights = F.softmax(sim_lg / 0.07, dim=1)
        z_local = torch.sum(weights.unsqueeze(-1) * local_tokens, dim=1)
        z_local = F.normalize(z_local + 0.10 * self.fusion_refine(z_local), dim=-1, eps=1e-8)
        consistency = (z_local * z_global).sum(dim=-1).clamp(-1.0, 1.0)
        delta = z_local - consistency.unsqueeze(-1) * z_global
        alpha, center, temperature = self.get_cgvf_gate_params()
        alpha = alpha.to(device=consistency.device, dtype=consistency.dtype)
        center = center.to(device=consistency.device, dtype=consistency.dtype)
        temperature = temperature.to(device=consistency.device, dtype=consistency.dtype).clamp_min(1e-6)
        gate = alpha * torch.sigmoid((consistency - center) / temperature)
        descriptor = F.normalize(z_global + gate.unsqueeze(-1) * delta, dim=-1, eps=1e-8)
        return local_tokens, descriptor, {
            "aggregation_weights": weights,
            "query_overlap_loss": overlap_loss,
            "query_overlap_mean": overlap_mean,
            "gate": gate,
        }


def stage2_objective(local_tokens, descriptor, z_global, z_text, positive_mask,
                     logit_scale, query_overlap_loss, epoch):
    """Use all raw captions in the batch; rows of positive_mask correspond to texts.

    Each text has exactly one paired image; each image can have several captions.
    The original first-two-epoch score warmup is preserved.
    """
    if int(epoch) < 1:
        raise ValueError("epoch must start at 1.")
    raw_scores = z_text @ z_global.t()
    fused_scores = z_text @ descriptor.t()
    alpha = min(1.0, float(epoch) / 2.0)
    final_scores = (1.0 - alpha) * raw_scores + alpha * fused_scores
    retrieval, i2t, t2i = multi_positive_contrastive_loss(
        final_scores, positive_mask, logit_scale, t2i_weight=1.2,
    )
    consistency = (descriptor * z_global).sum(dim=-1).clamp(-1.0, 1.0)
    global_alignment = (1.0 - consistency).mean()
    token_consistency = (local_tokens * z_global.unsqueeze(1)).sum(dim=-1).clamp(-1.0, 1.0)
    token_alignment = (1.0 - token_consistency).mean()
    token_diversity = diversity_loss(local_tokens)
    loss = (retrieval + 0.20 * global_alignment + 0.01 * token_alignment
            + 0.02 * token_diversity + 0.02 * query_overlap_loss)
    return loss, {
        "retrieval": retrieval, "i2t": i2t, "t2i": t2i,
        "global_alignment": global_alignment, "token_alignment": token_alignment,
        "token_diversity": token_diversity, "attention_overlap": query_overlap_loss,
    }

"""Stage-1 core objectives; global encoders alone are used for retrieval."""

import random
import re

import torch
from torch import nn
from torch.nn import functional as F


RS_PROMPT = 'An aerial satellite image of '

W_ALPHA = 0.2

W_BETA = 0.2

W_GAMMA = 0.05

W_CLIP = 1.0 - W_ALPHA - W_BETA - W_GAMMA

USE_ASYMMETRIC_T2I_LOSS = True

T2I_LOSS_WEIGHT = 1.2

USE_HIGH_SIM_NEG_DOWNWEIGHT = True

HIGH_SIM_NEG_THRESHOLD = 0.85

HIGH_SIM_NEG_MIN_WEIGHT = 0.6

HIGH_SIM_NEG_POWER = 2.0

USE_CROSS_ATTENTION = True

W_CROSS = 0.1

CROSS_NUM_HEADS = 8

CROSS_DROPOUT = 0.1

USE_ADAPTIVE_DETAIL_ALIGNMENT = True

W_TP = 0.02

TP_MARGIN = 0.05

TP_TOPK_PATCH = 3

TP_IDF_ALPHA = 1.0

TP_BM25_IDF_CLIP_PERCENTILE = 95.0

TP_IDF_BUILD_BATCH_SIZE = 2048

USE_COUNTERFACTUAL_DETAIL_RANK = True

W_CF = 0.01

CF_MARGIN = 0.05

CF_SUB_BATCH = 128

COUNTERFACTUAL_REPLACEMENTS = [(' one ', ' two '), (' two ', ' four '), (' three ', ' five '), (' four ', ' two '), (' five ', ' three '), (' several ', ' one '), (' many ', ' one '), (' next to ', ' far from '), (' near ', ' far from '), (' beside ', ' around '), (' around ', ' beside '), (' between ', ' around '), (' along ', ' across '), (' surrounded by ', ' beside '), (' blue ', ' white '), (' white ', ' blue '), (' red ', ' white '), (' circular ', ' rectangular '), (' rectangular ', ' circular '), (' large ', ' small '), (' small ', ' large '), (' baseball fields ', ' basketball courts '), (' baseball field ', ' basketball court '), (' basketball courts ', ' baseball fields '), (' basketball court ', ' baseball field '), (' tennis courts ', ' parking lots '), (' tennis court ', ' parking lot '), (' parking lots ', ' playgrounds '), (' parking lot ', ' playground '), (' cars ', ' buildings '), (' car ', ' building '), (' airplanes ', ' cars '), (' airplane ', ' car '), (' aircraft ', ' cars '), (' bridge ', ' building '), (' river ', ' road '), (' storage tanks ', ' buildings '), (' storage tank ', ' building ')]

class SimSiamPredictor(nn.Module):

    def __init__(self, dim=512):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), nn.ReLU(inplace=True))
        self.layer2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.layer2(self.layer1(x))

class NNSMemoryBank(nn.Module):

    def __init__(self, size=8192, dim=512):
        super().__init__()
        self.size = size
        self.dim = dim
        self.ptr = 0
        self.register_buffer('queue_t', F.normalize(torch.randn(size, dim), dim=-1))

    @torch.no_grad()
    def update(self, z_t):
        batch_size = z_t.size(0)
        if self.ptr + batch_size > self.size:
            batch_size = self.size - self.ptr
            z_t = z_t[:batch_size]
        self.queue_t[self.ptr:self.ptr + batch_size] = z_t.detach()
        self.ptr = (self.ptr + batch_size) % self.size

    @torch.no_grad()
    def get_nearest_neighbors(self, z_t):
        sim_t2t = z_t @ self.queue_t.T
        _, nn_idx = sim_t2t.max(dim=-1)
        return self.queue_t[nn_idx]

def encode_image_with_patch_tokens(model, images):
    visual = model.visual
    x = visual.conv1(images.type(visual.conv1.weight.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype)
    cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x)
    if visual.proj is not None:
        x = x @ visual.proj
    cls_feat = F.normalize(x[:, 0, :].float(), dim=-1, eps=1e-08)
    patch_feat = F.normalize(x[:, 1:, :].float(), dim=-1, eps=1e-08)
    return (cls_feat, patch_feat)

def encode_text_with_token_features(model, text):
    x = model.token_embedding(text).type(model.dtype)
    x = x + model.positional_embedding.type(model.dtype)
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).type(model.dtype)
    token_feat = x @ model.text_projection
    eot_idx = text.argmax(dim=-1)
    cls_feat = token_feat[torch.arange(text.shape[0], device=text.device), eot_idx]
    cls_feat = F.normalize(cls_feat.float(), dim=-1, eps=1e-08)
    token_feat = F.normalize(token_feat.float(), dim=-1, eps=1e-08)
    token_mask = text.ne(0)
    return (cls_feat, token_feat, token_mask)

class LightweightCrossAttentionAligner(nn.Module):

    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.txt_to_img = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.img_to_txt = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_img_q = nn.LayerNorm(dim)
        self.ln_txt_q = nn.LayerNorm(dim)
        self.ln_img_kv = nn.LayerNorm(dim)
        self.ln_txt_kv = nn.LayerNorm(dim)
        self.img_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.txt_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, z_img, img_tokens, z_txt, txt_tokens, txt_mask=None):
        q_txt = self.ln_txt_q(z_txt).unsqueeze(1)
        kv_img = self.ln_img_kv(img_tokens)
        local_img, _ = self.txt_to_img(q_txt, kv_img, kv_img, need_weights=False)
        local_img = local_img.squeeze(1)
        cross_img = F.normalize(z_img + self.img_out(local_img), dim=-1, eps=1e-08)
        q_img = self.ln_img_q(z_img).unsqueeze(1)
        kv_txt = self.ln_txt_kv(txt_tokens)
        key_padding_mask = None
        if txt_mask is not None:
            key_padding_mask = ~txt_mask
        local_txt, _ = self.img_to_txt(q_img, kv_txt, kv_txt, key_padding_mask=key_padding_mask, need_weights=False)
        local_txt = local_txt.squeeze(1)
        cross_txt = F.normalize(z_txt + self.txt_out(local_txt), dim=-1, eps=1e-08)
        return (cross_img, cross_txt)

def cross_attention_alignment_loss(cross_attn, z_img, img_tokens, z_txt, txt_tokens, txt_mask, logit_scale):
    cross_img, cross_txt = cross_attn(z_img, img_tokens, z_txt, txt_tokens, txt_mask)
    asym_w = T2I_LOSS_WEIGHT if USE_ASYMMETRIC_T2I_LOSS else 1.0
    loss_i2t = contrastive_loss(cross_img, z_txt, logit_scale, neg_ref1=z_txt, neg_ref2=z_txt, t2i_weight=asym_w)
    loss_t2i = contrastive_loss(z_img, cross_txt, logit_scale, neg_ref1=cross_txt, neg_ref2=cross_txt, t2i_weight=asym_w)
    return (loss_i2t + loss_t2i) / 2.0

def _replace_once_ignore_case(text, src_phrase, tgt_phrase):
    pattern = re.escape(src_phrase.strip())
    pattern = '(?<!\\w)' + pattern + '(?!\\w)'
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if m is None:
        return (text, False)
    replaced = re.sub(pattern, tgt_phrase.strip(), text, count=1, flags=re.IGNORECASE)
    return (replaced, replaced != text)

def build_counterfactual_caption(caption):
    pairs = COUNTERFACTUAL_REPLACEMENTS[:]
    random.shuffle(pairs)
    padded = ' ' + caption.strip() + ' '
    for src_phrase, tgt_phrase in pairs:
        cf, ok = _replace_once_ignore_case(padded, src_phrase, tgt_phrase)
        if ok:
            return (cf.strip(), True)
    return (caption, False)

class AdaptiveTokenPatchAligner(nn.Module):

    def __init__(self, dim=512, vocab_size=49408, token_idf=None, idf_alpha=1.0, topk_patch=3, margin=0.05):
        super().__init__()
        self.idf_alpha = float(idf_alpha)
        self.topk_patch = int(topk_patch)
        self.margin = float(margin)
        self.token_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        if token_idf is None:
            token_idf = torch.zeros(vocab_size, dtype=torch.float32)
        self.register_buffer('token_idf', token_idf.float())

    def compute_token_weights(self, txt_tokens, text_ids, txt_mask):
        gate = torch.sigmoid(self.token_gate(txt_tokens).squeeze(-1))
        idf = self.token_idf[text_ids].to(txt_tokens.device).float()
        valid = txt_mask.float()
        weights = 1.0 + self.idf_alpha * idf * gate
        weights = weights * valid
        return weights

    def detail_score(self, img_tokens, txt_tokens, text_ids, txt_mask):
        sim = torch.bmm(txt_tokens, img_tokens.transpose(1, 2))
        k = min(self.topk_patch, sim.size(-1))
        token_score = sim.topk(k=k, dim=-1).values.mean(dim=-1)
        weights = self.compute_token_weights(txt_tokens, text_ids, txt_mask)
        score = (token_score * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-06)
        return score

    def forward(self, img_tokens, txt_tokens, text_ids, txt_mask):
        if img_tokens.size(0) <= 1:
            return img_tokens.new_tensor(0.0)
        pos_score = self.detail_score(img_tokens, txt_tokens, text_ids, txt_mask)
        perm = torch.randperm(img_tokens.size(0), device=img_tokens.device)
        if torch.equal(perm, torch.arange(img_tokens.size(0), device=img_tokens.device)):
            perm = torch.roll(perm, shifts=1)
        neg_score = self.detail_score(img_tokens[perm], txt_tokens, text_ids, txt_mask)
        loss = F.relu(self.margin + neg_score - pos_score).mean()
        return loss

def counterfactual_detail_rank_loss(model, z_img, z_txt_pos, txt_cf, cf_mask, margin=0.05, sub_batch=128):
    idx = torch.nonzero(cf_mask.bool(), as_tuple=False).flatten()
    if idx.numel() == 0:
        return z_img.new_tensor(0.0)
    if idx.numel() > sub_batch:
        pick = torch.randperm(idx.numel(), device=idx.device)[:sub_batch]
        idx = idx[pick]
    z_cf = F.normalize(model.encode_text(txt_cf[idx]).float(), dim=-1, eps=1e-08)
    sim_pos = (z_img[idx] * z_txt_pos[idx]).sum(dim=-1)
    sim_cf = (z_img[idx] * z_cf).sum(dim=-1)
    return F.relu(float(margin) + sim_cf - sim_pos).mean()

def contrastive_loss(feat1, feat2, logit_scale, gate=None, neg_ref1=None, neg_ref2=None, neg_sim_threshold=HIGH_SIM_NEG_THRESHOLD, neg_min_weight=HIGH_SIM_NEG_MIN_WEIGHT, neg_power=HIGH_SIM_NEG_POWER, t2i_weight=1.0):
    logits = logit_scale * feat1 @ feat2.T
    labels = torch.arange(logits.size(0), device=logits.device)

    def weighted_ce(logits_, labels_, neg_weight_):
        log_w = torch.log(neg_weight_.clamp_min(1e-06))
        weighted_logits = logits_ + log_w
        pos_logits = logits_.gather(1, labels_.view(-1, 1)).squeeze(1)
        return -pos_logits + torch.logsumexp(weighted_logits, dim=1)
    if USE_HIGH_SIM_NEG_DOWNWEIGHT and neg_ref1 is not None and (neg_ref2 is not None):
        with torch.no_grad():
            ref1 = F.normalize(neg_ref1.detach().float(), dim=-1, eps=1e-08)
            ref2 = F.normalize(neg_ref2.detach().float(), dim=-1, eps=1e-08)
            neg_text_sim = torch.clamp(ref1 @ ref2.T, min=-1.0, max=1.0)
            denom = max(1e-06, 1.0 - float(neg_sim_threshold))
            down_strength = ((neg_text_sim - float(neg_sim_threshold)) / denom).clamp(0.0, 1.0)
            down_strength = down_strength.pow(float(neg_power))
            neg_weight = 1.0 - down_strength * (1.0 - float(neg_min_weight))
            eye = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
            neg_weight = neg_weight.masked_fill(eye, 1.0)
        loss1 = weighted_ce(logits, labels, neg_weight)
        loss2 = weighted_ce(logits.T, labels, neg_weight.T)
    else:
        loss1 = F.cross_entropy(logits, labels, reduction='none')
        loss2 = F.cross_entropy(logits.T, labels, reduction='none')
    t2i_weight = float(t2i_weight)
    loss = (loss1 + t2i_weight * loss2) / (1.0 + t2i_weight)
    if gate is not None:
        loss = loss * gate
    return loss.mean()

def simsiam_loss(p1, z2, p2, z1):
    criterion = nn.CosineSimilarity(dim=1)
    loss1 = -criterion(p1, z2.detach()).mean()
    loss2 = -criterion(p2, z1.detach()).mean()
    return (loss1 + loss2) * 0.5


@torch.no_grad()
def build_clipped_bm25_idf(tokenized_captions, vocab_size, percentile=95.0):
    """Compute the corpus prior from tokenized training captions, including both sources."""
    document_frequency = torch.zeros(vocab_size, dtype=torch.float32)
    document_count = 0
    for row in tokenized_captions:
        ids = torch.unique(torch.as_tensor(row).detach().cpu())
        ids = ids[(ids > 0) & (ids < vocab_size)].long()
        document_frequency[ids] += 1.0
        document_count += 1
    idf = torch.zeros(vocab_size, dtype=torch.float32)
    valid = document_frequency > 0
    if valid.any():
        df = document_frequency[valid]
        values = torch.log(1.0 + (float(max(1, document_count)) - df + 0.5) / (df + 0.5))
        percentile = min(max(float(percentile), 50.0), 100.0)
        cap = torch.quantile(values, percentile / 100.0).clamp_min(1e-6)
        idf[valid] = values.clamp(min=0.0, max=float(cap.item())) / cap
    idf[0] = 0.0
    return idf.clamp(0.0, 1.0)



class Stage1Objective(nn.Module):
    """Wrap the original training objectives around an OpenAI-CLIP-compatible model.

    txt1 is an AI-polished caption; txt2 is a raw caption of the same image.
    Input token IDs and images must already be on the model device.
    Update the NNS queue after the optimizer step, using the returned text features.
    """

    def __init__(self, model, token_idf, dim=512):
        super().__init__()
        self.model = model
        self.predictor = SimSiamPredictor(dim)
        self.memory_bank = NNSMemoryBank(size=8192, dim=dim)
        self.cross_attn = LightweightCrossAttentionAligner(dim, CROSS_NUM_HEADS, CROSS_DROPOUT)
        self.detail_aligner = AdaptiveTokenPatchAligner(
            dim=dim, vocab_size=token_idf.numel(), token_idf=token_idf,
            idf_alpha=TP_IDF_ALPHA, topk_patch=TP_TOPK_PATCH, margin=TP_MARGIN,
        )

    def forward(self, img_v1, img_v2, txt1, txt2, txt_cf, cf_mask):
        model = self.model
        predictor = self.predictor
        memory_bank = self.memory_bank
        cross_attn = self.cross_attn
        detail_aligner = self.detail_aligner
        z_i1, img_tokens1 = encode_image_with_patch_tokens(model, img_v1)
        z_i2 = F.normalize(model.encode_image(img_v2).float(), dim=-1, eps=1e-08)
        z_t1, txt_tokens1, txt_mask1 = encode_text_with_token_features(model, txt1)
        z_t2 = F.normalize(model.encode_text(txt2).float(), dim=-1, eps=1e-08)
        logit_scale = torch.clamp(model.logit_scale, max=4.6052).float().exp()
        asym_w = T2I_LOSS_WEIGHT if USE_ASYMMETRIC_T2I_LOSS else 1.0
        L_CLIP = contrastive_loss(z_i1, z_t1, logit_scale, neg_ref1=z_t1, neg_ref2=z_t1, t2i_weight=asym_w)
        L_MVS = (contrastive_loss(z_i1, z_t2, logit_scale, neg_ref1=z_t2, neg_ref2=z_t2, t2i_weight=asym_w) + contrastive_loss(z_i2, z_t1, logit_scale, neg_ref1=z_t1, neg_ref2=z_t1, t2i_weight=asym_w) + contrastive_loss(z_i2, z_t2, logit_scale, neg_ref1=z_t2, neg_ref2=z_t2, t2i_weight=asym_w)) / 3.0
        L_ISS = simsiam_loss(predictor(z_i1), z_i2, predictor(z_i2), z_i1)
        with torch.no_grad():
            sim_t = F.cosine_similarity(z_t1.detach(), z_t2.detach(), dim=-1)
            tss_gate = ((sim_t + 1.0) / 2.0) ** 2
        L_TSS = contrastive_loss(z_t1, z_t2, logit_scale, gate=tss_gate, neg_ref1=z_t1, neg_ref2=z_t2)
        nn_t = memory_bank.get_nearest_neighbors(z_t1)
        L_NNS = (contrastive_loss(z_i1, nn_t, logit_scale) + contrastive_loss(z_i2, nn_t, logit_scale)) / 2.0
        if USE_CROSS_ATTENTION:
            L_CROSS = cross_attention_alignment_loss(cross_attn, z_i1, img_tokens1, z_t1, txt_tokens1, txt_mask1, logit_scale)
        else:
            L_CROSS = z_i1.new_tensor(0.0)
        if USE_ADAPTIVE_DETAIL_ALIGNMENT:
            L_TP = detail_aligner(img_tokens1, txt_tokens1, txt1, txt_mask1)
        else:
            L_TP = z_i1.new_tensor(0.0)
        if USE_COUNTERFACTUAL_DETAIL_RANK:
            L_CF = counterfactual_detail_rank_loss(model, z_i1, z_t1, txt_cf, cf_mask, margin=CF_MARGIN, sub_batch=CF_SUB_BATCH)
        else:
            L_CF = z_i1.new_tensor(0.0)
        loss = W_CLIP * L_CLIP + W_ALPHA * (L_ISS + L_TSS) + W_BETA * L_MVS + W_GAMMA * L_NNS + W_CROSS * L_CROSS + W_TP * L_TP + W_CF * L_CF
        return loss, {
            "clip": L_CLIP, "iss": L_ISS, "tss": L_TSS, "mvs": L_MVS,
            "nns": L_NNS, "cross": L_CROSS, "adca": L_TP, "counterfactual": L_CF,
            "text_features": z_t1.detach(),
        }

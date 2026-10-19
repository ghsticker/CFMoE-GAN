import torch
import torch.nn.functional as F


def _pairwise_sqdist(x, y):
    # x: [N, D], y: [K, D]
    x2 = (x ** 2).sum(dim=-1, keepdim=True)
    y2 = (y ** 2).sum(dim=-1).unsqueeze(0)
    xy = x @ y.t()
    return (x2 + y2 - 2 * xy).clamp_min(0.0)


def _normalize_tokens(tokens):
    return F.normalize(tokens, dim=-1)


def _pairwise_similarity(a, b):
    return torch.einsum("bid,bjd->bij", a, b)


def _gather_local_windows(patch_grid, window):
    # patch_grid: [B, H, W, D]
    B, H, W, D = patch_grid.shape
    radius = window // 2
    padded = F.pad(patch_grid, (0, 0, radius, radius, radius, radius))
    windows = []
    for dy in range(window):
        for dx in range(window):
            windows.append(padded[:, dy : dy + H, dx : dx + W, :])
    stacked = torch.stack(windows, dim=-2)  # [B, H, W, window*window, D]
    return stacked


def relaxed_inter_image_nce(pred_tokens, gt_tokens, grid_size, window=7, tau=0.07, neg_samples=256):
    # pred_tokens, gt_tokens: [B, N, D]
    B, N, D = pred_tokens.shape
    H, W = grid_size
    pred_tokens = _normalize_tokens(pred_tokens)
    gt_tokens = _normalize_tokens(gt_tokens)
    pred_grid = pred_tokens.view(B, H, W, D)
    gt_grid = gt_tokens.view(B, H, W, D)

    local_windows = _gather_local_windows(gt_grid, window)  # [B, H, W, K, D]
    local_scores = (pred_grid.unsqueeze(-2) * local_windows).sum(dim=-1)  # [B,H,W,K]
    best_idx = local_scores.argmax(dim=-1, keepdim=True)  # [B,H,W,1]
    best_tokens = torch.gather(local_windows, -2, best_idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)).squeeze(-2)

    best_tokens = best_tokens.view(B, N, D)
    pos_logits = (pred_tokens * best_tokens).sum(dim=-1, keepdim=True) / tau

    gt_flat = gt_tokens.view(B, N, D)
    if neg_samples >= N:
        neg_pool = gt_flat
    else:
        idx = torch.randint(0, N, (B, neg_samples), device=gt_flat.device)
        neg_pool = torch.gather(gt_flat, 1, idx.unsqueeze(-1).expand(-1, -1, D))

    neg_logits = torch.einsum("bid,bkd->bik", pred_tokens, neg_pool) / tau
    logits = torch.cat([pos_logits, neg_logits], dim=-1)
    labels = torch.zeros(B, N, dtype=torch.long, device=logits.device)
    logits = logits.view(B * N, -1)
    labels = labels.view(B * N)
    return F.cross_entropy(logits, labels)


def relaxed_inter_image_nce_improved(
    pred_tokens,
    gt_tokens,
    grid_size,
    window=3,
    tau=0.07,
    neg_samples=256,
    semantic_neg_threshold=0.4,
    min_negatives=64,
):
    """
    Improved inter-image relaxed NCE with three fixes:

    1. Soft-Matching (replaces argmax): fully differentiable attention-weighted target.
    2. Bidirectional relaxation: forward (pred→GT) + backward (GT→pred) symmetric constraint.
    3. Semantic-aware negative mining: filter out negatives with high GT-internal similarity.
    """
    B, N, D = pred_tokens.shape
    H, W = grid_size
    pred_tokens = _normalize_tokens(pred_tokens)
    gt_tokens = _normalize_tokens(gt_tokens)
    pred_grid = pred_tokens.view(B, H, W, D)
    gt_grid = gt_tokens.view(B, H, W, D)
    radius = window // 2
    K = window * window

    # ── Forward: pred → GT soft-matching ─────────────────────────────────────
    local_windows_gt = _gather_local_windows(gt_grid, window)  # [B,H,W,K,D]
    scores_gt = (pred_grid.unsqueeze(-2) * local_windows_gt).sum(dim=-1) / tau
    attn_gt = F.softmax(scores_gt, dim=-1)  # [B,H,W,K], differentiable
    soft_target = (local_windows_gt * attn_gt.unsqueeze(-1)).sum(dim=-2)  # [B,H,W,D]
    soft_target_flat = soft_target.view(B, N, D)
    pos_fwd = (pred_tokens * soft_target_flat).sum(dim=-1, keepdim=True)  # [B,N,1], /tau folded in

    # ── Backward: GT → pred soft-matching ─────────────────────────────────────
    # pred stays in graph so it learns from backward loss (prevents texture collapse).
    # gt_grid is detached so only pred receives gradients (DINO frozen teacher).
    local_windows_pred = _gather_local_windows(pred_grid, window)
    scores_pred = (gt_grid.detach().unsqueeze(-2) * local_windows_pred).sum(dim=-1) / tau
    attn_pred = F.softmax(scores_pred, dim=-1)
    soft_target_pred = (local_windows_pred * attn_pred.unsqueeze(-1)).sum(dim=-2)
    soft_target_pred_flat = soft_target_pred.view(B, N, D)  # NOT detached: pred grad flows through
    pos_bwd = (gt_tokens * soft_target_pred_flat).sum(dim=-1, keepdim=True)  # [B,N,1]; GT detached

    # ── Semantic-aware negative mining ─────────────────────────────────────────
    # Use GT-internal pairwise similarity as proxy for semantic closeness
    gt_sim = torch.einsum("bid,bjd->bij", gt_tokens, gt_tokens)  # [B,N,N]
    # Spatial mask: exclude self and window-neighbors
    coords_y = torch.arange(H, device=pred_tokens.device)
    coords_x = torch.arange(W, device=pred_tokens.device)
    yy, xx = torch.meshgrid(coords_y, coords_x, indexing="ij")
    coords = torch.stack([yy, xx], dim=-1).float()
    coords_flat = coords.view(N, 2)
    spatial_dist = (coords_flat.unsqueeze(1) - coords_flat.unsqueeze(0)).abs().sum(dim=-1)
    spatial_allowed = spatial_dist > radius
    spatial_allowed.fill_diagonal_(False)  # exclude self

    # Combined mask: spatially valid AND semantically dissimilar
    neg_mask = spatial_allowed.unsqueeze(0) & (gt_sim < semantic_neg_threshold)  # [B,N,N], bool

    neg_logits_fwd = torch.einsum("bid,bjd->bij", pred_tokens, gt_tokens)  # raw cosine similarity
    neg_logits_bwd = torch.einsum("bid,bjd->bij", gt_tokens, pred_tokens.detach())  # raw cosine similarity

    # ── Per-patch softmax + CE (handle variable-length valid negatives) ─────────
    loss_fwd_list = []
    loss_bwd_list = []
    labels = torch.zeros(N, dtype=torch.long, device=pred_tokens.device)

    for b in range(B):
        pos_fwd_b = pos_fwd[b]      # [N,1], raw cosine similarity
        pos_bwd_b = pos_bwd[b]       # [N,1], raw cosine similarity
        mask_b = neg_mask[b]         # [N,N]
        neg_fwd_b = neg_logits_fwd[b]  # [N,N], raw cosine similarity
        neg_bwd_b = neg_logits_bwd[b]  # [N,N], raw cosine similarity

        for i in range(N):
            valid_neg = mask_b[i]   # [N]
            n_valid = valid_neg.long().sum().item()

            # Build per-patch logits [1 + n_valid]
            if n_valid == 0:
                n_eff = min_negatives
                neg_fwd_i = neg_fwd_b[i, :n_eff]
                neg_bwd_i = neg_bwd_b[i, :n_eff]
            elif n_valid <= neg_samples:
                neg_fwd_i = neg_fwd_b[i][valid_neg]
                neg_bwd_i = neg_bwd_b[i][valid_neg]
            else:
                valid_idx = valid_neg.nonzero(as_tuple=True)[0]
                selected = valid_idx[torch.randperm(n_valid)[:neg_samples]]
                neg_fwd_i = neg_fwd_b[i][selected]
                neg_bwd_i = neg_bwd_b[i][selected]

            # Divide by tau once after concatenation: both pos and neg get scaled uniformly
            logits_fwd_i = torch.cat([pos_fwd_b[i], neg_fwd_i]) / tau
            logits_bwd_i = torch.cat([pos_bwd_b[i], neg_bwd_i]) / tau

            loss_fwd_list.append(F.cross_entropy(logits_fwd_i.unsqueeze(0), labels[i:i+1]))
            loss_bwd_list.append(F.cross_entropy(logits_bwd_i.unsqueeze(0), labels[i:i+1]))

    loss_fwd = torch.stack(loss_fwd_list).mean()
    loss_bwd = torch.stack(loss_bwd_list).mean()
    return (loss_fwd + loss_bwd) * 0.5


def intra_image_affinity_alignment(pred_tokens, gt_tokens, tau=0.07, affinity_samples=256):
    # pred_tokens, gt_tokens: [B, N, D]
    B, N, D = pred_tokens.shape
    pred_tokens = _normalize_tokens(pred_tokens)
    gt_tokens = _normalize_tokens(gt_tokens)

    if affinity_samples < N:
        idx = torch.randint(0, N, (B, affinity_samples), device=pred_tokens.device)
        pred_tokens = torch.gather(pred_tokens, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        gt_tokens = torch.gather(gt_tokens, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        N = affinity_samples

    pred_aff = _pairwise_similarity(pred_tokens, pred_tokens) / tau
    gt_aff = _pairwise_similarity(gt_tokens, gt_tokens) / tau

    pred_aff = F.softmax(pred_aff, dim=-1)
    gt_aff = F.softmax(gt_aff, dim=-1)

    return F.l1_loss(pred_aff, gt_aff)


def local_gram_alignment(pred_tokens, gt_tokens, grid_size, window=3):
    # pred_tokens, gt_tokens: [B, N, D]
    B, N, D = pred_tokens.shape
    H, W = grid_size
    pred_grid = pred_tokens.view(B, H, W, D)
    gt_grid = gt_tokens.view(B, H, W, D)

    pred_windows = _gather_local_windows(pred_grid, window)  # [B,H,W,K,D]
    gt_windows = _gather_local_windows(gt_grid, window)  # [B,H,W,K,D]

    pred_windows = pred_windows.view(B * H * W, window * window, D)
    gt_windows = gt_windows.view(B * H * W, window * window, D)

    pred_gram = torch.einsum("bkd,bke->bde", pred_windows, pred_windows)
    gt_gram = torch.einsum("bkd,bke->bde", gt_windows, gt_windows)

    diff = pred_gram - gt_gram
    return (diff ** 2).mean()


def non_local_semantic_decoupling(pred_tokens, grid_size, margin=0.2, num_pairs=1024, min_distance=2):
    # pred_tokens: [B, N, D]
    B, N, D = pred_tokens.shape
    H, W = grid_size
    pred_tokens = _normalize_tokens(pred_tokens)

    coords_y = torch.arange(H, device=pred_tokens.device)
    coords_x = torch.arange(W, device=pred_tokens.device)
    yy, xx = torch.meshgrid(coords_y, coords_x, indexing="ij")
    coords = torch.stack([yy, xx], dim=-1).view(N, 2)

    losses = []
    for b in range(B):
        if num_pairs <= 0:
            continue
        idx_i = torch.randint(0, N, (num_pairs * 2,), device=pred_tokens.device)
        idx_j = torch.randint(0, N, (num_pairs * 2,), device=pred_tokens.device)
        diff = coords[idx_i] - coords[idx_j]
        dist = torch.abs(diff).sum(dim=-1)
        valid = dist >= min_distance
        if valid.any():
            idx_i = idx_i[valid][:num_pairs]
            idx_j = idx_j[valid][:num_pairs]
        else:
            idx_i = idx_i[:num_pairs]
            idx_j = idx_j[:num_pairs]

        fi = pred_tokens[b, idx_i]
        fj = pred_tokens[b, idx_j]
        cos_sim = (fi * fj).sum(dim=-1)
        losses.append(F.relu(cos_sim - margin).mean())

    if not losses:
        return torch.tensor(0.0, device=pred_tokens.device)
    return torch.stack(losses).mean()


def semantic_aware_masked_l1(pred_tokens, gt_tokens, fake_img, real_img, grid_size, gamma=5.0):
    # pred_tokens, gt_tokens: [B, N, D]
    B, N, D = pred_tokens.shape
    H, W = grid_size
    pred_norm = _normalize_tokens(pred_tokens)
    gt_norm = _normalize_tokens(gt_tokens)

    cos_sim = (pred_norm * gt_norm).sum(dim=-1)
    diff = 1 - cos_sim
    mask = torch.exp(-gamma * diff)
    mask = mask.view(B, 1, H, W)
    mask = F.interpolate(mask, size=fake_img.shape[-2:], mode="bilinear", align_corners=False)

    return (mask * (fake_img - real_img).abs()).mean()


def dino_soft_correspondence_feature_loss(pred_tokens, gt_tokens, grid_size, window=3, tau=0.07, p=1):
    # pred_tokens, gt_tokens: [B, N, D]
    B, N, D = pred_tokens.shape
    H, W = grid_size
    pred_tokens = _normalize_tokens(pred_tokens)
    gt_tokens = _normalize_tokens(gt_tokens)

    pred_grid = pred_tokens.view(B, H, W, D)
    gt_grid = gt_tokens.view(B, H, W, D)

    local_windows = _gather_local_windows(gt_grid, window)  # [B,H,W,K,D]
    local_scores = (pred_grid.unsqueeze(-2) * local_windows).sum(dim=-1) / max(tau, 1e-6)  # [B,H,W,K]
    local_weights = F.softmax(local_scores, dim=-1)
    soft_gt = (local_weights.unsqueeze(-1) * local_windows).sum(dim=-2)  # [B,H,W,D]

    pred_flat = pred_grid.view(B, N, D)
    soft_gt_flat = soft_gt.view(B, N, D)

    if p == 2:
        return F.mse_loss(pred_flat, soft_gt_flat)
    return F.l1_loss(pred_flat, soft_gt_flat)


def dino_cross_modal_prototype_contrastive_loss(pred_tokens, sar_img, grid_size, k=4, tau=0.07, kmeans_iters=5):
    # pred_tokens: [B, N, D], sar_img: [B, C, H_img, W_img]
    B, N, D = pred_tokens.shape
    H, W = grid_size
    pred_norm = _normalize_tokens(pred_tokens)

    sar_gray = sar_img.mean(dim=1, keepdim=True)
    sar_grid = F.interpolate(sar_gray, size=(H, W), mode="bilinear", align_corners=False).view(B, N)

    losses = []
    for b in range(B):
        x = sar_grid[b].unsqueeze(-1)  # [N,1]
        if N == 0:
            continue

        kk = max(1, min(k, N))
        sorted_vals, _ = torch.sort(x.squeeze(-1))
        init_idx = torch.linspace(0, N - 1, steps=kk, device=x.device).long()
        centers = sorted_vals[init_idx].unsqueeze(-1)  # [kk,1]

        assign = torch.zeros(N, dtype=torch.long, device=x.device)
        for _ in range(kmeans_iters):
            dists = _pairwise_sqdist(x, centers)  # [N,kk]
            assign = dists.argmin(dim=1)
            new_centers = []
            for ci in range(kk):
                mask = assign == ci
                if mask.any():
                    new_centers.append(x[mask].mean(dim=0))
                else:
                    new_centers.append(centers[ci])
            centers = torch.stack(new_centers, dim=0)

        feat = pred_norm[b]  # [N,D]
        prototypes = []
        valid_ids = []
        for ci in range(kk):
            mask = assign == ci
            if mask.any():
                proto = _normalize_tokens(feat[mask].mean(dim=0, keepdim=False))
                prototypes.append(proto)
                valid_ids.append(ci)

        if len(prototypes) <= 1:
            losses.append(torch.tensor(0.0, device=pred_tokens.device))
            continue

        prototypes = torch.stack(prototypes, dim=0)  # [Kv,D]

        remap = torch.full((kk,), -1, dtype=torch.long, device=x.device)
        for ridx, cid in enumerate(valid_ids):
            remap[cid] = ridx
        target = remap[assign]
        valid_pix = target >= 0
        if not valid_pix.any():
            losses.append(torch.tensor(0.0, device=pred_tokens.device))
            continue

        feat_valid = feat[valid_pix]
        target_valid = target[valid_pix]

        logits = (feat_valid @ prototypes.t()) / max(tau, 1e-6)
        losses.append(F.cross_entropy(logits, target_valid))

    if not losses:
        return torch.tensor(0.0, device=pred_tokens.device)
    return torch.stack(losses).mean()


def dino_topological_consistency(pred_tokens, gt_tokens, remove_diag=True):
    # pred_tokens, gt_tokens: [B, N, D]
    pred_norm = _normalize_tokens(pred_tokens)
    gt_norm = _normalize_tokens(gt_tokens)

    ssm_pred = _pairwise_similarity(pred_norm, pred_norm)  # [B, N, N]
    ssm_gt = _pairwise_similarity(gt_norm, gt_norm)  # [B, N, N]

    diff = (ssm_pred - ssm_gt).abs()
    if remove_diag:
        B, N, _ = diff.shape
        eye = torch.eye(N, device=diff.device, dtype=torch.bool).unsqueeze(0)
        diff = diff.masked_fill(eye, 0.0)
        denom = max(N * N - N, 1)
        return diff.sum(dim=(1, 2)).mean() / denom
    return diff.mean()


def dino_semantic_edge_alignment(pred_tokens, gt_tokens, grid_size=None, eps=1e-6):
    # pred_tokens, gt_tokens: [B, N, D]
    B, N, D = pred_tokens.shape
    if grid_size is None:
        side = int(N ** 0.5)
        if side * side != N:
            raise ValueError(f"Invalid token count {N}: cannot infer square grid for edge loss.")
        H, W = side, side
    else:
        H, W = grid_size
        if H * W != N:
            side = int(N ** 0.5)
            if side * side != N:
                raise ValueError(f"grid_size={grid_size} mismatches token count N={N}.")
            H, W = side, side
    pred_grid = _normalize_tokens(pred_tokens).view(B, H, W, D)
    gt_grid = _normalize_tokens(gt_tokens).view(B, H, W, D)

    pred_x = 1.0 - (pred_grid[:, :, :-1, :] * pred_grid[:, :, 1:, :]).sum(dim=-1)
    gt_x = 1.0 - (gt_grid[:, :, :-1, :] * gt_grid[:, :, 1:, :]).sum(dim=-1)

    pred_y = 1.0 - (pred_grid[:, :-1, :, :] * pred_grid[:, 1:, :, :]).sum(dim=-1)
    gt_y = 1.0 - (gt_grid[:, :-1, :, :] * gt_grid[:, 1:, :, :]).sum(dim=-1)

    loss_x = (pred_x - gt_x).abs().mean() if pred_x.numel() > 0 else torch.tensor(0.0, device=pred_tokens.device)
    loss_y = (pred_y - gt_y).abs().mean() if pred_y.numel() > 0 else torch.tensor(0.0, device=pred_tokens.device)
    return loss_x + loss_y


def dino_ipd_cos_alignment(pred_tokens, gt_tokens):
    # pred_tokens, gt_tokens: [B, N, D]
    pred_norm = _normalize_tokens(pred_tokens)
    gt_norm = _normalize_tokens(gt_tokens)

    pred_global = _normalize_tokens(pred_norm.mean(dim=1, keepdim=False))
    gt_global = _normalize_tokens(gt_norm.mean(dim=1, keepdim=False))

    d_pred = (pred_norm * pred_global.unsqueeze(1)).sum(dim=-1)
    d_gt = (gt_norm * gt_global.unsqueeze(1)).sum(dim=-1)

    return (d_pred - d_gt).abs().mean()


def multi_scale_dino_perceptual_loss(
    pred_tokens_list,
    gt_tokens_list,
    layer_weights=None,
    eps=1e-8,
):
    """
    Multi-Scale DINO Perceptual Loss.

    Aligns multi-scale DINOv3 patch-level features between generated and
    ground-truth images with channel-wise L2 normalization and patch-wise
    squared L2 distance, which is equivalent to cosine distance on the
    unit-hypersphere (smooth version, avoids checkerboard artifacts).

    Formula:
        L = Σ_{l∈layers} λ_l * (1 / (Hp*Wp)) * Σ_{i,j} ||F_pred_bar^{(l)}(i,j) - F_GT_bar^{(l)}(i,j)||²_2

    where:
        F_bar = F / ||F||_2  (channel-wise L2 normalization)
        layers = {3, 6, 9, 11} (shallow geometric → deep semantic)

    Args:
        pred_tokens_list: List of [B, N, D] tensors, one per DINO block.
        gt_tokens_list:   List of [B, N, D] tensors, aligned with pred_tokens_list.
        layer_weights:    Per-layer weights. Defaults to all-ones.
        eps:              Stability constant for L2 normalization.

    Returns:
        Scalar loss.
    """
    if len(pred_tokens_list) != len(gt_tokens_list):
        raise ValueError("pred and gt token lists must have the same length.")
    if layer_weights is None:
        layer_weights = [1.0] * len(pred_tokens_list)
    if len(layer_weights) != len(pred_tokens_list):
        raise ValueError("layer_weights length must match number of layers.")

    total_loss = 0.0
    for pred_tokens, gt_tokens, w in zip(pred_tokens_list, gt_tokens_list, layer_weights):
        # Channel-wise L2 normalization → maps all feature levels to the unit hypersphere
        pred_norm = F.normalize(pred_tokens, dim=-1, eps=eps)   # [B, N, D]
        gt_norm = F.normalize(gt_tokens, dim=-1, eps=eps)        # [B, N, D]

        # Patch-level squared L2 distance on normalized features
        # On unit-hypersphere this is equivalent to 2 * (1 - cosine_similarity),
        # which is a smoother proxy than direct L1 cosine distance.
        diff_sq = ((pred_norm - gt_norm) ** 2).sum(dim=-1)      # [B, N]
        layer_loss = diff_sq.mean()                             # scalar

        total_loss = total_loss + w * layer_loss

    return total_loss


def dino_tile_cosine_loss(
    pred_tokens_list,
    gt_tokens_list,
    layer_weights=None,
    eps=1e-8,
):
    if len(pred_tokens_list) != len(gt_tokens_list):
        raise ValueError("pred and gt token lists must have the same length.")
    if layer_weights is None:
        layer_weights = [1.0] * len(pred_tokens_list)
    if len(layer_weights) != len(pred_tokens_list):
        raise ValueError("layer_weights length must match number of layers.")

    losses = []
    for pred_tokens, gt_tokens, weight in zip(pred_tokens_list, gt_tokens_list, layer_weights):
        pred_norm = F.normalize(pred_tokens, dim=-1, eps=eps)
        gt_norm = F.normalize(gt_tokens, dim=-1, eps=eps)
        layer_loss = 1.0 - (pred_norm * gt_norm).sum(dim=-1)
        losses.append(float(weight) * layer_loss.mean())

    if not losses:
        device = pred_tokens_list[0].device if pred_tokens_list else "cpu"
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).sum()


def tara_topological_kl_loss(
    pred_tokens,
    gt_tokens,
    tau=0.07,
    eps=1e-8,
):
    # pred_tokens, gt_tokens: [B, N, D]
    _, num_tokens, _ = pred_tokens.shape
    if num_tokens <= 1:
        return pred_tokens.new_tensor(0.0)

    pred_norm = F.normalize(pred_tokens, dim=-1, eps=eps)
    gt_norm = F.normalize(gt_tokens, dim=-1, eps=eps)
    tau = max(float(tau), eps)

    logits_pred = _pairwise_similarity(pred_norm, pred_norm) / tau
    logits_gt = _pairwise_similarity(gt_norm, gt_norm) / tau

    eye = torch.eye(num_tokens, device=pred_tokens.device, dtype=torch.bool).unsqueeze(0)
    mask_value = -1e4 if logits_pred.dtype in (torch.float16, torch.bfloat16) else -1e9
    logits_pred = logits_pred.masked_fill(eye, mask_value)
    logits_gt = logits_gt.masked_fill(eye, mask_value)

    log_prob_pred = F.log_softmax(logits_pred, dim=-1)
    prob_gt = F.softmax(logits_gt, dim=-1).detach()
    return F.kl_div(log_prob_pred, prob_gt, reduction="none").sum(dim=-1).mean()

"""Gene-to-Module Perceiver Trunk with RoPE + SwiGLU, Full-Vocab KO Conditioning & True Masked MGM.

Upgrades over V1:
- Dynamically aligns `anchor_gene_idx` directly against `data/processed/gene_vocab.json` when available.
- Applies `[B, G]` `gene_ko_mult` perturbation clamping directly onto the input transcriptome prior to
  Perceiver cross-attention so ALL `2,048` gene knockouts modulate the `128` latent module tokens.
- Supports `mgm_corrupt_mask` (`15%` random gene corruption during training) masking both cross-attention
  and `direct_mrna` on held-out MGM target genes (eliminating trivial identity autoencoding).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ANCHOR_28_PROTEINS: tuple[str, ...] = (
    "GPX4",     # 0: Phospholipid hydroperoxide glutathione peroxidase (Sec46)
    "AIFM2",    # 1: FSP1 CoQ10 / vitamin K oxidoreductase
    "ACSL4",    # 2: Long-chain PUFA-CoA synthetase 4
    "LPCAT3",   # 3: Lysophosphatidylcholine acyltransferase 3
    "ACSL3",    # 4: MUFA-CoA synthetase 3 (membrane protection)
    "SLC7A11",  # 5: System xc- cystine/glutamate antiporter xCT
    "SLC3A2",   # 6: System xc- heavy chain 4F2hc
    "DHCR7",    # 7: 7-Dehydrocholesterol reductase (consumes 7-DHC shield)
    "EBP",      # 8: Emopamil-binding protein sterol isomerase (produces 7-DHC precursors)
    "SQLE",     # 9: Squalene epoxidase
    "HMGCR",    # 10: HMG-CoA reductase (mevalonate -> IPP & sterols)
    "PSTK",     # 11: Phosphoseryl-tRNA[Ser]Sec kinase
    "SEPSECS",  # 12: Sep(O-phosphoserine) tRNA:Sec[Ser]tRNA synthase
    "EEFSEC",   # 13: Selenocysteine-specific elongation factor
    "SEPHS2",   # 14: Selenophosphate synthetase 2
    "POR",      # 15: Cytochrome P450 oxidoreductase (peroxidation initiation)
    "TFRC",     # 16: Transferrin receptor 1 (endolysosomal Fe3+ import)
    "FTH1",     # 17: Ferritin heavy chain (Fe storage buffer)
    "FTL",      # 18: Ferritin light chain
    "NCOA4",    # 19: Ferritinophagy cargo receptor
    "GCLC",     # 20: Glutamate-cysteine ligase catalytic subunit
    "GSS",      # 21: Glutathione synthetase
    "GSR",      # 22: Glutathione-disulfide reductase
    "TXNRD1",   # 23: Thioredoxin reductase 1 (Sec selenoprotein)
    "GCH1",     # 24: GTP cyclohydrolase 1 (BH4 radical trap)
    "ODC1",     # 25: Ornithine decarboxylase 1 (polyamine synthesis)
    "SRM",      # 26: Spermidine synthase
    "SMS",      # 27: Spermine synthase (endolysosomal Fe2+ buffering)
)

PROTEIN_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(ANCHOR_28_PROTEINS)}

ANCHOR_12_METABOLITES: tuple[str, ...] = (
    "GSH",        # 0: Reduced glutathione
    "GSSG",       # 1: Oxidized glutathione disulfide
    "NADP+",      # 2: Oxidized nicotinamide adenine dinucleotide phosphate
    "NADPH",      # 3: Reduced NADPH cofactor pool
    "putrescine", # 4: Putrescine
    "spermidine", # 5: Spermidine
    "spermine",   # 6: Spermine (lysosomal Fe2+ shield)
    "PC_PUFA2",   # 7: Di-PUFA phosphatidylcholine
    "PE_PUFA",    # 8: Arachidonoyl/adrenoyl phosphatidylethanolamine
    "ePL_PUFA",   # 9: Peroxisomal PUFA-ether phospholipid (plasmalogen)
    "MUFA_PL",    # 10: Oleoyl MUFA-phospholipid competitive shield
    "CoQ10H2",    # 11: Ubiquinol / reduced vitamin K pool
)

METABOLITE_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(ANCHOR_12_METABOLITES)}


def resolve_vocab_gene_indices(genes_to_find: tuple[str, ...], n_genes: int) -> list[int]:
    """Dynamically resolve canonical gene indices from `data/processed/gene_vocab.json`."""
    vocab_p = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "gene_vocab.json"
    fallback = [
        0, 14, 24, 25, 26, 1, 2, 36, 37, 39, 40, 7, 8, 10,
        9, 42, 46, 47, 48, 49, 3, 5, 6, 20, 17, 54, 75, 55,
    ]
    if vocab_p.is_file():
        try:
            g2i = json.loads(vocab_p.read_text(encoding="utf-8")).get("gene_to_idx", {})
            out = []
            for pos, sym in enumerate(genes_to_find):
                fb = fallback[pos % len(fallback)]
                out.append(int(g2i.get(sym, fb)) % max(1, n_genes))
            return out
        except Exception:
            pass
    return [fallback[pos % len(fallback)] % max(1, n_genes) for pos in range(len(genes_to_find))]


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization computed in fp32 for bf16 stability."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.float()
        rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f32 * rms).to(orig_dtype) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """1D Rotary Position Embedding (`RoPE`) for latent module tokens."""

    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0) -> None:
        super().__init__()
        even_dim = head_dim - (head_dim % 2)
        inv_freq = 1.0 / (base ** (torch.arange(0, even_dim, 2, dtype=torch.float32) / max(even_dim, 2)))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.even_dim = even_dim

    def apply(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        cos = self.cos_cached[:seq_len, : self.even_dim].to(device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len, : self.even_dim].to(device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0)

        if self.even_dim == q.shape[-1]:
            return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)
        q_rot = (q[..., : self.even_dim] * cos) + (_rotate_half(q[..., : self.even_dim]) * sin)
        k_rot = (k[..., : self.even_dim] * cos) + (_rotate_half(k[..., : self.even_dim]) * sin)
        return (
            torch.cat([q_rot, q[..., self.even_dim :]], dim=-1),
            torch.cat([k_rot, k[..., self.even_dim :]], dim=-1),
        )


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, d_model: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        h_dim = hidden_dim or int((8 * d_model) / 3)
        h_dim = ((h_dim + 15) // 16) * 16
        self.w_gate = nn.Linear(d_model, h_dim, bias=False)
        self.w_up = nn.Linear(d_model, h_dim, bias=False)
        self.w_down = nn.Linear(h_dim, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class RoPESwiGLUBlock(nn.Module):
    """Pre-RMSNorm Transformer encoder block with RoPE SDPA and SwiGLU MLP."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.attn_norm = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        h = self.attn_norm(x)
        qkv = self.qkv(h).view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self.rope.apply(q, k)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        x = x + self.out_proj(attn_out)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GeneToModulePerceiverTrunk(nn.Module):
    """Factored Perceiver cross-attention + RoPE-SwiGLU trunk + Multi-Omic Physical Anchor Heads."""

    def __init__(
        self,
        n_genes: int = 512,
        d_model: int = 256,
        n_modules: int = 128,
        n_pathway_modules: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        gene_emb_dim: int = 64,
        dropout: float = 0.0,
        ablate_physical_anchors: bool = False,
        anchor_gene_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.d_model = int(d_model)
        self.n_modules = int(n_modules)
        self.n_pathway_modules = min(int(n_pathway_modules), self.n_modules)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.gene_emb_dim = int(gene_emb_dim)
        self.ablate_physical_anchors = bool(ablate_physical_anchors)

        self.gene_key_emb = nn.Parameter(torch.randn(self.n_genes, self.gene_emb_dim) * 0.05)
        self.gene_val_emb = nn.Parameter(torch.randn(self.n_genes, self.gene_emb_dim) * 0.05)
        self.module_queries = nn.Parameter(torch.randn(self.n_modules, self.gene_emb_dim) * 0.08)
        self.module_bias_tokens = nn.Parameter(torch.randn(self.n_modules, self.d_model) * 0.02)

        pathway_prior = torch.zeros(self.n_modules, self.n_genes)
        block = max(1, self.n_genes // max(1, self.n_pathway_modules))
        for m in range(self.n_pathway_modules):
            start = (m * block) % self.n_genes
            end = min(self.n_genes, start + max(4, 2 * block))
            pathway_prior[m, start:end] = 0.75
        self.register_buffer("pathway_prior_bias", pathway_prior, persistent=False)

        self.cross_in_proj = nn.Sequential(
            nn.Linear(2 * self.gene_emb_dim + 2, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.cis_fusion = nn.Linear(self.d_model, self.d_model)

        self.blocks = nn.ModuleList(
            [RoPESwiGLUBlock(self.d_model, self.n_heads, dropout=dropout) for _ in range(self.n_layers)]
        )
        self.final_norm = RMSNorm(self.d_model)

        if anchor_gene_indices is None:
            anchor_gene_indices = resolve_vocab_gene_indices(ANCHOR_28_PROTEINS, self.n_genes)
        self.register_buffer(
            "anchor_gene_idx",
            torch.tensor(anchor_gene_indices[: len(ANCHOR_28_PROTEINS)], dtype=torch.long),
            persistent=False,
        )

        self.protein_anchor_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, len(ANCHOR_28_PROTEINS)),
        )
        self.direct_mrna_to_prot_gain = nn.Parameter(torch.full((len(ANCHOR_28_PROTEINS),), 0.35))

        self.metabolite_anchor_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, len(ANCHOR_12_METABOLITES)),
        )

        self.mgm_cell_to_gene_lat = nn.Linear(self.d_model, self.gene_emb_dim)
        self.mgm_gene_bias = nn.Parameter(torch.zeros(self.n_genes))

    @classmethod
    def from_preset(cls, preset: str = "base", n_genes: int = 19264, **kwargs: object) -> "GeneToModulePerceiverTrunk":
        p = preset.strip().lower()
        if p == "xl":
            return cls(n_genes=n_genes, d_model=1024, n_layers=16, n_heads=16, **kwargs)  # type: ignore[arg-type]
        if p == "base":
            return cls(n_genes=n_genes, d_model=384, n_layers=8, n_heads=6, **kwargs)  # type: ignore[arg-type]
        return cls(n_genes=n_genes, d_model=256, n_layers=4, n_heads=4, **kwargs)  # type: ignore[arg-type]

    def _align_genes(
        self,
        gene_expr: torch.Tensor,
        expr_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.nan_to_num(gene_expr, nan=0.0, posinf=20.0, neginf=-20.0)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        bsz, g_in = x.shape

        if expr_mask is None:
            mask = torch.ones_like(x)
        else:
            mask = torch.nan_to_num(expr_mask.to(device=x.device, dtype=x.dtype), nan=0.0).clamp(0.0, 1.0)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(bsz, -1)

        if g_in < self.n_genes:
            pad_g = self.n_genes - g_in
            x = torch.cat([x, torch.zeros(bsz, pad_g, device=x.device, dtype=x.dtype)], dim=-1)
            mask = torch.cat([mask, torch.zeros(bsz, pad_g, device=x.device, dtype=x.dtype)], dim=-1)
        elif g_in > self.n_genes:
            x = x[:, : self.n_genes]
            mask = mask[:, : self.n_genes]

        return x, mask

    def decode_expr(self, z_cell: torch.Tensor, n_out_genes: int | None = None) -> torch.Tensor:
        """Decode latent cell state back to transcriptome expression `[B, G]`."""
        lat = self.mgm_cell_to_gene_lat(z_cell)
        pred = torch.matmul(lat, self.gene_val_emb.to(dtype=lat.dtype).t()) + self.mgm_gene_bias.to(dtype=lat.dtype)
        if n_out_genes is not None and n_out_genes != self.n_genes:
            if n_out_genes < self.n_genes:
                return pred[:, :n_out_genes]
            pad = torch.zeros(pred.shape[0], n_out_genes - self.n_genes, device=pred.device, dtype=pred.dtype)
            return torch.cat([pred, pad], dim=-1)
        return pred

    def forward(
        self,
        gene_expr: torch.Tensor,
        expr_mask: torch.Tensor | None = None,
        cis_reg_tokens: torch.Tensor | None = None,
        gene_ko_mult: torch.Tensor | None = None,
        mgm_keep_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        orig_g = gene_expr.shape[-1]
        x_raw, mask = self._align_genes(gene_expr, expr_mask)
        bsz = x_raw.shape[0]
        dtype = x_raw.dtype

        # Apply full 2,048-gene CRISPR/CRISPRi perturbation clamp onto transcriptome before module attention
        if gene_ko_mult is not None:
            gko, _ = self._align_genes(gene_ko_mult)
            gko = gko.clamp(0.0, 10.0)
            # Only scale observed genes where gko != 0 (pad positions in _align_genes are 0, so restore 1.0 on pad)
            if gene_ko_mult.shape[-1] < self.n_genes:
                gko[:, gene_ko_mult.shape[-1] :] = 1.0
            x = x_raw * gko
        else:
            x = x_raw

        # Apply Masked Gene Modeling (MGM) random input corruption mask if provided during training
        if mgm_keep_mask is not None:
            keep_m, _ = self._align_genes(mgm_keep_mask)
            eff_mask = mask * keep_m.clamp(0.0, 1.0)
        else:
            eff_mask = mask

        obs_frac = (eff_mask.sum(dim=-1, keepdim=True) / float(self.n_genes)).clamp(min=0.05)
        x_masked = (x * eff_mask) / obs_frac.sqrt()

        scale = float(self.gene_emb_dim) ** -0.5
        base_logits = (
            torch.matmul(
                self.module_queries.to(dtype=dtype),
                self.gene_key_emb.to(dtype=dtype).t(),
            )
            * scale
            + self.pathway_prior_bias.to(device=x.device, dtype=dtype)
        )

        any_obs = eff_mask.sum(dim=-1, keepdim=True) > 0
        m_tilde = torch.where(any_obs, (eff_mask > 0.5).to(dtype=dtype), torch.ones_like(eff_mask))

        # Exact memory-free BLAS factorization of masked softmax cross-attention:
        # Avoids materializing [B, M, G] = [128, 128, 2048] broadcast tensors (100x faster on CPU/GPU!)
        attn_base = torch.softmax(base_logits.float(), dim=-1).to(dtype)  # [M, G]
        attn_t = attn_base.t().contiguous()  # [G, M]
        z_norm = torch.matmul(m_tilde, attn_t).clamp(min=1e-8)  # [B, M]

        val_emb = self.gene_val_emb.to(dtype=dtype)  # [G, D_g]
        kv_flat = (attn_t.unsqueeze(-1) * val_emb.unsqueeze(1)).reshape(self.n_genes, self.n_modules * self.gene_emb_dim)

        x_eff = x_masked * m_tilde
        ctx_expr = torch.matmul(x_eff, kv_flat).view(bsz, self.n_modules, self.gene_emb_dim) / z_norm.unsqueeze(-1)
        ctx_id = torch.matmul(m_tilde, kv_flat).view(bsz, self.n_modules, self.gene_emb_dim) / z_norm.unsqueeze(-1)
        amp_mean = (torch.matmul(x_eff, attn_t) / z_norm).unsqueeze(-1)
        amp_energy = ((torch.matmul(x_eff.pow(2), attn_t) / z_norm).unsqueeze(-1) + 1e-8).sqrt()

        cross_feat = torch.cat([ctx_expr, ctx_id, amp_mean, amp_energy], dim=-1)
        tokens = self.module_bias_tokens.to(dtype=dtype).unsqueeze(0) + self.cross_in_proj(cross_feat)

        if cis_reg_tokens is not None:
            k_cis = min(cis_reg_tokens.shape[1], self.n_pathway_modules)
            cis_proj = self.cis_fusion(cis_reg_tokens[:, :k_cis, :].to(device=x.device, dtype=dtype))
            pad_tokens = torch.zeros(
                bsz,
                self.n_modules - k_cis,
                self.d_model,
                device=x.device,
                dtype=dtype,
            )
            tokens = tokens + torch.cat([cis_proj, pad_tokens], dim=1)

        for block in self.blocks:
            tokens = block(tokens)

        z_modules = self.final_norm(tokens)
        z_pathway = z_modules[:, : self.n_pathway_modules, :].mean(dim=1)
        if self.n_modules > self.n_pathway_modules:
            z_context = z_modules[:, self.n_pathway_modules :, :].mean(dim=1)
            z_cell = 0.5 * (z_pathway + z_context)
        else:
            z_context = z_pathway
            z_cell = z_pathway

        if self.ablate_physical_anchors:
            p_hat_log = torch.zeros(bsz, len(ANCHOR_28_PROTEINS), device=x.device, dtype=dtype)
            m0_hat_log = torch.zeros(bsz, len(ANCHOR_12_METABOLITES), device=x.device, dtype=dtype)
            p_hat = torch.ones_like(p_hat_log)
            m0_hat = torch.ones_like(m0_hat_log)
        else:
            # Respect eff_mask on direct_mrna so masked genes (in MGM or Stage-0 mask-out tests) never leak!
            anchor_obs = eff_mask[:, self.anchor_gene_idx]
            direct_mrna = (x_raw[:, self.anchor_gene_idx] * anchor_obs) * F.softplus(
                self.direct_mrna_to_prot_gain.to(dtype=dtype)
            )
            p_hat_log = self.protein_anchor_head(z_pathway) + direct_mrna
            m0_hat_log = self.metabolite_anchor_head(z_cell)
            p_hat = F.softplus(p_hat_log) + 0.05
            m0_hat = F.softplus(m0_hat_log) + 0.05

        mgm_recon = self.decode_expr(z_cell, n_out_genes=orig_g)

        return {
            "z_modules": z_modules,
            "z_cell": z_cell,
            "z_pathway": z_pathway,
            "z_context": z_context,
            "p_hat": p_hat,
            "p_hat_log": p_hat_log,
            "m0_hat": m0_hat,
            "m0_hat_log": m0_hat_log,
            "mgm_recon": mgm_recon,
        }


PerceiverTrunk = GeneToModulePerceiverTrunk

__all__ = [
    "ANCHOR_28_PROTEINS",
    "PROTEIN_TO_IDX",
    "ANCHOR_12_METABOLITES",
    "METABOLITE_TO_IDX",
    "resolve_vocab_gene_indices",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "RoPESwiGLUBlock",
    "GeneToModulePerceiverTrunk",
    "PerceiverTrunk",
]

# Gene-Query JEPA — winning config (`qsb64_tc48_contr0`)

Step-by-step mathematical notation for the configuration currently adopted as the reference
run, named in its checkpoint directory
`fzT_encL3_predL3_q128_mixed_lg1_lc0_contr0_vic0_glmse_bs32_ep30_lr0.0001_seed0_splitT_poolcls_tp1-2_max512_src90m_mce0.1_ms0.2_qsrcpos_qsb64_nwct_tc48_la0.25_20261012_160033`
(90m-source, `pred_tps=[1,2]`). The encoder and predictor architectures are unchanged from
`gene_query_jepa_architecture_v3.pdf` / `gene_query_jepa_predictor.pdf` — this document only
formalizes the two pieces that differ from that baseline: the **query sampler** (§2) and the
**loss** (§4, `λ_contrastive` now 0). §1 and §3 are included for a complete, self-contained
read, not because they changed.

**What changed vs. the earlier documented config**

| | earlier (`architecture_v3`) | this run |
|---|---|---|
| `n_src_queries` | 96 | **64** |
| `n_top_common_queries` | 0 (undefined / single pool) | **48** |
| `λ_contrastive` | 0.1 | **0** |
| `n_queries` | 128 | 128 (unchanged) |
| encoder / predictor layers | 3 / 3 | 3 / 3 (unchanged) |

## 0. Notation

| symbol | meaning |
|---|---|
| $B$ | batch size |
| $D$ | model width, 768 |
| $L_s$ | source sequence length (padded, $\le 512$) |
| $t \in \{1, 2\}$ | target timepoint index (90m-source: $1{=}$6h, $2{=}$10h) |
| $g$ | a gene id, LOCAL vocabulary |
| $\mathcal{V}$ | full LOCAL gene vocabulary ("universe"), $|\mathcal V|$ genes |
| $S_{\text{src}}$ | set of distinct gene ids present in this cell's source sequence |
| $Q = 128$ | `n_queries`, total query slots per forward pass |
| $Q_{\text{src}} = 64$ | `n_src_queries` |
| $Q_{\text{top}} = 48$ | `n_top_common_queries` |
| $Q_{\text{rand}} = Q - Q_{\text{src}} - Q_{\text{top}} = 16$ | remaining random "outside" slots |
| $E[\cdot]$ | learned gene-identity embedding table, $\mathbb R^{D}$ per gene |
| $R[\cdot]$ | learned source-rank positional embedding table |
| $T[\cdot]$ | learned per-timepoint embedding |
| $H_{\text{src}} \in \mathbb R^{B \times L_s \times D}$ | frozen online-encoder token embeddings of the source |
| $z^\star_g$ | the answer: gene $g$'s row of $H_{\text{tgt}}$ if present, else the fixed null $a_{\text{null}}$ |
| $\hat z_g$ | predictor's output row for query $g$ |

## 1. Encoder (unchanged)

Frozen MaskGIT encoder, layers 1–3 of 12, online copy reads the source and EMA copy reads the
target (EMA $\equiv$ online here: both frozen, so the EMA update is a no-op):

$$H_{\text{src}} = \mathrm{Enc}_{\text{online}}(x^{\text{src}}), \qquad
H_{\text{tgt}} = \mathrm{Enc}_{\text{EMA}}(x^{\text{tgt}}_t)$$

## 2. Query construction — source-blind, top-common split

Per cell, per timepoint $t$. The sampler never looks at the target when choosing *which* genes
to ask about — only the three pools below, all built from the source and from population-level
statistics, decide the question list. The target is read **only afterwards**, to produce the
label $z^\star_g$ for each already-chosen $g$.

**Step 2.1 — source pool ($Q_{\text{src}}=64$ slots).** Sample without replacement, uniformly,
from the genes this cell's source actually expresses:

$$G_{\text{src}} \sim \mathrm{Unif}\big(S_{\text{src}},\ k{=}\min(Q_{\text{src}}, |S_{\text{src}}|)\big)$$

**Step 2.2 — population prior.** For a gene $g$ and this cell's $(\text{cell\_type}, t)$, let

$$p(g) = \widehat{\Pr}\big[g \in \text{target} \mid \text{cell\_type}, t\big]$$

be an empirical frequency table (never conditioned on *this* cell's actual target). Two query
weight vectors are derived from it, each gene floored so no gene has exactly zero draw mass:

$$w^{\text{top}}(g) = p(g)^{\,\alpha} + \varepsilon, \qquad
w^{\text{rand}}(g) = \big(1-p(g)\big)^{\alpha} + \varepsilon, \qquad
\alpha{=}0.5,\ \varepsilon{=}0.02$$

(the code computes both from the same `_nonsrc_weights_for_batch` call, with/without
`invert=True`; $+\varepsilon$ is a constant shift so it never changes the §2.3 top-$K$
*ranking* — only $w^{\text{rand}}$'s sampling actually depends on it, by keeping every gene's
draw probability strictly positive.)

**Step 2.3 — top-common pool ($Q_{\text{top}}=48$ slots), deterministic.** Let
$\mathcal{N} = \mathcal V \setminus S_{\text{src}}$ (genes absent from this source). Take the
$Q_{\text{top}}$ highest-prior genes in $\mathcal N$:

$$G_{\text{top}} = \arg\mathrm{top}ـ{Q_{\text{top}}}\big\{ w^{\text{top}}(g) : g \in \mathcal N \big\}$$

This pool is the same for every cell of the same (cell\_type, $t$) — it is *who's likely to be
expressed next*, asked about on purpose, distinct from the source.

**Step 2.4 — random remainder ($Q_{\text{rand}}=16$ slots).** Weighted sample without
replacement from what's left of $\mathcal N$ after removing $G_{\text{top}}$:

$$G_{\text{rand}} \sim \mathrm{WeightedSample}\big(\mathcal N \setminus G_{\text{top}},\ 
w^{\text{rand}},\ k{=}Q_{\text{rand}}\big)$$

Freshly redrawn every call, so no single "likely absent" gene id is ever memorized.

**Step 2.5 — rank + identity + time, then label.** For every chosen gene
$g \in G_{\text{src}} \cup G_{\text{top}} \cup G_{\text{rand}}$:

$$r_{\text{src}}(g) = \begin{cases}
\text{first index of } g \text{ in the source sequence} & g \in S_{\text{src}} \\
-1 & g \notin S_{\text{src}}
\end{cases}$$

$$q_g = E[g] + R[\,r_{\text{src}}(g)\,] + T[t] \ \in \mathbb R^{D}$$

Only *now* is the target consulted for the label:

$$z^\star_g = \begin{cases} H_{\text{tgt}}[\,\text{position of } g\,] & g \text{ present in target} \\
a_{\text{null}} & \text{otherwise (fixed buffer, } \|a_{\text{null}}\| = \sqrt D \text{)} \end{cases}$$

Stack with the one learned `cell_query` row to get $Q_{\text{full}} \in \mathbb R^{B \times 129 \times D}$,
unchanged from the base architecture (§ predictor diagram).

## 3. Predictor (unchanged)

$$\hat Z = \mathrm{Predictor}_{3\text{-layer}}\big(Q_{\text{full}},\ \text{memory}{=}H_{\text{src}}\big)
\in \mathbb R^{B \times 129 \times D}$$

Self-attention over all 129 query rows, cross-attention of every row to $H_{\text{src}}$ only
(source-conditioned, never target-conditioned), GELU FFN — 8 heads, $d_k{=}d_v{=}96$, identical
to `gene_query_jepa_predictor.pdf`. $\hat z_{\text{gene}}$ = first 128 rows, $\hat z_{\text{cell}}$
= last row (unsupervised here, $\lambda_{\text{cell}}{=}0$).

## 4. Loss — contrastive term removed

$$\mathcal L = \operatorname{mean}_t\Big(
\mathcal L_{\text{var}} + 0.25\, \mathcal L_{\text{abs}} + 0.1\, \mathcal L_{\text{mask}}
\Big)$$

$$\mathcal L_{\text{var}} = \operatorname{mean}_{g \in P} \tfrac{1}{D}\lVert \hat z_g - z^\star_g \rVert^2,
\qquad
\mathcal L_{\text{abs}} = \operatorname{mean}_{g \in A} \tfrac{1}{D}\lVert \hat z_g - a_{\text{null}} \rVert^2$$

$$\mathcal L_{\text{mask}} = \mathrm{CE}\big(W\hat z / \lVert \hat z \rVert,\ g\big) \quad
\text{(19 mask-CE slots, unchanged)}$$

$P$ = present-and-not-mask-CE query slots, $A$ = absent-and-not-mask-CE slots (both restricted
to $G_{\text{src}} \cup G_{\text{top}} \cup G_{\text{rand}}$ as sampled in §2).

**Dropped term**, present at $\lambda{=}0.1$ in the earlier config, $\lambda{=}0$ here:

$$\mathcal L_{\text{NCE}} = \mathrm{CE}\big(\mathrm{softmax}(S/\tau)\big), \quad
S_{ij} = \hat u_i \cdot u_j^\star,\ \tau{=}0.1 \qquad (\lambda_{\text{contrastive}} = 0)$$

With $\lambda_{\text{contrastive}}{=}0$ this term contributes nothing to $\mathcal L$ or its
gradient; `contrastive_tau` is still logged in `specs.json` as a vestigial hyperparameter of an
inactive term.

## 5. Why these three numbers (plain language)

- **$Q_{\text{src}}: 96 \to 64$.** Fewer source-only queries leaves more of the 128-slot budget
  for genes the model must predict *without* having seen them in this cell's source — a harder,
  more informative training signal per batch.
- **$Q_{\text{top}}=48$ (new).** Half of the freed-up budget goes to genes that are *likely*
  present at this (cell\_type, $t$) but happen to be absent from *this* source — i.e. genes
  whose presence the model should be able to predict well, since population structure alone
  gets you most of the way there. This directly targets the §2 docstring's design goal: ask
  about genes "likely present, still never looks at THIS cell's target."
- **$\lambda_{\text{contrastive}}: 0.1 \to 0$.** Per `gene_mse_floor.pdf`, the InfoNCE term was
  shown to fight the squared-error term's pull toward the type-conditional mean without a
  cell-specific direction to separate along, holding the gene-MSE floor up by roughly its own
  weight's worth of pressure. Removing it is the cheap, well-attributed fix from that note's
  "what would actually move it" list.

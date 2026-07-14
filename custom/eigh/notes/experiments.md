# eigh — experiment ledger

Results only (strategy/reasoning lives in `design.md`). The `benchmark-runner` agent
maintains this after every remote run.

Ranking metric: **geometric mean of runtime over the 13 benchmark cases on B200**
(lower is better). Only B200 popcorn numbers count — never record a local number.

Why three parts: the geomean is the *ranked scalar*, but the champion is built as a
**shape-dispatcher**, so we also track the best time **per case**. An attempt can
lose on geomean yet set a per-case record worth grafting into the dispatcher — the
per-case table below keeps that knowledge even though such an attempt is not a
`winners/` entry.

---

## 1. Champion (overall geomean → decides `winners/`)

| Metric | Value | Holder |
|---|---|---|
| Passes all correctness tests | yes | v0 (torch baseline, submission 875739) |
| Best overall benchmark geomean (µs) | 48774.0 | v0 (torch baseline, submission 875753) |

## 2. Per-case best-of  (updated on ANY cell beaten, regardless of geomean)

The 13 benchmark cases (from `task.yml`). Record best measured time + the attempt
that holds it. A new record here is valuable even if that attempt's overall geomean
regressed — it's a candidate to graft into the dispatcher.

| # | shape (batch×n) | case | best (µs) | holder `vN` |
|---|---|---|---|---|
| 0 | 20×32 | dense | 134 | v0 |
| 1 | 40×176 | dense | 5610 | v0 |
| 2 | 40×352 | dense | 12100 | v0 |
| 3 | 640×512 | dense | 167000 | v0 |
| 4 | 60×1024 | dense | 91800 | v0 |
| 5 | 8×2048 | dense | 128000 | v0 |
| 6 | 640×512 | mixed | 159000 | v0 |
| 7 | 60×1024 | mixed | 91100 | v0 |
| 8 | 640×512 | rankdef | 162000 | v0 |
| 9 | 640×512 | clustered | 134000 | v0 |
| 10 | 60×1024 | nearrank | 90600 | v0 |
| 11 | 640×512 | lapack_dense_even_spectrum | 198000 | v0 |
| 12 | 60×1024 | lapack_dense_geometric_spectrum | 87800 | v0 |

## 3. Runs log  (one row per attempt)

Record per-case times (at least the movers) + the computed geomean. `records` =
which of {champion, per-case cells} this attempt set. `in winners/?` = did overall
geomean improve.

| `vN` | parent | date | hypothesis / change | correctness | geomean (µs) | per-case notes | records set | in winners/? | verdict |
|------|--------|------|---------------------|-------------|-------------|----------------|-------------|--------------|---------|
| v0 | — | 2026-07-14 | torch.linalg.eigh baseline (seed ledger, submission 875739) | 39/39 | — (test only) | — | — | no | baseline — run benchmark next for geomean |
| v0 | — | 2026-07-14 | torch.linalg.eigh baseline benchmark (submission 875753, notes/runs/v0_benchmark.json) | passed | 48774.0 | per-case (µs): #0 134, #1 5610, #2 12100, #3 167000, #4 91800, #5 128000, #6 159000, #7 91100, #8 162000, #9 134000, #10 90600, #11 198000, #12 87800 | champion + all 13 per-case cells (seeds baseline) | no | baseline geomean seeded; beat everywhere going forward |
| v3 | v0 | 2026-07-14 | Householder→tridiag→**QR** pipeline; inner solve deferred to torch eigh(T_dense) (submission 875874, notes/runs/v3_benchmark.json) | passed | 107331.6 | Triton path engages only n≥1024 (cases 4,5,7,10,12); all n≤512 cases = torch fallback (≈v0). Movers (µs): #4 619000, #5 1173000, #7 653000, #10 640000, #12 608000 — all ~6× slower than v0 | none (v0 holds every cell) | no | **dead end as-is**: deferred eigh(T) re-tridiagonalizes → strictly more work than v0. Value = validated Triton reduction/backtransform infra |
| v4 | v0 | 2026-07-14 | Householder→tridiag→**Cuppen D&C** pipeline; genuine D&C inner solve in torch (submission 875875, notes/runs/v4_benchmark.json) | passed | 90295.8 | Triton path engages only n≥1024; all n≤512 = torch fallback (≈v0). Movers (µs): #4 439000, #5 729000, #7 428000, #10 427000, #12 427000 — ~4.5× slower than v0 but **clearly beats v3** (n=2048: 729k vs 1173k, −38%). #9 133000 vs v0 134000 = noise on identical torch-fallback path, NOT a real record | none (v0 holds every cell; #9 tie within noise) | no | best of the three Triton attempts; **D&C > QR confirmed** for large-n all-vectors. Still loses to torch — the redundant reduction is the cost to remove next |
| v2 | v1 | 2026-07-14 | Two-level **block-Jacobi** (Brent–Luk block ordering; reuses v1 in-SRAM Jacobi as the local block-pair EVD) for n=176/352/512 (submission 875873, only notes/runs/v2_benchmark.stderr.log — no JSON) | not measured (killed) | — (killed) | **cancelled at ~360 s** while still `pending` on the n=512×640 case; deleted via `popcorn submissions delete 875873`. No completed per-case numbers | none | no | **too slow at n=512**: matmul-form Jacobi is O(sweeps·n⁴); machinery is correct but n⁴ dominates. May still be competitive at n=176/352 — untested (needs a single-index benchmark) |
| v5 | v3 | 2026-07-14 | Improved-coding v3: **native implicit-shift QL/QR inner solve consuming `(d,e)` directly** (drops v3's redundant `eigh(T_dense)`), leaner dlatrd, fused ormqr backtransform, TF32 (submission 875965, notes/runs/v5_benchmark.json) | passed | 138129.3 | Triton path engages only n≥1024; all n≤512 = torch fallback (≈v0; sub-ms diffs = noise). Movers (µs): #4 1130000, #5 3500000, #7 1140000, #10 1013000, #12 1187000 — all ~1.9–3× **worse than v3** | none (v0 holds every cell; #3 166k vs 167k = torch-fallback noise per v4 #9 precedent) | no | **regression vs parent v3**: an on-GPU sequential/divergent QR inner solve is far slower than v3's deferral to torch LAPACK. Removing the redundant retridiag did not pay off — native QR is the wrong inner solver. Dead end |
| v6 | v4 | 2026-07-14 | Improved-coding v4: **blocked WY Q1/backtransform** replacing the unblocked `sorgtr` loop, TF32, autotuned GEMM, cheap fp32 probe self-check (submission 875976, notes/runs/v6_benchmark.json) | passed | 108438.6 | Triton path engages only n≥1024; all n≤512 = torch fallback (noise). Movers (µs): #4 631000, #5 1113000, #7 627000, #10 623000, #12 620000 — ~1.4–1.5× **worse than v4**, but **still beats sibling v5** (n=2048: 1113k vs 3500k) | none (v0 holds every cell; sub-ms n≤512 diffs = torch-fallback noise) | no | **regression vs parent v4**; blocked-WY reduction did not recover v4's deferred-D&C speed. Confirms **D&C > QR** again (v6 < v5). Still loses to torch |

<!--
Conventions:
- vN matches attempts/submission_vN.py (and winners/ if it won on geomean).
- correctness: "39/39" or "FAIL: <case> <which gate>".
- geomean: computed from the 13 per-case means in notes/runs/v<N>_benchmark.json
  (the saved --output artifact); "-" for a test-only run.
- records set: e.g. "case #5 (8×2048)" and/or "champion".
- in winners/?: yes only if overall geomean beat the previous champion.
- On discard, also note the reason in design.md "Dead ends".
-->

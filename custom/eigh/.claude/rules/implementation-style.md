# Implementation style

## Single self-contained file

The platform submits **one file** as `submission.py` (see `task.yml`: `@SUBMISSION@`)
and imports `custom_kernel` from it. It does **not** upload other local modules.
Therefore the working **`submission.py`** — and every archived copy in
`attempts/submission_vN.py` (and its `winners/` copy if it wins) — **must be
self-contained**: all Triton kernels and helpers inline in the same file. A
`kernels/` directory may hold drafts/snippets, but anything used must be **inlined**
before a run.

You iterate in place on `submission.py`; each idea is archived to `attempts/`, and
overall-geomean winners are additionally copied to `winners/` (see `CLAUDE.md`).

## Size cap: ≤ 3000 lines

The implementation file must stay **at or under 3000 lines**. This is deliberate:
the kernel should remain **feasible and usable**, not an unmaintainable monster. If
an approach can't fit, it's the wrong approach — simplify or split the *idea*, not
the intent of this rule.

## Comment heavily — up to ~⅓ of lines

Dense comments are wanted and encouraged; **up to about one third of the lines being
comments is fine and welcome.** Comments should explain three things:

1. **The code** — what a kernel/block does and why it's written this way.
2. **The design process** — why this approach/variant was chosen, what alternative
   was rejected and why (link to `notes/design.md` reasoning where useful).
3. **Results/observations** — relevant measured behavior (e.g. "sweep cap 12 was
   the knee for n=512; more sweeps didn't move the residual") so the file itself
   documents its own history.

A reader should be able to understand both *what* the kernel does and *how we got
here* from the file alone.

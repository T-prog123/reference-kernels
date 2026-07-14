# attempts/ — every idea we try

The complete archive of implementation ideas, **working or not**. Each attempt is a
full self-contained `submission_vN.py` (Triton inline) with a header comment stating
its **parent** and **one-line hypothesis**.

`kernel-implementer` copies the current `submission.py` here **every time** it
produces a distinct idea worth evaluating — even ones we expect might fail. Nothing
is thrown away: a failed or slower attempt still teaches us something, and a
single-case win here can be grafted into the dispatcher later.

- Overall-geomean winners are additionally copied to `../winners/` (same `vN`).
- Per-attempt results and per-case times live in `../notes/experiments.md`.

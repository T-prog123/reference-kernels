# winners/ — attempts that improved the ranked metric

Only attempts that **improved the overall benchmark geomean** (and passed all
correctness gates) land here. This is the record progression of submittable
champions — the newest `vN` here is what you'd hand to the ranked leaderboard.

A winner is a **copy** of its `../attempts/submission_vN.py` (same `vN`), so it's
always traceable back to the full archive.

`benchmark-runner` promotes here **only** when the measured geomean beats the
previous champion. Note this is distinct from **per-case** wins: an attempt can set
a new best time for one benchmark case yet regress the overall geomean — that does
**not** come here, but its per-case record is still captured in
`../notes/experiments.md` (the per-case best-of table) so the win isn't lost.

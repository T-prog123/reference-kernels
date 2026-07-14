# submission.py — active working kernel for the eigh competition.
#
# v0 — BASELINE. parent: none.
# Hypothesis: establish a correct reference and seed the ledger. This is the
# torch.linalg.eigh reference itself (same solver the checker compares invariants
# against), so it must pass all correctness gates. Its benchmark geomean is the
# baseline every Triton attempt must beat. NOT a Triton kernel yet — it's the
# starting point the real work departs from.
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    # torch returns (eigenvalues ascending, eigenvectors as columns); the contract
    # wants (Q, L) = (vectors, values). Same convention as the reference.
    values, vectors = torch.linalg.eigh(data)
    return vectors, values

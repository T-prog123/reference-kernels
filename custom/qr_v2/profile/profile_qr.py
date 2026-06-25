import argparse
from pathlib import Path
import sys

import torch
from torch.profiler import ProfilerActivity, profile, record_function


BENCHMARK_SHAPES = (
    (20, 32),
    (40, 176),
    (40, 352),
    (640, 512),
    (60, 1024),
    (8, 2048),
    (2, 4096),
)


def _make_input(batch: int, n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    return torch.randn((batch, n, n), device="cuda", dtype=torch.float32, generator=gen)


def _sync() -> None:
    torch.cuda.synchronize()


def _load_custom_kernel(problem_dir: Path):
    sys.path.insert(0, str(problem_dir))
    from submission import custom_kernel

    return custom_kernel


def _run_custom(data: torch.Tensor, custom_kernel):
    with record_function("custom_kernel"):
        return custom_kernel(data)


def _run_geqrf(data: torch.Tensor):
    with record_function("torch.geqrf"):
        return torch.geqrf(data)


def _profile_call(label: str, fn, data: torch.Tensor, warmup: int, repeat: int):
    for _ in range(warmup):
        fn(data)
    _sync()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for idx in range(repeat):
            range_name = f"{label}.iteration_{idx}"
            with record_function(range_name):
                _sync()
                fn(data)
                _sync()

    return prof


def _print_tables(label: str, batch: int, n: int, prof) -> None:
    header = f"{label}: batch={batch}, n={n}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    print("\n-- sorted by cuda_time_total --")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))

    print("\n-- sorted by cpu_time_total --")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=40))


def _shape_filter(shapes: tuple[tuple[int, int], ...], only_n: int | None) -> tuple[tuple[int, int], ...]:
    if only_n is None:
        return shapes
    return tuple((batch, n) for batch, n in shapes if n == only_n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the qr_v2 submission with torch.profiler.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--only-n", type=int, default=None)
    parser.add_argument("--skip-geqrf", action="store_true")
    default_problem_dir = Path(__file__).resolve().parents[3] / "problems" / "linalg" / "qr_v2"
    parser.add_argument("--problem-dir", type=Path, default=default_problem_dir)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiler harness")

    custom_kernel = _load_custom_kernel(args.problem_dir.resolve())

    shapes = _shape_filter(BENCHMARK_SHAPES, args.only_n)
    if not shapes:
        raise ValueError(f"No benchmark shape matched --only-n={args.only_n}")

    for shape_idx, (batch, n) in enumerate(shapes):
        data = _make_input(batch, n, args.seed + shape_idx)

        custom_prof = _profile_call(
            "custom_kernel",
            lambda tensor: _run_custom(tensor, custom_kernel),
            data,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        _print_tables("custom_kernel", batch, n, custom_prof)

        if not args.skip_geqrf:
            geqrf_prof = _profile_call(
                "torch.geqrf",
                _run_geqrf,
                data,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            _print_tables("torch.geqrf", batch, n, geqrf_prof)


if __name__ == "__main__":
    main()

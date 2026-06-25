# QR v2 CUDA Profiling Runtimes

Each runtime isolates one `blocked_v8.cu` path.

| Runtime | Internal path | Shape |
| --- | --- | --- |
| `profile_tiny.cu` | `tiny_single_panel` | dense, batch 20, n 32, cond 1 |
| `profile_small.cu` | `small_no_t` | dense, batch 40, n 176, cond 1 |
| `profile_blocked.cu` | `blocked_cublas` | one scenario selected in source |

`profile_blocked.cu` scenarios:

- `StressMixed512`: mixed, batch 640, n 512, cond 2
- `TallDense4096`: dense, batch 2, n 4096, cond 1

Switch the blocked scenario by changing `kScenario` near the top of
`profile_blocked.cu`.

Build example:

```bash
make profile KERNEL=blocked_v8 PROFILE=blocked
```

This writes `build/profile_blocked` and links cuBLAS only. The measured loop is
wrapped with `cudaProfilerStart/Stop` so Nsight can skip setup and warmup.

Use `--force-overwrite=true` when reusing an Nsight output name, and
`--force-export=true` when rerunning `nsys stats`.

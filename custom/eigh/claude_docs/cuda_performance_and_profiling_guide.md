# CUDA Performance and Profiling Guide

A compact reference for improving CUDA kernels and interpreting Nsight Systems (`nsys`) and Nsight Compute (`ncu`) reports.

## 1. Main CUDA performance issues

### 1.1 Algorithmic work

Kernel tuning cannot compensate for unnecessary work.

Check first:

- Can the number of passes over the matrix be reduced?
- Are intermediate matrices being materialized unnecessarily?
- Can symmetric structure reduce storage or arithmetic?
- Are expensive operations repeated instead of reused?
- Is a kernel launch being used for work small enough to fold into another kernel?

For eigendecomposition, algorithm choice and the number of global matrix sweeps will usually matter more than small instruction-level changes.

---

### 1.2 Global-memory access

A warp should normally access adjacent, aligned addresses. Large strides or scattered addresses cause extra memory transactions and waste bandwidth.

Good targets:

- Consecutive threads access consecutive elements.
- Loads and stores are naturally aligned.
- Matrix layout matches the direction in which warps traverse data.
- Vectorized accesses are used only when alignment and layout truly support them.
- Data is not repeatedly reloaded from global memory.

Typical symptoms in `ncu`:

- High actual memory traffic but low useful bandwidth.
- Excessive sectors or transactions per request.
- Poor global load/store efficiency.
- High DRAM activity for a computation that should reuse data.

Likely fixes:

- Change thread-to-data mapping.
- Tile or transpose data through shared memory.
- Process contiguous rows or columns per warp.
- Fuse passes so intermediate results remain on-chip.

---

### 1.3 Data reuse and arithmetic intensity

Performance often improves more by transferring fewer bytes than by making arithmetic slightly cheaper.

Use registers or shared memory when values are reused. Avoid writing temporary values to global memory only to read them again in the next kernel.

Estimate:

```text
arithmetic intensity = useful arithmetic operations / bytes transferred
```

If a kernel is bandwidth-bound, reducing bytes is usually more valuable than reducing arithmetic.

---

### 1.4 Shared memory

Shared memory is useful for:

- Reusing data loaded from global memory.
- Converting strided global accesses into coalesced accesses.
- Staging matrix tiles.
- Performing block-level reductions or transposes.

It is not automatically faster. Shared memory can hurt through:

- Bank conflicts.
- Excessive synchronization.
- Excessive capacity per block, reducing resident blocks.
- Loading data that is not reused enough to repay the staging cost.

Shared memory has 32 banks on current CUDA architectures. A common transpose fix is padding a tile dimension:

```cpp
__shared__ float tile[TILE][TILE + 1];
```

This can remove severe bank conflicts caused by column access.

For supported architectures, hardware-assisted global-to-shared copies can overlap tile loading with computation and reduce intermediate register use. Use double buffering only when there is enough computation to hide the copy latency.

---

### 1.5 Registers and register spilling

Registers are the fastest storage, but each SM has a finite register file. High registers per thread can reduce the number of resident warps or blocks.

If the compiler runs out of registers, values spill into local memory. Local memory is backed by device memory and can be extremely expensive.

Look for:

- Registers per thread in `LaunchStats`.
- Local load/store instructions.
- Register-spill loads and stores.
- Low occupancy caused by register allocation.

Do not blindly force a lower register count. Reducing registers can increase spilling and make the kernel slower. Compare both runtime and spill metrics.

Compile with `-Xptxas=-v` to see register, shared-memory, and spill information.

---

### 1.6 Occupancy and available parallelism

Occupancy is the fraction of the hardware's possible resident warps that are active. It helps hide latency, but maximum occupancy is not the goal.

Low occupancy can be caused by:

- Too many registers per thread.
- Too much shared memory per block.
- Blocks that are too large.
- Too few blocks in the grid.

Important distinctions:

- **Theoretical occupancy:** allowed by launch configuration and resource usage.
- **Achieved occupancy:** what occurred during execution.

A large gap between them often indicates imbalance, tail effects, or blocks with uneven work.

A kernel with moderate occupancy can still be fast if it has strong instruction-level parallelism and little memory latency. Treat occupancy as a constraint, not a score to maximize.

---

### 1.7 Launch configuration and tail effects

Check that:

- Threads per block are multiples of 32.
- There are enough blocks to occupy all SMs.
- Work is balanced across blocks and warps.
- The final execution wave is not mostly empty.

Small batches or one-block-per-matrix designs can leave much of the GPU idle. Large matrices may require parallelism inside each matrix; large batches may instead provide enough parallelism across matrices.

Try 128–256 threads per block as an initial range, then measure. The best block size depends on registers, shared memory, barriers, and work per thread.

---

### 1.8 Warp divergence

Threads in a warp execute together. When they take different branches, the paths may execute serially.

Common causes:

- Per-element special cases.
- Irregular convergence logic.
- Different iteration counts inside one warp.
- Boundary handling spread throughout the main loop.

Possible fixes:

- Group similar work together.
- Separate rare cases from the main path.
- Use predication for short branches.
- Arrange matrix or batch assignments so a warp handles similar inputs.

Do not remove branches merely because they exist. Measure branch efficiency and whether scheduler issue is actually impaired.

---

### 1.9 Synchronization and reductions

Block-wide barriers stall every participating warp until all required warps arrive.

Reduce their cost by:

- Using warp-level primitives when communication stays within one warp.
- Combining multiple operations between barriers.
- Avoiding barriers inside tight loops where possible.
- Ensuring warps perform similar amounts of work before each barrier.

For reductions, use warp shuffles for warp-local stages and shared memory only for cross-warp combination.

---

### 1.10 Instruction cost and pipeline pressure

Expensive operations include division, square root, transcendental functions, integer division, and repeated type conversion.

Check:

- Whether reciprocal or reciprocal-square-root forms are valid.
- Whether constants accidentally promote FP32 expressions to FP64.
- Whether address calculations contain avoidable integer division or modulo.
- Whether one execution pipeline is saturated while others are idle.
- Whether loop unrolling reduces overhead or instead causes register pressure and code-size growth.

Use fast math or approximate intrinsics selectively. They can change denormal handling, special cases, and numerical accuracy.

---

### 1.11 Lower precision and specialized matrix hardware

FP16, BF16, TF32, FP8, or tensor-core operations can greatly increase throughput, but only when the algorithm tolerates their error.

Safer patterns include:

- Lower-precision bulk operations with FP32 accumulation.
- FP32 normalization and convergence tests.
- FP32 residual calculation.
- Iterative refinement after an approximate result.
- Dynamically falling back to FP32 for difficult matrices.

For symmetric eigendecomposition, clustered eigenvalues, rank deficiency, and near-rank deficiency are particularly sensitive. A low-precision method may produce reasonable eigenvalues while badly damaging orthogonality or eigenvectors. Always validate the eigen-equation, reconstruction, orthogonality, and eigenvalue ordering against the original FP32 matrix.

---

### 1.12 Kernel count, fusion, and temporary allocations

Many short kernels can be dominated by launch overhead rather than GPU execution.

Potential improvements:

- Fuse adjacent elementwise or reduction stages.
- Keep intermediates in registers or shared memory.
- Reuse preallocated buffers.
- Avoid allocation and initialization inside the timed region.
- Avoid device-wide synchronization between phases unless required.

Fusion is not always beneficial: an oversized fused kernel may increase register pressure, reduce occupancy, or duplicate work.

---

## 2. What to inspect in an `nsys` report

`nsys` answers: **Where is total time going?**

Start with the GPU timeline and kernel summaries.

### 2.1 Dominant kernels

Inspect:

- Total GPU time per kernel.
- Number of calls.
- Average, median, minimum, maximum, and variance.
- Grid and block dimensions.

Useful reports:

```bash
nsys stats --report cuda_gpu_kern_sum profile.nsys-rep
nsys stats --report cuda_gpu_kern_gb_sum profile.nsys-rep
```

Optimize kernels by total contribution, not merely the slowest single invocation.

### 2.2 Gaps between kernels

Visible GPU-idle gaps can indicate:

- CPU launch overhead.
- Python or framework work between launches.
- Synchronization.
- Allocation or deallocation.
- JIT compilation or first-run initialization.
- Too many very small kernels.

If kernels are short and gaps are significant, kernel fusion or fewer launches may matter more than kernel-body tuning.

### 2.3 API time, queue time, and kernel time

The `cuda_kern_exec_sum` report separates:

- **API time:** CPU time spent issuing the launch.
- **Queue time:** delay before the GPU begins the kernel.
- **Kernel time:** actual GPU execution.

```bash
nsys stats --report cuda_kern_exec_sum profile.nsys-rep
```

Queue time is not inherently bad. It often means the GPU was already busy. A lack of queueing combined with idle gaps may instead mean the GPU is starved for work.

### 2.4 Expensive CUDA API calls

```bash
nsys stats --report cuda_api_sum profile.nsys-rep
```

Look for repeated or expensive:

- Synchronization calls.
- Memory allocation/free calls.
- Memory copies or sets.
- Module loading or compilation activity.

### 2.5 Memory operations

```bash
nsys stats --report cuda_gpu_mem_time_sum profile.nsys-rep
nsys stats --report cuda_gpu_mem_size_sum profile.nsys-rep
```

Check whether copies, clears, or intermediate-buffer traffic form a meaningful fraction of runtime.

### 2.6 Capture tips

Use a release build and profile only a stable, warmed-up region:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o profile <command>
```

Avoid long captures and unnecessary tracing options. Profiling itself adds overhead, so use the ordinary benchmark for final timing.

---

## 3. What to inspect in an `ncu` report

`ncu` answers: **Why is a selected kernel slow?**

Profile only representative launches of kernels already identified by `nsys`. Metric collection may replay kernels and heavily perturb runtime; do not treat profiled kernel duration as the competition timing.

### 3.1 First-pass sections

Start with:

- `SpeedOfLight`
- `LaunchStats`
- `Occupancy`

Then add only the section needed for the suspected bottleneck:

- `MemoryWorkloadAnalysis`
- `ComputeWorkloadAnalysis`
- `SchedulerStats`
- `WarpStateStats`
- `SourceCounters`
- `InstructionStats`

Example:

```bash
ncu \
  --section SpeedOfLight \
  --section LaunchStats \
  --section Occupancy \
  --launch-count 1 \
  -o kernel_profile \
  <command>
```

Use kernel-name filtering plus launch skip/count when the same kernel executes many times.

### 3.2 SpeedOfLight: first bottleneck classification

Interpret the high-level compute and memory throughput:

- **Memory high, compute lower:** likely bandwidth-bound. Reduce bytes and improve reuse/coalescing.
- **Compute high, memory lower:** likely arithmetic-pipeline-bound. Reduce instructions or use faster hardware paths.
- **Both low:** often latency, dependencies, insufficient parallelism, imbalance, barriers, or a grid too small to fill the GPU.
- **Both moderately high:** the kernel may already use the device well; algorithmic reductions may be required for a major gain.

### 3.3 LaunchStats and Occupancy

Inspect:

- Threads per block.
- Number of blocks.
- Registers per thread.
- Static and dynamic shared memory.
- Theoretical and achieved occupancy.
- Resource limiting factor.

Red flags:

- Very few blocks relative to SM count.
- One resource sharply limiting resident blocks.
- Large theoretical/achieved occupancy gap.
- High register use plus spills.
- Large shared-memory allocation with little demonstrated reuse.

### 3.4 MemoryWorkloadAnalysis

Inspect:

- DRAM, L2, and L1/TEX throughput.
- Bytes read and written.
- Cache hit rates.
- Sectors or transactions per request.
- Global load/store efficiency.
- Shared-memory bank conflicts.
- Local-memory traffic and spills.

Interpretation:

- High DRAM throughput near the hardware limit: reduce total data traffic.
- Low useful bandwidth but high actual traffic: fix coalescing or wasted transactions.
- Low cache hit rate on reused data: improve tiling or locality.
- High L2 hit rate with low DRAM use: data reuse may already be effective.
- Shared bank conflicts: change indexing, layout, or padding.
- Unexpected local traffic: investigate register spills or large thread-local arrays.

### 3.5 ComputeWorkloadAnalysis and InstructionStats

Inspect:

- Achieved instructions per cycle.
- Utilization of FP32, FP64, integer, special-function, tensor, and load/store pipelines.
- Instruction mix.
- Expensive conversions or divisions.
- Total instruction count.

A highly utilized pipeline may be the actual throughput limit. A low-utilization kernel with many instructions is more likely dependency- or latency-bound.

### 3.6 SchedulerStats

Important values:

- Active warps per scheduler.
- Eligible warps per scheduler.
- Issued warps per cycle.
- Skipped issue slots.

Interpretation:

- Many active warps but few eligible warps: warps are waiting on dependencies, memory, barriers, or execution units.
- Few active warps: occupancy or grid-size problem.
- Eligible warps available but low issue: possible pipeline saturation or instruction scheduling constraint.
- Many skipped issue slots: poor latency hiding; inspect warp stalls next.

### 3.7 WarpStateStats and SourceCounters

Only prioritize stall reasons when schedulers frequently fail to issue instructions. A large stall percentage is not automatically a performance problem.

Common patterns:

- **Long scoreboard:** waiting for global/local-memory data; improve locality, coalescing, reuse, or independent work.
- **Short scoreboard:** often shared-memory or other on-chip dependency; inspect bank conflicts and dependency chains.
- **Barrier:** excessive or imbalanced synchronization.
- **Branch-related stalls:** divergence or expensive control flow.
- **Execution dependency:** long dependent arithmetic chain; increase instruction-level parallelism.

Use source/SASS correlation to find the exact instructions responsible. Do not optimize the name of a stall without locating the code that causes it.

---

## 4. Profiling rules that prevent misleading conclusions

1. Warm up before capture. Exclude compilation, module loading, allocations, and first-use library setup.
2. Use optimized code with line information; do not profile a debug build.
3. Use `nsys` to locate the time, then `ncu` to explain a selected kernel.
4. Profile the exact GPU, input shape, and numerical case relevant to the benchmark.
5. Compare reports using identical clocks, software versions, launch configuration, and input.
6. Collect the smallest useful metric set. More `ncu` sections mean more replay and perturbation.
7. Benchmark separately. Profiler timings are diagnostic, not final performance numbers.
8. Check multiple shapes. An optimization for batch-heavy 512×512 matrices may harm 2048×2048 or 4096×4096 cases.
9. Record bytes, instructions, registers, occupancy, and kernel count—not only runtime.
10. Re-run correctness after every precision, ordering, or synchronization change.

---

## 5. Fast interpretation table

| Observation | Most likely issue | First things to try |
|---|---|---|
| Many tiny kernels and visible gaps | Launch/host overhead | Fuse stages; remove unnecessary launches and synchronization |
| Memory throughput near peak | Bandwidth-bound | Reduce reads/writes; increase reuse; remove intermediates |
| High actual traffic, low useful traffic | Poor coalescing | Remap threads; tile/transpose; align accesses |
| Shared-memory conflicts | Bank conflicts | Change indexing; pad tile dimensions |
| Low occupancy from registers | Register pressure | Reduce live values or unrolling; check spills before limiting registers |
| Low occupancy from shared memory | Oversized tiles | Smaller tiles; staged processing; less per-block storage |
| High occupancy, few eligible warps | Dependency/latency | Increase independent work; reduce dependency chains; improve locality |
| Barrier stalls | Synchronization or imbalance | Warp-level operations; fewer barriers; balance work |
| Compute pipeline near peak | Compute-bound | Reduce instruction count; use specialized operations where valid |
| Compute and memory both low | Too little parallelism or latency | Increase blocks/work concurrency; inspect scheduler and stall data |
| Lower precision is fast but fails | Numerical instability | FP32 accumulation/residual/refinement; selective fallback |

---

## Official NVIDIA references

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Systems Post-Collection Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

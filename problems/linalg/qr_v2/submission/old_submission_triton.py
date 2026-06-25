import torch
import triton
import triton.language as tl
from task import input_t, output_t


UPDATE_PRECISION = "fp32"  # "fp32", "tf32", "tf32x3", "bf16", "fp16", "fp8"
BLOCK_SIZE = 32
PANEL_COL_BLOCK = 32
BATCHED_TRAILING_COL_BLOCK = 32
LARGE_TRAILING_COL_BLOCK = 32
ROW_BLOCK = 256
WY_ROW_BLOCK = 128
LARGE_UPDATE_ROW_BLOCK = 128


def _precision_config() -> tuple[str, str]:
    if UPDATE_PRECISION == "fp32":
        return "ieee", "fp32"
    if UPDATE_PRECISION in ("tf32", "tf32x3"):
        return UPDATE_PRECISION, "fp32"
    if UPDATE_PRECISION in ("fp16", "bf16"):
        return "ieee", UPDATE_PRECISION
    if UPDATE_PRECISION == "fp8":
        raise RuntimeError("UPDATE_PRECISION='fp8' needs explicit scaling and is not enabled")
    raise RuntimeError(f"Unsupported UPDATE_PRECISION={UPDATE_PRECISION!r}")


@triton.jit
def _apply_reflector_kernel(
    h_ptr,
    tau_ptr,
    n: tl.constexpr,
    k: tl.constexpr,
    col_start: tl.constexpr,
    col_end: tl.constexpr,
    col_block: tl.constexpr,
    row_block: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    cols = col_start + pid_c * col_block + tl.arange(0, col_block)
    col_mask = cols < col_end

    base = pid_b * n * n
    tau = tl.load(tau_ptr + pid_b * n + k)
    acc = tl.zeros((col_block,), tl.float32)

    for row_start in range(k, n, row_block):
        r = row_start + tl.arange(0, row_block)
        row_mask = r < n
        v = tl.load(h_ptr + base + r * n + k, mask=row_mask, other=0.0)
        v = tl.where(r == k, 1.0, v)
        a = tl.load(
            h_ptr + base + r[:, None] * n + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(v[:, None] * a, axis=0)

    acc *= tau

    for row_start in range(k, n, row_block):
        r = row_start + tl.arange(0, row_block)
        row_mask = r < n
        v = tl.load(h_ptr + base + r * n + k, mask=row_mask, other=0.0)
        v = tl.where(r == k, 1.0, v)
        offs = h_ptr + base + r[:, None] * n + cols[None, :]
        old = tl.load(offs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        new = old - v[:, None] * acc[None, :]
        tl.store(offs, new, mask=row_mask[:, None] & col_mask[None, :])


def _apply_reflector_columns_triton(
    h: torch.Tensor,
    tau: torch.Tensor,
    k: int,
    col_start: int,
    col_end: int,
) -> None:
    if col_start >= col_end:
        return
    batch, n, _ = h.shape
    grid = (batch, triton.cdiv(col_end - col_start, PANEL_COL_BLOCK))
    _apply_reflector_kernel[grid](
        h,
        tau,
        n,
        k,
        col_start,
        col_end,
        col_block=PANEL_COL_BLOCK,
        row_block=ROW_BLOCK,
        num_warps=8,
    )


def _factor_panel_column(h: torch.Tensor, tau: torch.Tensor, k: int) -> None:
    batch, n, _ = h.shape
    x = h[:, k:, k]
    alpha = x[:, 0].clone()
    one = torch.ones((batch,), device=h.device, dtype=torch.float32)
    zero = torch.zeros((batch,), device=h.device, dtype=torch.float32)

    if k + 1 < n:
        x_tail = x[:, 1:]
        tail_norm = torch.linalg.vector_norm(x_tail, dim=1)
    else:
        x_tail = None
        tail_norm = zero

    x_norm = torch.sqrt(alpha * alpha + tail_norm * tail_norm)
    beta = -torch.where(alpha >= 0.0, one, -one) * x_norm
    active = tail_norm > 0.0
    tau_k = torch.where(active, (beta - alpha) / beta, zero)
    denom = torch.where(active, alpha - beta, one)

    h[:, k, k] = torch.where(active, beta, alpha)
    tau[:, k] = tau_k

    if k + 1 < n:
        h[:, k + 1 :, k] = torch.where(
            active[:, None],
            x_tail / denom[:, None],
            torch.zeros_like(x_tail),
        )


def _make_v_block(h: torch.Tensor, block_start: int, block_end: int) -> torch.Tensor:
    batch, n, _ = h.shape
    width = block_end - block_start
    v = torch.tril(h[:, block_start:, block_start:block_end].clone())
    diag = torch.arange(width, device=h.device)
    v[:, diag, diag] = 1.0
    return v


def _build_forward_t(v: torch.Tensor, tau_block: torch.Tensor) -> torch.Tensor:
    batch, _, width = v.shape
    t = torch.zeros((batch, width, width), device=v.device, dtype=torch.float32)

    for j in range(width):
        tau_j = tau_block[:, j]
        t[:, j, j] = tau_j
        if j == 0:
            continue

        y = torch.bmm(v[:, :, :j].transpose(1, 2), v[:, :, j : j + 1]).squeeze(-1)
        z = torch.bmm(t[:, :j, :j], y.unsqueeze(-1)).squeeze(-1)
        t[:, :j, j] = -tau_j[:, None] * z

    return t


@triton.jit
def _apply_block_reflector_kernel(
    h_ptr,
    t_ptr,
    n: tl.constexpr,
    block_start: tl.constexpr,
    block_end: tl.constexpr,
    block_size: tl.constexpr,
    col_block: tl.constexpr,
    row_block: tl.constexpr,
    dot_precision: tl.constexpr,
    operand_precision: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    cols = block_end + pid_c * col_block + tl.arange(0, col_block)
    col_mask = cols < n
    rows_i = tl.arange(0, row_block)
    js = tl.arange(0, block_size)
    v_cols = block_start + js
    base = pid_b * n * n

    t = tl.load(
        t_ptr + pid_b * block_size * block_size + js[:, None] * block_size + js[None, :]
    )
    acc = tl.zeros((block_size, col_block), tl.float32)

    for row_start in range(block_start, n, row_block):
        rows = row_start + rows_i
        row_mask = rows < n
        v = tl.load(
            h_ptr + base + rows[:, None] * n + v_cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        v = tl.where(rows[:, None] < v_cols[None, :], 0.0, v)
        v = tl.where(rows[:, None] == v_cols[None, :], 1.0, v)
        a = tl.load(
            h_ptr + base + rows[:, None] * n + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )

        if operand_precision == "fp16":
            v_dot = v.to(tl.float16)
            a_dot = a.to(tl.float16)
        elif operand_precision == "bf16":
            v_dot = v.to(tl.bfloat16)
            a_dot = a.to(tl.bfloat16)
        else:
            v_dot = v
            a_dot = a

        acc += tl.dot(tl.trans(v_dot), a_dot, input_precision=dot_precision)

    w = tl.dot(t, acc, input_precision=dot_precision)

    for row_start in range(block_start, n, row_block):
        rows = row_start + rows_i
        row_mask = rows < n
        v = tl.load(
            h_ptr + base + rows[:, None] * n + v_cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        v = tl.where(rows[:, None] < v_cols[None, :], 0.0, v)
        v = tl.where(rows[:, None] == v_cols[None, :], 1.0, v)

        if operand_precision == "fp16":
            v_dot = v.to(tl.float16)
            w_dot = w.to(tl.float16)
        elif operand_precision == "bf16":
            v_dot = v.to(tl.bfloat16)
            w_dot = w.to(tl.bfloat16)
        else:
            v_dot = v
            w_dot = w

        update = tl.dot(v_dot, w_dot, input_precision=dot_precision)
        offs = h_ptr + base + rows[:, None] * n + cols[None, :]
        old = tl.load(offs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        tl.store(offs, old - update, mask=row_mask[:, None] & col_mask[None, :])


def _apply_block_reflector_triton(
    h: torch.Tensor,
    t_update: torch.Tensor,
    block_start: int,
    block_end: int,
    col_block: int,
) -> None:
    batch, n, _ = h.shape
    if block_end >= n:
        return

    dot_precision, operand_precision = _precision_config()
    width = block_end - block_start
    grid = (batch, triton.cdiv(n - block_end, col_block))
    _apply_block_reflector_kernel[grid](
        h,
        t_update,
        n,
        block_start,
        block_end,
        width,
        col_block=col_block,
        row_block=WY_ROW_BLOCK,
        dot_precision=dot_precision,
        operand_precision=operand_precision,
        num_warps=8,
    )


@triton.jit
def _compute_block_w_kernel(
    h_ptr,
    t_ptr,
    w_ptr,
    n: tl.constexpr,
    block_start: tl.constexpr,
    block_end: tl.constexpr,
    block_size: tl.constexpr,
    num_col_tiles: tl.constexpr,
    col_block: tl.constexpr,
    row_block: tl.constexpr,
    dot_precision: tl.constexpr,
    operand_precision: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    cols = block_end + pid_c * col_block + tl.arange(0, col_block)
    col_mask = cols < n
    rows_i = tl.arange(0, row_block)
    js = tl.arange(0, block_size)
    v_cols = block_start + js
    base = pid_b * n * n

    t = tl.load(
        t_ptr + pid_b * block_size * block_size + js[:, None] * block_size + js[None, :]
    )
    acc = tl.zeros((block_size, col_block), tl.float32)

    for row_start in range(block_start, n, row_block):
        rows = row_start + rows_i
        row_mask = rows < n
        v = tl.load(
            h_ptr + base + rows[:, None] * n + v_cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        v = tl.where(rows[:, None] < v_cols[None, :], 0.0, v)
        v = tl.where(rows[:, None] == v_cols[None, :], 1.0, v)
        a = tl.load(
            h_ptr + base + rows[:, None] * n + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )

        if operand_precision == "fp16":
            v_dot = v.to(tl.float16)
            a_dot = a.to(tl.float16)
        elif operand_precision == "bf16":
            v_dot = v.to(tl.bfloat16)
            a_dot = a.to(tl.bfloat16)
        else:
            v_dot = v
            a_dot = a

        acc += tl.dot(tl.trans(v_dot), a_dot, input_precision=dot_precision)

    w = tl.dot(t, acc, input_precision=dot_precision)
    cs = tl.arange(0, col_block)
    tl.store(
        w_ptr
        + ((pid_b * num_col_tiles + pid_c) * block_size + js[:, None]) * col_block
        + cs[None, :],
        w,
    )


@triton.jit
def _apply_block_w_kernel(
    h_ptr,
    w_ptr,
    n: tl.constexpr,
    block_start: tl.constexpr,
    block_end: tl.constexpr,
    block_size: tl.constexpr,
    num_col_tiles: tl.constexpr,
    col_block: tl.constexpr,
    row_block: tl.constexpr,
    dot_precision: tl.constexpr,
    operand_precision: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)

    rows = block_start + pid_r * row_block + tl.arange(0, row_block)
    cols = block_end + pid_c * col_block + tl.arange(0, col_block)
    row_mask = rows < n
    col_mask = cols < n
    js = tl.arange(0, block_size)
    v_cols = block_start + js
    base = pid_b * n * n

    v = tl.load(
        h_ptr + base + rows[:, None] * n + v_cols[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    v = tl.where(rows[:, None] < v_cols[None, :], 0.0, v)
    v = tl.where(rows[:, None] == v_cols[None, :], 1.0, v)

    cs = tl.arange(0, col_block)
    w = tl.load(
        w_ptr
        + ((pid_b * num_col_tiles + pid_c) * block_size + js[:, None]) * col_block
        + cs[None, :]
    )

    if operand_precision == "fp16":
        v_dot = v.to(tl.float16)
        w_dot = w.to(tl.float16)
    elif operand_precision == "bf16":
        v_dot = v.to(tl.bfloat16)
        w_dot = w.to(tl.bfloat16)
    else:
        v_dot = v
        w_dot = w

    update = tl.dot(v_dot, w_dot, input_precision=dot_precision)
    offs = h_ptr + base + rows[:, None] * n + cols[None, :]
    old = tl.load(offs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
    tl.store(offs, old - update, mask=row_mask[:, None] & col_mask[None, :])


def _apply_block_reflector_split_triton(
    h: torch.Tensor,
    t_update: torch.Tensor,
    block_start: int,
    block_end: int,
    col_block: int,
) -> None:
    batch, n, _ = h.shape
    if block_end >= n:
        return

    dot_precision, operand_precision = _precision_config()
    width = block_end - block_start
    num_col_tiles = triton.cdiv(n - block_end, col_block)
    w = torch.empty(
        (batch, num_col_tiles, width, col_block),
        device=h.device,
        dtype=torch.float32,
    )

    _compute_block_w_kernel[(batch, num_col_tiles)](
        h,
        t_update,
        w,
        n,
        block_start,
        block_end,
        width,
        num_col_tiles,
        col_block=col_block,
        row_block=WY_ROW_BLOCK,
        dot_precision=dot_precision,
        operand_precision=operand_precision,
        num_warps=8,
    )

    row_tiles = triton.cdiv(n - block_start, LARGE_UPDATE_ROW_BLOCK)
    _apply_block_w_kernel[(batch, row_tiles, num_col_tiles)](
        h,
        w,
        n,
        block_start,
        block_end,
        width,
        num_col_tiles,
        col_block=col_block,
        row_block=LARGE_UPDATE_ROW_BLOCK,
        dot_precision=dot_precision,
        operand_precision=operand_precision,
        num_warps=8,
    )


def _factor_panel(h: torch.Tensor, tau: torch.Tensor, block_start: int, block_end: int) -> None:
    for k in range(block_start, block_end):
        _factor_panel_column(h, tau, k)
        _apply_reflector_columns_triton(h, tau, k, k + 1, block_end)


def _blocked_qr(
    data: torch.Tensor,
    trailing_col_block: int,
    split_large_update: bool,
) -> output_t:
    h = data.contiguous().clone()
    batch, n, _ = h.shape
    tau = torch.empty((batch, n), device=h.device, dtype=torch.float32)

    for block_start in range(0, n, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, n)
        _factor_panel(h, tau, block_start, block_end)

        if block_end < n:
            v = _make_v_block(h, block_start, block_end)
            t = _build_forward_t(v, tau[:, block_start:block_end])
            t_update = t.transpose(1, 2).contiguous()
            if split_large_update:
                _apply_block_reflector_split_triton(
                    h,
                    t_update,
                    block_start,
                    block_end,
                    trailing_col_block,
                )
            else:
                _apply_block_reflector_triton(
                    h,
                    t_update,
                    block_start,
                    block_end,
                    trailing_col_block,
                )

    return h, tau


def batched_blocked_qr(data: torch.Tensor) -> output_t:
    return _blocked_qr(data, BATCHED_TRAILING_COL_BLOCK, False)


def large_blocked_qr(data: torch.Tensor) -> output_t:
    return _blocked_qr(data, LARGE_TRAILING_COL_BLOCK, True)


def custom_kernel(data: input_t) -> output_t:
    n = data.shape[-1]
    if n <= 1024:
        return batched_blocked_qr(data)
    return large_blocked_qr(data)

# def custom_kernel(data: input_t) -> output_t:
#     return torch.geqrf(data)
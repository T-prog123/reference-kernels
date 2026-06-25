import torch
import triton
import triton.language as tl
from task import input_t, output_t


PANEL_SIZE = 32
TINY_CUTOFF = 32
SMALL_NO_T_CUTOFF = 256
MULTI_TILE_PANEL_MIN_N = 2048
MULTI_TILE_PANEL_MAX_BATCH = 8

UPDATE_PRECISION = "tf32"


@triton.jit
def _tiny_qr_kernel(a_ptr, h_ptr, tau_ptr, n: tl.constexpr):
    pid_b = tl.program_id(0)
    rows = tl.arange(0, 32)
    cols = tl.arange(0, 32)
    base = pid_b * n * n
    mask = (rows[:, None] < n) & (cols[None, :] < n)
    panel = tl.load(
        a_ptr + base + rows[:, None] * n + cols[None, :],
        mask=mask,
        other=0.0,
    )

    for j in tl.static_range(0, 32):
        if j < n:
            col_j = tl.sum(tl.where(cols[None, :] == j, panel, 0.0), axis=1)
            alpha = tl.sum(tl.where(rows == j, col_j, 0.0), axis=0)
            tail_mask = (rows > j) & (rows < n)
            norm2 = tl.sum(tl.where(tail_mask, col_j * col_j, 0.0), axis=0)
            norm = tl.sqrt(alpha * alpha + norm2)
            active = norm2 > 0.0
            sign = tl.where(alpha >= 0.0, 1.0, -1.0)
            beta = tl.where(active, -sign * norm, alpha)
            tau_j = tl.where(active, (beta - alpha) / beta, 0.0)
            scale = tl.where(active, 1.0 / (alpha - beta), 0.0)

            new_col = tl.where(rows == j, beta, tl.where(rows > j, col_j * scale, col_j))
            panel = tl.where(cols[None, :] == j, new_col[:, None], panel)
            tl.store(tau_ptr + pid_b * n + j, tau_j)

            v = tl.where(rows == j, 1.0, tl.where(rows > j, new_col, 0.0))
            trailing = (cols > j) & (cols < n)
            dots = tau_j * tl.sum(
                tl.where((rows[:, None] >= j) & trailing[None, :], v[:, None] * panel, 0.0),
                axis=0,
            )
            panel = tl.where(
                (rows[:, None] >= j) & trailing[None, :],
                panel - v[:, None] * dots[None, :],
                panel,
            )

    tl.store(
        h_ptr + base + rows[:, None] * n + cols[None, :],
        panel,
        mask=mask,
    )


@triton.jit
def _row_to_col_kernel(a_ptr, acol_ptr, total: tl.constexpr, n: tl.constexpr):
    offs = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    mask = offs < total
    col = offs % n
    row = (offs // n) % n
    batch = offs // (n * n)
    values = tl.load(a_ptr + offs, mask=mask, other=0.0)
    tl.store(acol_ptr + batch * n * n + row + col * n, values, mask=mask)


@triton.jit
def _col_to_row_kernel(acol_ptr, h_ptr, total: tl.constexpr, n: tl.constexpr):
    offs = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    mask = offs < total
    col = offs % n
    row = (offs // n) % n
    batch = offs // (n * n)
    values = tl.load(acol_ptr + batch * n * n + row + col * n, mask=mask, other=0.0)
    tl.store(h_ptr + offs, values, mask=mask)


@triton.jit
def _panel_factor_single_kernel(
    acol_ptr,
    tau_ptr,
    n: tl.constexpr,
    block_start,
    ib: tl.constexpr,
):
    pid_b = tl.program_id(0)
    base = pid_b * n * n
    row_offsets = tl.arange(0, 256)
    local_cols = tl.arange(0, 32)

    for j in tl.static_range(0, 32):
        if j < ib:
            col = block_start + j
            row0 = col
            norm2 = tl.full((), 0.0, tl.float32)

            for row_start in tl.static_range(0, n, 256):
                rows = row_start + row_offsets
                mask = (rows > row0) & (rows < n)
                x = tl.load(acol_ptr + base + rows + col * n, mask=mask, other=0.0)
                norm2 += tl.sum(x * x, axis=0)

            alpha = tl.load(acol_ptr + base + row0 + col * n)
            norm = tl.sqrt(alpha * alpha + norm2)
            active = norm2 > 0.0
            sign = tl.where(alpha >= 0.0, 1.0, -1.0)
            beta = tl.where(active, -sign * norm, alpha)
            tau_j = tl.where(active, (beta - alpha) / beta, 0.0)
            scale = tl.where(active, 1.0 / (alpha - beta), 0.0)
            tl.store(acol_ptr + base + row0 + col * n, beta)
            tl.store(tau_ptr + pid_b * n + col, tau_j)

            for row_start in tl.static_range(0, n, 256):
                rows = row_start + row_offsets
                mask = (rows > row0) & (rows < n)
                x = tl.load(acol_ptr + base + rows + col * n, mask=mask, other=0.0)
                tl.store(acol_ptr + base + rows + col * n, x * scale, mask=mask)

            if j + 1 < ib:
                update_cols = block_start + j + 1 + local_cols
                col_mask = local_cols < (ib - j - 1)
                dots = tl.zeros((32,), tl.float32)

                for row_start in tl.static_range(0, n, 256):
                    rows = row_start + row_offsets
                    row_mask = (rows >= row0) & (rows < n)
                    v = tl.load(acol_ptr + base + rows + col * n, mask=row_mask, other=0.0)
                    v = tl.where(rows == row0, 1.0, tl.where(rows > row0, v, 0.0))
                    c = tl.load(
                        acol_ptr + base + rows[:, None] + update_cols[None, :] * n,
                        mask=row_mask[:, None] & col_mask[None, :],
                        other=0.0,
                    )
                    dots += tl.sum(v[:, None] * c, axis=0)

                dots *= tau_j

                for row_start in tl.static_range(0, n, 256):
                    rows = row_start + row_offsets
                    row_mask = (rows >= row0) & (rows < n)
                    v = tl.load(acol_ptr + base + rows + col * n, mask=row_mask, other=0.0)
                    v = tl.where(rows == row0, 1.0, tl.where(rows > row0, v, 0.0))
                    offs = acol_ptr + base + rows[:, None] + update_cols[None, :] * n
                    old = tl.load(
                        offs,
                        mask=row_mask[:, None] & col_mask[None, :],
                        other=0.0,
                    )
                    tl.store(
                        offs,
                        old - v[:, None] * dots[None, :],
                        mask=row_mask[:, None] & col_mask[None, :],
                    )


@triton.jit
def _panel_norm_kernel(
    acol_ptr,
    partial_norms_ptr,
    n: tl.constexpr,
    block_start,
    j,
    max_panel_tiles: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    rows = block_start + pid_t * 256 + tl.arange(0, 256)
    col = block_start + j
    base = pid_b * n * n
    mask = (rows > col) & (rows < n)
    x = tl.load(acol_ptr + base + rows + col * n, mask=mask, other=0.0)
    norm2 = tl.sum(x * x, axis=0)
    tl.store(partial_norms_ptr + pid_b * max_panel_tiles + pid_t, norm2)


@triton.jit
def _panel_scalar_kernel(
    acol_ptr,
    tau_ptr,
    partial_norms_ptr,
    scalars_ptr,
    n: tl.constexpr,
    block_start,
    j,
    current_panel_tiles,
    max_panel_tiles: tl.constexpr,
):
    pid_b = tl.program_id(0)
    tiles = tl.arange(0, 32)
    mask = tiles < current_panel_tiles
    norm2 = tl.sum(
        tl.load(
            partial_norms_ptr + pid_b * max_panel_tiles + tiles,
            mask=mask,
            other=0.0,
        ),
        axis=0,
    )
    col = block_start + j
    base = pid_b * n * n
    alpha = tl.load(acol_ptr + base + col + col * n)
    norm = tl.sqrt(alpha * alpha + norm2)
    active = norm2 > 0.0
    sign = tl.where(alpha >= 0.0, 1.0, -1.0)
    beta = tl.where(active, -sign * norm, alpha)
    tau_j = tl.where(active, (beta - alpha) / beta, 0.0)
    scale = tl.where(active, 1.0 / (alpha - beta), 0.0)
    tl.store(acol_ptr + base + col + col * n, beta)
    tl.store(tau_ptr + pid_b * n + col, tau_j)
    tl.store(scalars_ptr + pid_b * 2, tau_j)
    tl.store(scalars_ptr + pid_b * 2 + 1, scale)


@triton.jit
def _panel_scale_kernel(
    acol_ptr,
    scalars_ptr,
    n: tl.constexpr,
    block_start,
    j,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    rows = block_start + pid_t * 256 + tl.arange(0, 256)
    col = block_start + j
    base = pid_b * n * n
    scale = tl.load(scalars_ptr + pid_b * 2 + 1)
    mask = (rows > col) & (rows < n)
    x = tl.load(acol_ptr + base + rows + col * n, mask=mask, other=0.0)
    tl.store(acol_ptr + base + rows + col * n, x * scale, mask=mask)


@triton.jit
def _panel_dot_kernel(
    acol_ptr,
    partial_dots_ptr,
    n: tl.constexpr,
    block_start,
    j,
    ib: tl.constexpr,
    max_panel_tiles: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    rows = block_start + pid_t * 256 + tl.arange(0, 256)
    local_cols = tl.arange(0, 32)
    ref_col = block_start + j
    update_cols = block_start + j + 1 + local_cols
    col_mask = local_cols < (ib - j - 1)
    row_mask = (rows >= ref_col) & (rows < n)
    base = pid_b * n * n

    v = tl.load(acol_ptr + base + rows + ref_col * n, mask=row_mask, other=0.0)
    v = tl.where(rows == ref_col, 1.0, tl.where(rows > ref_col, v, 0.0))
    c = tl.load(
        acol_ptr + base + rows[:, None] + update_cols[None, :] * n,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    dots = tl.sum(v[:, None] * c, axis=0)
    tl.store(
        partial_dots_ptr
        + (pid_b * max_panel_tiles + pid_t) * 32
        + local_cols,
        tl.where(col_mask, dots, 0.0),
    )


@triton.jit
def _panel_dot_reduce_kernel(
    partial_dots_ptr,
    scalars_ptr,
    j,
    ib: tl.constexpr,
    current_panel_tiles,
    max_panel_tiles: tl.constexpr,
):
    pid_b = tl.program_id(0)
    tiles = tl.arange(0, 32)
    local_cols = tl.arange(0, 32)
    col_mask = local_cols < (ib - j - 1)
    tile_mask = tiles < current_panel_tiles
    values = tl.load(
        partial_dots_ptr
        + (pid_b * max_panel_tiles + tiles[:, None]) * 32
        + local_cols[None, :],
        mask=tile_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    dots = tl.sum(values, axis=0) * tl.load(scalars_ptr + pid_b * 2)
    tl.store(
        partial_dots_ptr + pid_b * max_panel_tiles * 32 + local_cols,
        tl.where(col_mask, dots, 0.0),
        mask=col_mask,
    )


@triton.jit
def _panel_update_kernel(
    acol_ptr,
    partial_dots_ptr,
    n: tl.constexpr,
    block_start,
    j,
    ib: tl.constexpr,
    max_panel_tiles: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    rows = block_start + pid_t * 256 + tl.arange(0, 256)
    local_cols = tl.arange(0, 32)
    ref_col = block_start + j
    update_cols = block_start + j + 1 + local_cols
    col_mask = local_cols < (ib - j - 1)
    row_mask = (rows >= ref_col) & (rows < n)
    base = pid_b * n * n

    v = tl.load(acol_ptr + base + rows + ref_col * n, mask=row_mask, other=0.0)
    v = tl.where(rows == ref_col, 1.0, tl.where(rows > ref_col, v, 0.0))
    dots = tl.load(
        partial_dots_ptr + pid_b * max_panel_tiles * 32 + local_cols,
        mask=col_mask,
        other=0.0,
    )
    offs = acol_ptr + base + rows[:, None] + update_cols[None, :] * n
    old = tl.load(offs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
    tl.store(
        offs,
        old - v[:, None] * dots[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _direct_update_kernel(
    acol_ptr,
    tau_ptr,
    n: tl.constexpr,
    block_start,
    ib: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows_i = tl.arange(0, 256)
    cols_i = tl.arange(0, 16)
    cols = block_start + ib + pid_c * 16 + cols_i
    col_mask = cols < n
    base = pid_b * n * n

    for j in tl.static_range(0, 32):
        if j < ib:
            ref_col = block_start + j
            tau_j = tl.load(tau_ptr + pid_b * n + ref_col)
            dots = tl.zeros((16,), tl.float32)

            for row_start in tl.static_range(0, n, 256):
                rows = row_start + rows_i
                row_mask = (rows >= ref_col) & (rows < n)
                v = tl.load(acol_ptr + base + rows + ref_col * n, mask=row_mask, other=0.0)
                v = tl.where(rows == ref_col, 1.0, tl.where(rows > ref_col, v, 0.0))
                c = tl.load(
                    acol_ptr + base + rows[:, None] + cols[None, :] * n,
                    mask=row_mask[:, None] & col_mask[None, :],
                    other=0.0,
                )
                dots += tl.sum(v[:, None] * c, axis=0)

            dots *= tau_j

            for row_start in tl.static_range(0, n, 256):
                rows = row_start + rows_i
                row_mask = (rows >= ref_col) & (rows < n)
                v = tl.load(acol_ptr + base + rows + ref_col * n, mask=row_mask, other=0.0)
                v = tl.where(rows == ref_col, 1.0, tl.where(rows > ref_col, v, 0.0))
                offs = acol_ptr + base + rows[:, None] + cols[None, :] * n
                old = tl.load(offs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
                tl.store(
                    offs,
                    old - v[:, None] * dots[None, :],
                    mask=row_mask[:, None] & col_mask[None, :],
                )


@triton.jit
def _build_gram_kernel(
    acol_ptr,
    gram_ptr,
    n: tl.constexpr,
    block_start,
    ib: tl.constexpr,
    dot_precision: tl.constexpr,
):
    pid_b = tl.program_id(0)
    rows_i = tl.arange(0, 128)
    js = tl.arange(0, 32)
    cols = block_start + js
    base = pid_b * n * n
    acc = tl.zeros((32, 32), tl.float32)

    for row_start in tl.static_range(0, n, 128):
        rows = row_start + rows_i
        row_mask = (rows >= block_start) & (rows < n)
        v = tl.load(
            acol_ptr + base + rows[:, None] + cols[None, :] * n,
            mask=row_mask[:, None] & (js[None, :] < ib),
            other=0.0,
        )
        v = tl.where(rows[:, None] < cols[None, :], 0.0, v)
        v = tl.where(rows[:, None] == cols[None, :], 1.0, v)
        acc += tl.dot(tl.trans(v), v, input_precision=dot_precision)

    tl.store(
        gram_ptr
        + pid_b * 32 * 32
        + js[:, None] * 32
        + js[None, :],
        acc,
        mask=(js[:, None] < ib) & (js[None, :] < ib),
    )


@triton.jit
def _build_t_kernel(
    gram_ptr,
    tau_ptr,
    t_ptr,
    n: tl.constexpr,
    block_start,
    ib: tl.constexpr,
):
    pid_b = tl.program_id(0)
    r = tl.arange(0, 32)
    c = tl.arange(0, 32)
    rr = r[:, None]
    cc = c[None, :]
    t = tl.zeros((32, 32), tl.float32)
    gram_base = pid_b * 32 * 32

    for i in tl.static_range(0, 32):
        if i < ib:
            tau_i = tl.load(tau_ptr + pid_b * n + block_start + i)
            g_col = tl.load(
                gram_ptr + gram_base + c * 32 + i,
                mask=c < i,
                other=0.0,
            )
            z = -tau_i * g_col
            accum = tl.sum(t * z[None, :], axis=1)
            t = tl.where((cc == i) & (rr < i), accum[:, None], t)
            t = tl.where((rr == i) & (cc == i), tau_i, t)

    tl.store(
        t_ptr + gram_base + rr * 32 + cc,
        t,
        mask=(rr < ib) & (cc < ib),
    )


@triton.jit
def _wy_update_fused_kernel(
    acol_ptr,
    t_ptr,
    n: tl.constexpr,
    block_start,
    ib: tl.constexpr,
    dot_precision: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows_i = tl.arange(0, 128)
    js = tl.arange(0, 32)
    cs = tl.arange(0, 32)
    v_cols = block_start + js
    cols = block_start + ib + pid_c * 32 + cs
    col_mask = cols < n
    base = pid_b * n * n
    s = tl.zeros((32, 32), tl.float32)

    for row_start in tl.static_range(0, n, 128):
        rows = row_start + rows_i
        row_mask = (rows >= block_start) & (rows < n)
        v = tl.load(
            acol_ptr + base + rows[:, None] + v_cols[None, :] * n,
            mask=row_mask[:, None] & (js[None, :] < ib),
            other=0.0,
        )
        v = tl.where(rows[:, None] < v_cols[None, :], 0.0, v)
        v = tl.where(rows[:, None] == v_cols[None, :], 1.0, v)
        c = tl.load(
            acol_ptr + base + rows[:, None] + cols[None, :] * n,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        s += tl.dot(tl.trans(v), c, input_precision=dot_precision)

    m = tl.arange(0, 32)
    l = tl.arange(0, 32)
    tt = tl.load(
        t_ptr
        + pid_b * 32 * 32
        + l[None, :] * 32
        + m[:, None],
        mask=(m[:, None] < ib) & (l[None, :] < ib),
        other=0.0,
    )
    s2 = tl.dot(tt, s, input_precision=dot_precision)

    for row_start in tl.static_range(0, n, 128):
        rows = row_start + rows_i
        row_mask = (rows >= block_start) & (rows < n)
        v = tl.load(
            acol_ptr + base + rows[:, None] + v_cols[None, :] * n,
            mask=row_mask[:, None] & (js[None, :] < ib),
            other=0.0,
        )
        v = tl.where(rows[:, None] < v_cols[None, :], 0.0, v)
        v = tl.where(rows[:, None] == v_cols[None, :], 1.0, v)
        update = tl.dot(v, s2, input_precision=dot_precision)
        offs = acol_ptr + base + rows[:, None] + cols[None, :] * n
        old = tl.load(offs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        tl.store(offs, old - update, mask=row_mask[:, None] & col_mask[None, :])


@triton.jit
def _wy_partial_s_kernel(
    acol_ptr,
    partial_s_ptr,
    n: tl.constexpr,
    block_start,
    ib: tl.constexpr,
    max_update_row_tiles: tl.constexpr,
    max_col_tiles: tl.constexpr,
    dot_precision: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)
    rows = block_start + pid_r * 128 + tl.arange(0, 128)
    js = tl.arange(0, 32)
    cs = tl.arange(0, 32)
    v_cols = block_start + js
    cols = block_start + ib + pid_c * 32 + cs
    row_mask = rows < n
    col_mask = cols < n
    base = pid_b * n * n

    v = tl.load(
        acol_ptr + base + rows[:, None] + v_cols[None, :] * n,
        mask=row_mask[:, None] & (js[None, :] < ib),
        other=0.0,
    )
    v = tl.where(rows[:, None] < v_cols[None, :], 0.0, v)
    v = tl.where(rows[:, None] == v_cols[None, :], 1.0, v)
    c = tl.load(
        acol_ptr + base + rows[:, None] + cols[None, :] * n,
        mask=row_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    s = tl.dot(tl.trans(v), c, input_precision=dot_precision)
    out = (
        ((pid_b * max_update_row_tiles + pid_r) * max_col_tiles + pid_c)
        * 32
        * 32
    )
    tl.store(
        partial_s_ptr + out + js[:, None] * 32 + cs[None, :],
        s,
        mask=(js[:, None] < ib) & col_mask[None, :],
    )


@triton.jit
def _wy_reduce_s_kernel(
    partial_s_ptr,
    t_ptr,
    final_s_ptr,
    current_update_row_tiles,
    ib: tl.constexpr,
    max_update_row_tiles: tl.constexpr,
    max_col_tiles: tl.constexpr,
    dot_precision: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    js = tl.arange(0, 32)
    cs = tl.arange(0, 32)
    s = tl.zeros((32, 32), tl.float32)

    for rt in tl.static_range(0, 64):
        if rt < max_update_row_tiles:
            tile_active = rt < current_update_row_tiles
            off = (
                ((pid_b * max_update_row_tiles + rt) * max_col_tiles + pid_c)
                * 32
                * 32
            )
            s += tl.load(
                partial_s_ptr + off + js[:, None] * 32 + cs[None, :],
                mask=tile_active & (js[:, None] < ib),
                other=0.0,
            )

    m = tl.arange(0, 32)
    l = tl.arange(0, 32)
    tt = tl.load(
        t_ptr
        + pid_b * 32 * 32
        + l[None, :] * 32
        + m[:, None],
        mask=(m[:, None] < ib) & (l[None, :] < ib),
        other=0.0,
    )
    s2 = tl.dot(tt, s, input_precision=dot_precision)
    out = (pid_b * max_col_tiles + pid_c) * 32 * 32
    tl.store(
        final_s_ptr + out + js[:, None] * 32 + cs[None, :],
        s2,
        mask=js[:, None] < ib,
    )


@triton.jit
def _wy_apply_s_kernel(
    acol_ptr,
    final_s_ptr,
    n: tl.constexpr,
    block_start,
    ib: tl.constexpr,
    max_col_tiles: tl.constexpr,
    dot_precision: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)
    rows = block_start + pid_r * 128 + tl.arange(0, 128)
    js = tl.arange(0, 32)
    cs = tl.arange(0, 32)
    v_cols = block_start + js
    cols = block_start + ib + pid_c * 32 + cs
    row_mask = rows < n
    col_mask = cols < n
    base = pid_b * n * n

    v = tl.load(
        acol_ptr + base + rows[:, None] + v_cols[None, :] * n,
        mask=row_mask[:, None] & (js[None, :] < ib),
        other=0.0,
    )
    v = tl.where(rows[:, None] < v_cols[None, :], 0.0, v)
    v = tl.where(rows[:, None] == v_cols[None, :], 1.0, v)
    s2 = tl.load(
        final_s_ptr
        + (pid_b * max_col_tiles + pid_c) * 32 * 32
        + js[:, None] * 32
        + cs[None, :],
        mask=js[:, None] < ib,
        other=0.0,
    )
    update = tl.dot(v, s2, input_precision=dot_precision)
    offs = acol_ptr + base + rows[:, None] + cols[None, :] * n
    old = tl.load(offs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
    tl.store(offs, old - update, mask=row_mask[:, None] & col_mask[None, :])


def _factor_panel_staged(
    acol: torch.Tensor,
    tau: torch.Tensor,
    partial_norms: torch.Tensor,
    partial_dots: torch.Tensor,
    scalars: torch.Tensor,
    block_start: int,
    ib: int,
    max_panel_tiles: int,
) -> None:
    batch, n, _ = acol.shape
    current_panel_tiles = triton.cdiv(n - block_start, 256)
    for j in range(ib):
        _panel_norm_kernel[(batch, current_panel_tiles)](
            acol,
            partial_norms,
            n,
            block_start,
            j,
            max_panel_tiles,
            num_warps=8,
        )
        _panel_scalar_kernel[(batch,)](
            acol,
            tau,
            partial_norms,
            scalars,
            n,
            block_start,
            j,
            current_panel_tiles,
            max_panel_tiles,
            num_warps=1,
        )
        _panel_scale_kernel[(batch, current_panel_tiles)](
            acol,
            scalars,
            n,
            block_start,
            j,
            num_warps=8,
        )
        if j + 1 < ib:
            _panel_dot_kernel[(batch, current_panel_tiles)](
                acol,
                partial_dots,
                n,
                block_start,
                j,
                ib,
                max_panel_tiles,
                num_warps=8,
            )
            _panel_dot_reduce_kernel[(batch,)](
                partial_dots,
                scalars,
                j,
                ib,
                current_panel_tiles,
                max_panel_tiles,
                num_warps=1,
            )
            _panel_update_kernel[(batch, current_panel_tiles)](
                acol,
                partial_dots,
                n,
                block_start,
                j,
                ib,
                max_panel_tiles,
                num_warps=8,
            )


def _blocked_qr_triton(data: torch.Tensor) -> output_t:
    batch, n, _ = data.shape
    h = torch.empty_like(data)
    tau = torch.empty((batch, n), device=data.device, dtype=torch.float32)

    if n <= TINY_CUTOFF:
        _tiny_qr_kernel[(batch,)](data, h, tau, n, num_warps=1)
        return h, tau

    acol = torch.empty_like(data)
    total = batch * n * n
    copy_grid = (triton.cdiv(total, 1024),)
    _row_to_col_kernel[copy_grid](data, acol, total, n, num_warps=8)

    use_staged_panel = n >= MULTI_TILE_PANEL_MIN_N and batch <= MULTI_TILE_PANEL_MAX_BATCH
    max_panel_tiles = triton.cdiv(n, 256)
    partial_norms = None
    partial_dots = None
    scalars = None
    if use_staged_panel:
        partial_norms = torch.empty((batch, max_panel_tiles), device=data.device, dtype=torch.float32)
        partial_dots = torch.empty(
            (batch, max_panel_tiles, 32),
            device=data.device,
            dtype=torch.float32,
        )
        scalars = torch.empty((batch, 2), device=data.device, dtype=torch.float32)

    use_blocked_update = n > SMALL_NO_T_CUTOFF
    gram = None
    tmat = None
    partial_s = None
    final_s = None
    max_update_row_tiles = triton.cdiv(n, 128)
    max_col_tiles = triton.cdiv(n, 32)
    if use_blocked_update:
        gram = torch.empty((batch, 32, 32), device=data.device, dtype=torch.float32)
        tmat = torch.empty((batch, 32, 32), device=data.device, dtype=torch.float32)
        if use_staged_panel:
            partial_s = torch.empty(
                (batch, max_update_row_tiles, max_col_tiles, 32, 32),
                device=data.device,
                dtype=torch.float32,
            )
            final_s = torch.empty(
                (batch, max_col_tiles, 32, 32),
                device=data.device,
                dtype=torch.float32,
            )

    for block_start in range(0, n, 32):
        ib = min(32, n - block_start)
        if use_staged_panel:
            _factor_panel_staged(
                acol,
                tau,
                partial_norms,
                partial_dots,
                scalars,
                block_start,
                ib,
                max_panel_tiles,
            )
        else:
            _panel_factor_single_kernel[(batch,)](
                acol,
                tau,
                n,
                block_start,
                ib,
                num_warps=8,
            )

        trailing = n - block_start - ib
        if trailing <= 0:
            continue

        if not use_blocked_update:
            col_tiles = triton.cdiv(trailing, 16)
            _direct_update_kernel[(batch, col_tiles)](
                acol,
                tau,
                n,
                block_start,
                ib,
                num_warps=8,
            )
            continue

        _build_gram_kernel[(batch,)](
            acol,
            gram,
            n,
            block_start,
            ib,
            UPDATE_PRECISION,
            num_warps=8,
        )
        _build_t_kernel[(batch,)](
            gram,
            tau,
            tmat,
            n,
            block_start,
            ib,
            num_warps=8,
        )

        col_tiles = triton.cdiv(trailing, 32)
        if use_staged_panel:
            row_tiles = triton.cdiv(n - block_start, 128)
            _wy_partial_s_kernel[(batch, row_tiles, col_tiles)](
                acol,
                partial_s,
                n,
                block_start,
                ib,
                max_update_row_tiles,
                max_col_tiles,
                UPDATE_PRECISION,
                num_warps=4,
            )
            _wy_reduce_s_kernel[(batch, col_tiles)](
                partial_s,
                tmat,
                final_s,
                row_tiles,
                ib,
                max_update_row_tiles,
                max_col_tiles,
                UPDATE_PRECISION,
                num_warps=8,
            )
            _wy_apply_s_kernel[(batch, row_tiles, col_tiles)](
                acol,
                final_s,
                n,
                block_start,
                ib,
                max_col_tiles,
                UPDATE_PRECISION,
                num_warps=4,
            )
        else:
            _wy_update_fused_kernel[(batch, col_tiles)](
                acol,
                tmat,
                n,
                block_start,
                ib,
                UPDATE_PRECISION,
                num_warps=4,
            )

    _col_to_row_kernel[copy_grid](acol, h, total, n, num_warps=8)
    return h, tau


def custom_kernel(data: input_t) -> output_t:
    return _blocked_qr_triton(data)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The one-launch DeepSeek-V4.1 decode metadata of gfx942 against the vLLM
ops that it replaces.

swa_decode_meta.launch replaces the decode part of the ROCm SWA metadata
build: torch.ge for is_valid_token, the zero fill of the unused lengths,
_compute_swa_indices_and_lens_kernel (or the DSpark draft block's
ComputeDSparkNoncausalSWAIndicesKernel), build_ragged_indices_from_dense and
_copy_ragged_to_graph_buffers. topk_pack.launch replaces
compute_global_topk_ragged_indices_and_indptr. For each random decode batch
vLLM's ops run first. Then every output buffer is overwritten with -7
(is_valid_token with its negation), the fused launch runs, and every output
that a reader uses must be equal.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="the fused launches are for ROCm"
)

WINDOW = 128
MAX_TOKENS = 16384
MAX_SEQS = 128
TOPK = 512


def _swa_case(g, max_rows):
    """A random decode batch. It mixes short sequences, sequences at 128k,
    padded tokens with slot -1, a replay start inside the window, null blocks
    (block 0) and block sizes 64 and 128. Returns None when the batch has more
    rows than the fused launch takes."""
    block_size = (64, 128)[int(torch.randint(0, 2, (1,), generator=g))]
    num_reqs = int(torch.randint(1, 6, (1,), generator=g))
    q_len = (1, 6)[int(torch.randint(0, 2, (1,), generator=g))]
    lens = []
    for _ in range(num_reqs):
        kind = int(torch.randint(0, 3, (1,), generator=g))
        if kind == 0:
            lens.append(int(torch.randint(q_len, 140, (1,), generator=g)))
        elif kind == 1:
            lens.append(int(torch.randint(140, 5000, (1,), generator=g)))
        else:
            lens.append(131072 + int(torch.randint(0, 64, (1,), generator=g)))
    n = num_reqs * q_len
    if n > max_rows:
        return None
    max_blocks = (max(lens) + block_size - 1) // block_size + 1
    block_table = torch.randint(
        1, 200000, (MAX_SEQS, max_blocks), generator=g, dtype=torch.int32
    )
    # The sliding window manager points blocks outside the window at the
    # null block 0. Put some of those inside windows as well.
    block_table[torch.rand(block_table.shape, generator=g) < 0.05] = 0
    slot_mapping = torch.randint(0, 1 << 30, (n,), generator=g, dtype=torch.int64)
    slot_mapping[torch.rand(n, generator=g) < 0.15] = -1
    replay = torch.zeros(MAX_SEQS + 1, dtype=torch.int32)
    for r in range(num_reqs):
        if float(torch.rand(1, generator=g)) < 0.3:
            replay[r] = max(0, lens[r] - int(torch.randint(1, 160, (1,), generator=g)))
    return dict(
        n=n,
        block_size=block_size,
        slot_mapping=slot_mapping.cuda(),
        query_start_loc=(torch.arange(num_reqs + 1, dtype=torch.int32) * q_len).cuda(),
        seq_lens=torch.tensor(lens, dtype=torch.int32).cuda(),
        token_to_req=(torch.arange(n, dtype=torch.int32) // q_len).cuda(),
        block_table=block_table.cuda(),
        replay_start=replay.cuda(),
    )


def _swa_buffers(width, noncausal):
    """Buffers of the size that the metadata builder allocates."""
    return dict(
        noncausal=noncausal,
        width=width,
        is_valid=torch.zeros(MAX_TOKENS, dtype=torch.bool, device="cuda"),
        indices=torch.zeros(MAX_TOKENS, 1, width, dtype=torch.int32, device="cuda"),
        lens=torch.full((MAX_TOKENS,), 5, dtype=torch.int32, device="cuda"),
        ragged=torch.empty(MAX_TOKENS * width, dtype=torch.int32, device="cuda"),
        indptr=torch.empty(MAX_TOKENS + 1, dtype=torch.int32, device="cuda"),
    )


def _swa_old_ops(c, b):
    """The decode part of vLLM's SWA metadata build and its ROCm ragged
    step."""
    from vllm.models.deepseek_v41.amd.rocm import _copy_ragged_to_graph_buffers
    from vllm.v1.attention.backends.mla.sparse_swa import (
        _COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL,
        _COMPUTE_SWA_INDICES_AND_LENS_KERNEL,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
    )

    n, width = c["n"], b["width"]
    is_valid = b["is_valid"][:n]
    torch.ge(c["slot_mapping"], 0, out=is_valid)
    b["lens"][n:] = 0
    common = (c["query_start_loc"], c["seq_lens"], c["token_to_req"], is_valid)
    if b["noncausal"]:
        _COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL(
            b["indices"],
            b["lens"],
            WINDOW,
            width,
            *common,
            c["block_table"],
            c["block_size"],
            num_tokens=n,
            token_offset=0,
        )
    else:
        _COMPUTE_SWA_INDICES_AND_LENS_KERNEL(
            b["indices"],
            b["lens"],
            WINDOW,
            width,
            b["lens"],
            b["lens"],
            *common,
            c["block_table"],
            c["block_size"],
            c["replay_start"],
            num_tokens=n,
            token_offset=0,
        )
    ragged, indptr = build_ragged_indices_from_dense(
        b["indices"][:n].reshape(n, width), b["lens"][:n]
    )
    _copy_ragged_to_graph_buffers(ragged, indptr, b["ragged"], b["indptr"], n, width)


@pytest.mark.parametrize("noncausal", [False, True])
def test_swa_decode_meta_matches_vllm_ops(noncausal):
    from vllm.models.deepseek_v41.amd.ops import swa_decode_meta
    from vllm.v1.attention.backends.mla.compressor_utils import (
        get_dspark_swa_index_width,
    )

    width = get_dspark_swa_index_width(WINDOW, 5) if noncausal else WINDOW
    g = torch.Generator().manual_seed(1234)
    failed, done = [], 0
    while done < 300:
        c = _swa_case(g, swa_decode_meta.MAX_ROWS)
        if c is None:
            continue
        done += 1
        n = c["n"]
        b = _swa_buffers(width, noncausal)
        _swa_old_ops(c, b)
        nnz = int(b["indptr"][n])
        want = dict(
            is_valid=b["is_valid"][:n].clone(),
            indices=b["indices"][:n].clone(),
            lens=b["lens"].clone(),
            indptr=b["indptr"][: n + 1].clone(),
            ragged=b["ragged"][:nnz].clone(),
        )
        b["is_valid"][:n].copy_(~want["is_valid"])
        for key in ("indices", "lens", "ragged", "indptr"):
            b[key].fill_(-7)
        swa_decode_meta.launch(
            c["slot_mapping"],
            b["is_valid"][:n],
            b["indices"],
            b["lens"],
            b["ragged"],
            b["indptr"],
            c["query_start_loc"],
            c["seq_lens"],
            c["token_to_req"],
            c["block_table"],
            c["replay_start"],
            n,
            WINDOW,
            c["block_size"],
            noncausal=noncausal,
        )
        got = dict(
            is_valid=b["is_valid"][:n],
            indices=b["indices"][:n],
            lens=b["lens"],
            indptr=b["indptr"][: n + 1],
            ragged=b["ragged"][:nnz],
        )
        bad = [k for k in want if not torch.equal(want[k], got[k])]
        if bad:
            failed.append((done, n, c["block_size"], bad))
    assert not failed, (
        f"batches that differ (case, rows, block size, outputs): {failed}"
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 33])
def test_swa_builder_without_mono_matches_vllm_ops(num_tokens):
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_vllm_config,
    )
    from vllm.models.deepseek_v41.amd import ops, rocm
    from vllm.models.deepseek_v41.amd.ops import swa_decode_meta
    from vllm.platforms.rocm import on_gfx942
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    if not on_gfx942():
        pytest.skip("ordinary decode SWA fusion is enabled only on gfx942")
    config = create_vllm_config(
        max_model_len=4096,
        block_size=128,
        max_num_seqs=64,
        max_num_batched_tokens=64,
        hf_config_override={"sliding_window": WINDOW, "compress_ratios": [0, 1, 2]},
    )
    assert not config.kernel_config.monokernel
    ops.record_monokernel(config)
    device = torch.device("cuda")
    builder = rocm.DeepseekV41ROCMAiterSparseSWAMetadataBuilder(
        SlidingWindowMLASpec(
            block_size=128,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            sliding_window=WINDOW,
            model_version="deepseek_v4",
        ),
        ["layer0"],
        config,
        device,
    )
    batch = create_common_attn_metadata(
        BatchSpec([256 + row for row in range(num_tokens)], [1] * num_tokens),
        128,
        device,
        arange_block_indices=True,
    )
    fused = swa_decode_meta.build(
        builder, 0, batch, False, None, rocm.DeepseekV41ROCMAiterSparseSWAMetadata
    )
    assert (fused is not None) == (num_tokens <= 32)
    assert not ops.enabled()
    assert not ops.monokernel_cdna3_or_4()
    buffers = _swa_buffers(WINDOW, False)
    reference_inputs = dict(
        n=num_tokens,
        block_size=128,
        slot_mapping=batch.slot_mapping,
        query_start_loc=batch.query_start_loc,
        seq_lens=batch.seq_lens,
        token_to_req=torch.arange(num_tokens, device=device, dtype=torch.int32),
        block_table=batch.block_table_tensor,
        replay_start=builder.no_replay_start,
    )
    for _ in range(3):
        builder.build(0, batch)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = builder.build(0, batch)
    assert metadata.decode_swa_ragged_indices is not None
    assert batch.seq_lens_cpu_upper_bound is not None
    pointer = metadata.decode_swa_ragged_indices.data_ptr()
    for replay in range(3):
        batch.slot_mapping.fill_(-1 if replay == 1 else 1)
        batch.seq_lens.sub_(1)
        batch.seq_lens_cpu_upper_bound.sub_(1)
        batch.block_table_tensor.add_(1)
        builder.decode_swa_lens.fill_(-7)
        graph.replay()
        _swa_old_ops(reference_inputs, buffers)
        nnz = int(buffers["indptr"][num_tokens])
        for actual, expected in (
            (metadata.is_valid_token, buffers["is_valid"][:num_tokens]),
            (metadata.decode_swa_indices, buffers["indices"][:num_tokens]),
            (metadata.decode_swa_lens, buffers["lens"][:num_tokens]),
            (metadata.decode_swa_ragged_indptr, buffers["indptr"][: num_tokens + 1]),
            (metadata.decode_swa_ragged_indices[:nnz], buffers["ragged"][:nnz]),
        ):
            assert torch.equal(actual, expected)
        assert not builder.decode_swa_lens[num_tokens:].any()
        assert metadata.decode_swa_ragged_indices.data_ptr() == pointer
    for query_lens in ([2], [1, 2]):
        unsupported = create_common_attn_metadata(
            BatchSpec([256] * len(query_lens), query_lens), 128, device
        )
        assert (
            swa_decode_meta.build(
                builder,
                0,
                unsupported,
                False,
                None,
                rocm.DeepseekV41ROCMAiterSparseSWAMetadata,
            )
            is None
        )
    for unsupported in (
        batch.replace(
            slot_mapping=torch.cat((batch.slot_mapping, batch.slot_mapping[:1]))
        ),
        batch.replace(causal=False),
    ):
        assert (
            swa_decode_meta.build(
                builder,
                0,
                unsupported,
                False,
                None,
                rocm.DeepseekV41ROCMAiterSparseSWAMetadata,
            )
            is None
        )


def _topk_case(
    g, max_rows
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """A decode batch of top-512 rows as the indexer writes them: valid local
    indices first and -1 after them, some rows shorter than 512, some rows
    with a -1 inside the valid entries, and padded tokens."""
    n = int(torch.randint(1, max_rows + 1, (1,), generator=g))
    num_reqs = int(torch.randint(1, n + 1, (1,), generator=g))
    block_size = (32, 64, 128)[int(torch.randint(0, 3, (1,), generator=g))]
    comp_len = int(torch.randint(1, 70000, (1,), generator=g))
    blocks = (comp_len + block_size - 1) // block_size
    block_table = torch.randint(
        0, 300000, (MAX_SEQS, blocks + 1), generator=g, dtype=torch.int32
    )
    topk = torch.full((n, TOPK), -1, dtype=torch.int32)
    for t in range(n):
        valid = min(TOPK, int(torch.randint(0, comp_len + 1, (1,), generator=g)))
        if float(torch.rand(1, generator=g)) < 0.5:
            valid = min(TOPK, comp_len)
        topk[t, :valid] = torch.randperm(comp_len, generator=g)[:valid].to(torch.int32)
        if valid > 4 and float(torch.rand(1, generator=g)) < 0.2:
            topk[t, int(torch.randint(0, valid, (1,), generator=g))] = -1
    t2r = torch.sort(torch.randint(0, num_reqs, (n,), generator=g)).values
    is_valid = torch.rand(n, generator=g) > 0.15
    return (
        topk.cuda(),
        t2r.to(torch.int32).cuda(),
        block_table.cuda(),
        block_size,
        is_valid.cuda(),
    )


def test_topk_pack_matches_vllm():
    from vllm.models.deepseek_v41.amd import rocm
    from vllm.models.deepseek_v41.amd.ops import topk_pack

    g = torch.Generator().manual_seed(4321)
    failed = []
    for i in range(500):
        case = _topk_case(g, topk_pack.MAX_ROWS)
        ref = rocm._compute_global_topk_ragged_indices_and_indptr_old(*case)
        out = topk_pack.launch(*case, poison=True)
        nnz = int(ref[1][-1])
        bad = [
            name
            for name, a, b in (
                ("topk_lens", out[2], ref[2]),
                ("topk_indptr", out[1], ref[1]),
                ("global_topk_ragged", out[0][:nnz], ref[0][:nnz]),
            )
            if a.shape != b.shape or not torch.equal(a, b)
        ]
        if out[0].shape != ref[0].shape:
            bad.append("ragged shape")
        if bad:
            failed.append((i, case[0].shape[0], bad))
    assert not failed, f"batches that differ (case, rows, outputs): {failed}"

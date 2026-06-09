# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""AMD MLA attention kernels (ROCm AITER), exposed through the kernel boundary.

This is the single place that imports the third-party ``aiter`` package; runtime
code must call these re-exports rather than importing ``aiter`` directly (see
AGENTS.md). The kernels target AMD CDNA3/CDNA4 (gfx942/gfx950).

DeepSeek-style MLA serving uses two paths (mirroring the NVIDIA FlashMLA backend):

- **Decode** (absorbed): ``mla_decode_fwd(q, kv_buffer, o, qo_indptr, kv_indptr,
  kv_indices, kv_last_page_lens, max_seqlen_q, page_size=1, nhead_kv=1,
  sm_scale=None, ...)`` where ``kv_buffer`` is the 4-D paged latent cache
  ``[num_page, page_size, 1, kv_lora_rank + qk_rope_head_dim]`` and ``o`` is
  ``[total_q, nhead, kv_lora_rank]``.
- **Prefill** (full MHA on decompressed q/k/v): ``flash_attn_varlen_func`` — a
  ragged causal/non-causal attention with optional log-sum-exp output, used by
  the model's chunked-prefill core and merged across cached chunks.
"""

from __future__ import annotations

try:
    from aiter import flash_attn_varlen_func
    from aiter.mla import mla_decode_fwd

    _AVAILABLE = True
except ImportError:  # aiter is only present in AMD ROCm images
    flash_attn_varlen_func = None  # type: ignore[assignment]
    mla_decode_fwd = None  # type: ignore[assignment]
    _AVAILABLE = False


def is_available() -> bool:
    """Return True if the AITER kernels are importable on this platform."""
    return _AVAILABLE


__all__ = ["mla_decode_fwd", "flash_attn_varlen_func", "is_available"]

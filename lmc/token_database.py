# Mirrors lmcache/v1/token_database.py (subset).
# - TokenDatabase abstract base with the helpers SegmentTokenDatabase
#   actually uses (_canonicalize_hash_inputs, _hash_tokens,
#   _make_key_by_hash, process_tokens signature, ProcessTokensResult,
#   NONE_HASH).
# - SegmentTokenDatabase (the whole class): __init__,
#   _fast_split_by_subtensor, process_tokens.
# - ChunkedTokenDatabase is NOT ported (CacheBlend uses Segment only).
# The hash function is fixed to Python's builtin `hash` to match
# LMCache's `pre_caching_hash_algorithm = "builtin"` default.
from __future__ import annotations

import abc
import os
from typing import Any, Iterable, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer

from lmc.config import LMCacheEngineConfig
from lmc.storage import CacheEngineKey, LMCacheMetadata


NONE_HASH = 0

# Mirrors the type alias at lmcache/v1/token_database.py:35.
ProcessTokensResult = Tuple[int, int, Union[CacheEngineKey, int]]


class TokenDatabase(metaclass=abc.ABCMeta):
    """
    Subset of LMCache's TokenDatabase ABC.

    LMCache's version supports plugging in non-builtin hash functions
    (sha256_cbor etc.) via vLLM's `get_hash_fn_by_name`. The HF port
    runs in a single process, so we fix to Python's builtin `hash`
    (matches LMCache's `pre_caching_hash_algorithm = "builtin"`
    default). `_hash_tokens` and `_canonicalize_hash_inputs` carry the
    same shape as the LMCache versions so a future port to a different
    hash function only needs to swap `self.hash_func`.
    """

    @abc.abstractmethod
    def __init__(
        self,
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheMetadata] = None,
    ):
        algorithm = (
            config.pre_caching_hash_algorithm
            if config is not None
            else "builtin"
        )
        # Single-process port: builtin hash is the only one we wire up.
        # If the user picks something else we surface that loudly rather
        # than silently fall back, so misconfiguration is caught early.
        if algorithm != "builtin":
            raise NotImplementedError(
                f"pre_caching_hash_algorithm={algorithm!r} not supported in HF "
                "port; only 'builtin' is wired up."
            )
        self.hash_func = hash

        # PYTHONHASHSEED reproducibility note (matches LMCache's warning).
        if os.environ.get("PYTHONHASHSEED") is None:
            # In a single-process port this only matters if you need
            # determinism across separate runs (e.g. comparing chunk
            # hashes between two pytest invocations).
            pass

        self.metadata = metadata

    @abc.abstractmethod
    def process_tokens(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
        request_configs: Optional[dict] = None,
    ) -> Iterable[ProcessTokensResult]:
        raise NotImplementedError

    def _make_key_by_hash(
        self,
        chunk_hash: int,
        request_configs: Optional[dict] = None,
    ) -> CacheEngineKey:
        assert self.metadata is not None
        return CacheEngineKey(
            model_name=self.metadata.model_name,
            world_size=self.metadata.world_size,
            worker_id=self.metadata.worker_id,
            chunk_hash=chunk_hash,
            dtype=self.metadata.kv_dtype,
            request_configs=request_configs,
        )

    def _canonicalize_hash_inputs(
        self,
        prefix_hash: Optional[int],
        tokens_tuple: Tuple[int, ...],
        extra_keys: Optional[List[Any]],
    ) -> Tuple[int, Tuple[int, ...], Tuple[Any, ...]]:
        return (
            prefix_hash if prefix_hash is not None else NONE_HASH,
            tokens_tuple,
            tuple(extra_keys) if extra_keys is not None else (),
        )

    def _hash_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
        prefix_hash: Optional[int] = None,
        extra_keys: Optional[list[Any]] = None,
    ) -> int:
        if isinstance(tokens, torch.Tensor):
            tokens_tuple = tuple(tokens.cpu().tolist())
        elif isinstance(tokens, list):
            tokens_tuple = tuple(tokens)
        else:
            raise ValueError(f"Unsupported tokens type: {type(tokens)}")
        canon = self._canonicalize_hash_inputs(prefix_hash, tokens_tuple, extra_keys)
        return self.hash_func(canon)


class SegmentTokenDatabase(TokenDatabase):
    """
    Verbatim port of lmcache/v1/token_database.py:423-551
    (SegmentTokenDatabase). Differences vs LMCache:
      - `self.tokenizer = AutoTokenizer.from_pretrained(metadata.model_name)`
        identical;
      - separator tokens stripped of the BOS via [1:] (same heuristic);
      - process_tokens yields ProcessTokensResult of (start, end, key);
      - start_idx advancement: at idx==0 the chunk starts at position 0
        in the full input; at idx>0 we add sep_len to skip the
        separator tokens that precede this chunk. The audit-corrected
        formula in docs/LMCACHE_IMPLEMENTATION.md §2.7 is implemented
        line-by-line.
    """

    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheMetadata):
        super().__init__(config, metadata)

        self.tokenizer = AutoTokenizer.from_pretrained(metadata.model_name)

        # TODO(Jiayi in LMCache): figure out when to use [1:].
        # We follow the source heuristic verbatim.
        sep = self.tokenizer.encode(config.blend_special_str)
        if len(sep) > 1 and sep[0] == getattr(self.tokenizer, "bos_token_id", None):
            sep = sep[1:]
        elif len(sep) > 1:
            # LMCache always does [1:], regardless of what tokenizer adds.
            # We match that behaviour.
            sep = sep[1:]
        self.sep_tokens = torch.tensor(sep, dtype=torch.long, device="cpu")
        self.sep_len = len(self.sep_tokens)

    def _fast_split_by_subtensor(
        self,
        tokens: torch.Tensor,
    ) -> Iterable[torch.Tensor]:
        """Match `sep_tokens` with sliding windows (LMCache:441-462)."""
        if self.sep_len == 0 or len(tokens) < self.sep_len:
            yield tokens
            return

        windows = tokens.unfold(0, self.sep_len, 1)
        matches = (
            (windows == self.sep_tokens).all(dim=1).nonzero(as_tuple=True)[0].tolist()
        )

        start = 0
        for idx in matches:
            yield tokens[start:idx]
            start = idx + self.sep_len
        yield tokens[start:]

    def process_tokens(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
        request_configs: Optional[dict] = None,
    ) -> Iterable[ProcessTokensResult]:
        """Verbatim port of LMCache:464-551."""
        if tokens is not None:
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long, device="cpu")
            else:
                tokens = tokens.to(device="cpu", dtype=torch.long)

            if mask is not None:
                num_falses = mask.numel() - mask.long().sum().item()
            else:
                num_falses = 0
            assert num_falses < len(tokens), (
                "The number of Falses in the mask shouldn't be less than the length of tokens."
            )

            token_chunks = self._fast_split_by_subtensor(tokens)
            start_idx = 0
            for idx, token_chunk in enumerate(token_chunks):
                token_chunk_len = len(token_chunk)
                end_idx = start_idx + token_chunk_len
                if idx > 0:
                    start_idx += self.sep_len
                    end_idx += self.sep_len
                if start_idx >= num_falses:
                    if make_key:
                        yield (
                            start_idx,
                            end_idx,
                            self._make_key_by_hash(
                                self._hash_tokens(token_chunk),
                                request_configs,
                            ),
                        )
                    else:
                        yield start_idx, end_idx, self._hash_tokens(token_chunk)
                start_idx = end_idx
        elif hashes is not None:
            assert offsets is not None
            start_idx = 0
            for hash_val, offset in zip(hashes, offsets, strict=False):
                end_idx = start_idx + offset
                if make_key:
                    yield (
                        start_idx,
                        end_idx,
                        self._make_key_by_hash(hash_val, request_configs),
                    )
                else:
                    yield start_idx, end_idx, hash_val
                start_idx = end_idx
        else:
            raise ValueError("Either tokens or hashes must be provided.")

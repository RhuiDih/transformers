from typing import Any, Dict, List, Optional, Tuple

import torch


class Cache:
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache. These are specific to each subclass and allow new types of cache to
                be created -- all used keys are validated.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")


class DynamicCache(Cache):
    def __init__(self) -> None:
        self.key_cache: List[Tuple[torch.Tensor]] = []
        self.value_cache: List[Tuple[torch.Tensor]] = []

    def __getitem__(self, key: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if key == 0:
            return self.key_cache
        elif key == 1:
            return self.value_cache
        else:
            raise KeyError(f"Cache only supports 0 (key) and 1 (value) indexing, got {key}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        yield self.key_cache
        yield self.value_cache

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache. No additional arguments are used in `DynamicCache`.
        """
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        return (
            tuple(self.key_cache[layer_idx] for layer_idx in range(len(self.key_cache))),
            tuple(self.value_cache[layer_idx] for layer_idx in range(len(self.value_cache))),
        )

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[List[torch.FloatTensor]]) -> "DynamicCache":
        if past_key_values is None:
            return cls()
        cache = cls()
        for layer_idx, (key_states, value_states) in enumerate(zip(*past_key_values)):
            cache.update(key_states, value_states, layer_idx)
        return cache

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))


class SinkCache(Cache):
    def __init__(self, window_length: int, num_sink_tokens: int) -> None:
        self.key_cache: List[Tuple[torch.Tensor]] = []
        self.value_cache: List[Tuple[torch.Tensor]] = []
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self.cos_sin_cache = {}

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_key_rotary_pos_emb(self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        rotated_key_states = (key_states * cos) + (self.rotate_half(key_states) * sin)
        return rotated_key_states

    def get_rerotation_cos_sin(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape[-2] not in self.cos_sin_cache:
            # Upcast to float32 temporarily for better accuracy
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)

            # Compute the cos and sin required for back- and forward-rotating to one position earlier in the sequence
            original_cos = cos[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_cos = cos[self.num_sink_tokens : -key_states.shape[-2]]
            original_sin = sin[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_sin = sin[self.num_sink_tokens : -key_states.shape[-2]]
            rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
            rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

            self.cos_sin_cache[key_states.shape[-2]] = (
                rerotation_cos.to(key_states.dtype).unsqueeze(0),
                rerotation_sin.to(key_states.dtype).unsqueeze(0),
            )
        return self.cos_sin_cache[key_states.shape[-2]]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        if len(self.key_cache) <= layer_idx:
            cache_length = 0
        cache_length = self.key_cache[layer_idx].shape[-2]
        return min(cache_length, self.window_length - 1)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache. The following arguments can be used in `SinkCache`: `sin`, `cos`
                and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the rotation
                as the tokens are shifted.
        """
        # Optional kwargs for `SinkCache` -- needed on models using RoPE. `partial_rotation_size` is used on models
        # with partially rotated position embeddings, like Phi or Persimmon.
        sin = cache_kwargs.get("sin")
        cos = cache_kwargs.get("cos")
        partial_rotation_size = cache_kwargs.get("partial_rotation_size")
        using_rope = cos is not None and sin is not None

        # [bsz, num_heads, seq_len, head_dim]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        elif key_states.shape[-2] + self.get_seq_length(layer_idx) < self.window_length:
            # Growing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        else:
            # Shifting cache
            keys_to_keep = self.key_cache[layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + key_states.shape[-2] :
            ]

            # On RoPE models, we need to recompute the Key rotation as the tokens are shifted
            if using_rope:
                rerotation_cos, rerotation_sin = self.get_rerotation_cos_sin(key_states, cos, sin)
                if partial_rotation_size is not None:
                    keys_to_keep, keys_pass = (
                        key_states[..., :partial_rotation_size],
                        key_states[..., partial_rotation_size:],
                    )
                keys_to_keep = self.apply_key_rotary_pos_emb(keys_to_keep, rerotation_cos, rerotation_sin)
                if partial_rotation_size is not None:
                    keys_to_keep = torch.cat((keys_to_keep, keys_pass), dim=-1)

            # Concatenate sink tokens, shifted & rotated tokens (if needed), and new tokens
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx][:, :, : self.num_sink_tokens], keys_to_keep, key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [
                    self.value_cache[layer_idx][:, :, : self.num_sink_tokens],
                    self.value_cache[layer_idx][
                        :, :, -self.window_length + self.num_sink_tokens + value_states.shape[-2] :
                    ],
                    value_states,
                ],
                dim=-2,
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))
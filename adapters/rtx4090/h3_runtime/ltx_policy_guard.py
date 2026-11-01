"""Trusted installation and observed routing for the editable LTX policy.

Only video attn1's local-head callable inside the existing SP wrapper is
replaced. Audio, text/cross attention and masked attention retain their original
objects. Route counts prove invocation, not kernel identity or approximation
quality; CUDA traces and complete request comparisons provide that evidence.
"""
from __future__ import annotations

from functools import partial
import re


_PROTECTED = ('attn2', 'audio_attn1', 'audio_attn2',
              'audio_to_video_attn', 'video_to_audio_attn')


def transformer_blocks(transformer):
    current = transformer
    visited = set()
    for _ in range(4):
        if id(current) in visited:
            break
        visited.add(id(current))
        blocks = getattr(current, 'transformer_blocks', None)
        if blocks is not None:
            return tuple(getattr(block, '_orig_mod', block) for block in blocks)
        if hasattr(current, 'velocity_model'):
            current = current.velocity_model
        elif hasattr(current, 'model'):
            current = current.model
        else:
            break
    raise TypeError('Cannot locate LTX transformer blocks')


def _identity(module):
    def callable_identity(value):
        return (id(value), id(getattr(value, 'original_attention', None)))
    return (id(module), callable_identity(getattr(module, 'attention_function', None)),
            callable_identity(getattr(module, 'masked_attention_function', None)))


class CheckedVideoAttentionPolicy:
    def __init__(self, transformer, policy, *, full_attention_prefix_tokens=0,
                 layers=48, steps=3, video_tokens=16128):
        self.blocks = transformer_blocks(transformer)
        if len(self.blocks) != layers:
            raise RuntimeError(f'Expected {layers} LTX transformer layers')
        backend = getattr(policy, 'backend_id', None)
        if not isinstance(backend, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', backend):
            raise ValueError('LTX policy requires a stable backend_id')
        if not callable(getattr(policy, 'attention', None)):
            raise TypeError('LTX policy requires an attention callback')
        if type(full_attention_prefix_tokens) is not int or not 0 <= full_attention_prefix_tokens <= video_tokens:
            raise ValueError('Invalid protected prefix length')
        self.policy, self.backend = policy, backend
        self.layers, self.steps, self.video_tokens = layers, steps, video_tokens
        self.prefix = full_attention_prefix_tokens
        self.protected = tuple(tuple(_identity(getattr(block, name)) for name in _PROTECTED)
                               for block in self.blocks)
        self.masked = tuple(_identity(block.attn1)[2] for block in self.blocks)
        self.wrappers = tuple(block.attn1.attention_function for block in self.blocks)
        self.originals = tuple(getattr(wrapper, 'original_attention', None) for wrapper in self.wrappers)
        if not all(callable(original) for original in self.originals):
            raise TypeError('LTX policy requires existing sequence-parallel attention wrappers')
        self.routes = tuple(partial(self._call, index, original)
                            for index, original in enumerate(self.originals))
        for wrapper, route in zip(self.wrappers, self.routes):
            wrapper.original_attention = route
        self.begin_denoise()

    def begin_denoise(self):
        self.calls = self.dense_calls = self.policy_calls = self.non_dense_calls = self.protected_calls = 0
        self.per_step = [0] * self.steps

    def _call(self, layer, original, *args, **kwargs):
        expected_layer = self.calls % self.layers
        step = self.calls // self.layers
        if step >= self.steps or layer != expected_layer:
            raise RuntimeError('LTX video attention changed the frozen layer/step execution order')
        query = args[0] if args else kwargs.get('q')
        if query is None or query.shape[1] != self.video_tokens:
            raise RuntimeError('LTX video attention token geometry changed')
        dense_invocations = 0
        def dense(*values, **options):
            nonlocal dense_invocations
            dense_invocations += 1
            if dense_invocations > 1:
                raise RuntimeError('An LTX attention route invoked its dense fallback more than once')
            return original(*values, **options)
        mask = kwargs.get('mask', args[4] if len(args) > 4 else None)
        if self.prefix or mask is not None:
            # Conservative full dense protection, including every prefix query
            # and key. Masked module callables also remain wholly untouched.
            result = dense(*args, **kwargs)
            self.protected_calls += 1
        else:
            result = self.policy.attention(dense, *args, layer_index=layer, step_index=step,
                full_attention_prefix_tokens=self.prefix, **kwargs)
            self.policy_calls += 1
        self.calls += 1
        self.per_step[step] += 1
        self.dense_calls += dense_invocations
        self.non_dense_calls += int(dense_invocations == 0)
        return result

    def stats(self):
        current = tuple(tuple(_identity(getattr(block, name)) for name in _PROTECTED)
                        for block in self.blocks)
        if current != self.protected or tuple(_identity(block.attn1)[2] for block in self.blocks) != self.masked:
            raise RuntimeError('LTX policy changed protected masked/audio/text/cross attention')
        if any(block.attn1.attention_function is not wrapper or wrapper.original_attention is not route
               for block, wrapper, route in zip(self.blocks, self.wrappers, self.routes)):
            raise RuntimeError('LTX policy bypassed the trusted routing observer or SP wrapper')
        if self.calls != self.layers * self.steps or self.per_step != [self.layers] * self.steps:
            raise RuntimeError('LTX policy did not observe all three complete transformer forwards')
        if self.backend == 'dense' and self.dense_calls != self.calls:
            raise RuntimeError('Dense LTX policy bypassed the original dense backend')
        if self.backend != 'dense' and not self.non_dense_calls:
            raise RuntimeError('Requested LTX policy silently fell back to dense for every invocation')
        return dict(backend=self.backend, selected_backend=self.backend,
            completed_steps=self.steps, video_self_attention_calls=self.calls,
            dense_calls=self.dense_calls, policy_calls=self.policy_calls,
            non_dense_calls=self.non_dense_calls, protected_calls=self.protected_calls,
            calls_per_step=list(self.per_step), full_attention_prefix_tokens=self.prefix,
            protection='Full dense prefix and masks; audio/text/cross attention unchanged.',
            verification_scope='Observed Python callable routing; kernel execution requires CUDA trace evidence.',
            policy_callable=f'{type(self.policy).__module__}.{type(self.policy).__qualname__}.attention')

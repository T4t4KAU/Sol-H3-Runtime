"""Optional streaming evidence for independent prediction and scheduler replay."""
from __future__ import annotations

import dataclasses
from contextlib import contextmanager
import json
from pathlib import Path
import resource

import torch
import torch.distributed as dist

from .distributed_loading import tensor_digest


def capture_resources(directory, rank, device):
    """Persist allocator high-water marks alongside external NVML sampling."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    cgroup = {}
    for name in ('memory.current', 'memory.peak', 'memory.events'):
        source = Path('/sys/fs/cgroup') / name
        if source.exists():
            cgroup[name] = source.read_text().strip()
    (path / f'resources-rank{rank}.json').write_text(json.dumps(dict(
        rank=rank, cuda_max_allocated_bytes=torch.cuda.max_memory_allocated(device),
        cuda_max_reserved_bytes=torch.cuda.max_memory_reserved(device),
        process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cgroup=cgroup, cgroup_peak_scope='allocation lifetime'), indent=2))


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(value, **{f.name:cpu(getattr(value,f.name)) for f in dataclasses.fields(value)})
    if isinstance(value, dict):
        return {k:cpu(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return type(value)(cpu(v) for v in value)
    return value


class PredictionCapture:
    def __init__(self, model, directory, *, label='prediction', trace=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank, self.count, self.label = dist.get_rank(), 0, label
        self.pending = self.profiler = None
        self.trace = trace
        self.handles = [model.register_forward_pre_hook(self.before, with_kwargs=True),
                        model.register_forward_hook(self.after)]

    def before(self, model, args, kwargs):
        if self.rank == 0:
            self.pending = dict(args=cpu(args), inputs=cpu(kwargs))
            if self.trace and self.count == 0:
                self.profiler = torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                self.profiler.__enter__()

    def after(self, model, args, output):
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler.export_chrome_trace(str(self.directory / (self.label+'-first-forward.json')))
            kernels = [event.key for event in self.profiler.key_averages()]
            (self.directory/(self.label+'-operators.json')).write_text(json.dumps(kernels, indent=2))
            if any('_scaled_dot_product_attention_math' in key for key in kernels):
                raise RuntimeError('DiT silently selected math attention')
            self.profiler = None
        if self.rank == 0:
            torch.save(dict(index=self.count, predictions=cpu(output), **self.pending),
                       self.directory/f'{self.label}-{self.count}.pt')
            self.pending = None
        self.count += 1
        print(f'rank={self.rank} {self.label}_forward={self.count}', flush=True)

    def close(self, expected=None):
        """Always detach hooks; only completed calls produce a count receipt."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
        self.pending = None
        if expected is None:
            return
        if self.count != expected:
            raise RuntimeError(f'{self.label}: expected {expected} forwards, observed {self.count}')
        (self.directory/f'{self.label}-count-rank{self.rank}.json').write_text(json.dumps(dict(
            forwards=self.count, expected=expected, status='PASS')))


class H3Evidence:
    def __init__(self, model, block, components, state, directory, cpu_group):
        self.directory, self.group = Path(directory), cpu_group
        self.rank = dist.get_rank()
        self.capture = PredictionCapture(model, directory, label='prediction', trace=True)
        self.update_block = block.sub_blocks['update']
        self.original_class = type(self.update_block)
        original = self.original_class.__call__
        observer = self
        def update(this, components, state, i, t):
            before = (components.scheduler.step_index, components.audio_scheduler.step_index)
            components, state = original(this, components, state, i, t)
            if observer.rank == 0:
                torch.save(dict(index=i, scheduler_indices_before=before,
                    scheduler_indices_after=(components.scheduler.step_index, components.audio_scheduler.step_index),
                    video_t=cpu(t), audio_t=cpu(state.audio_timesteps[i]),
                    after=cpu((state.latents, state.audio_latents))), observer.directory/f'update-{i}.pt')
            digests=[None]*dist.get_world_size()
            digest=(tensor_digest(state.latents), tensor_digest(state.audio_latents))
            dist.all_gather_object(digests,digest,group=observer.group)
            if any(d != digest for d in digests):
                raise RuntimeError(f'H3 rank latent divergence at step {i}')
            return components,state
        self.update_block.__class__ = type('Observed'+self.original_class.__name__,
            (self.original_class,), {'__call__':update})
        if self.rank == 0:
            torch.save(dict(condition=cpu(dict(prompt_embeds=state.prompt_embeds,text_token_tags=state.text_token_tags)),
                geometry=(state.num_latent_frames,state.latent_height,state.latent_width,state.num_audio_latents),
                layout=cpu(dataclasses.asdict(state.layout)), initial=cpu((state.latents,state.audio_latents)),
                video_sigmas=cpu(components.scheduler.sigmas), audio_sigmas=cpu(components.audio_scheduler.sigmas),
                video_timesteps=cpu(state.timesteps), audio_timesteps=cpu(state.audio_timesteps),
                row_plan=cpu(state.row_timestep_plan)), self.directory/'request.pt')

    def close(self, state=None):
        self.update_block.__class__ = self.original_class
        self.capture.close(None if state is None else len(state.timesteps))
        if state is not None and self.rank == 0:
            torch.save(cpu(dict(video=state.latents, audio=state.audio_latents)), self.directory/'decode-latents.pt')


class ObservedEuler:
    """Delegate every update to the official stepper; record its actual arguments."""
    def __init__(self, stepper, directory):
        self.stepper, self.directory, self.count = stepper, Path(directory), 0

    def step(self, sample, denoised_sample, sigmas, step_index, **kwargs):
        output = self.stepper.step(sample, denoised_sample, sigmas, step_index, **kwargs)
        if dist.get_rank() == 0:
            torch.save(cpu(dict(sample=sample, denoised_sample=denoised_sample, sigmas=sigmas,
                               step_index=step_index, after=output)),
                       self.directory/f'update-{step_index}-{self.count%2}.pt')
        self.count += 1
        return output


@contextmanager
def capture_decode(vae, directory, label):
    """Observe the actual denormalized tensor at the upstream VAE decode entry."""
    if directory is None:
        yield
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    original = vae.decode
    had_override = 'decode' in vae.__dict__
    def decode(value, *args, **kwargs):
        torch.save(cpu(value), directory / (label+'-vae-input.pt'))
        return original(value, *args, **kwargs)
    vae.decode = decode
    try:
        yield
    finally:
        if had_override:
            vae.decode = original
        else:
            del vae.decode

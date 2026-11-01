"""Bounded CPU staging of native decoder chunks, before any media encoding."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SevenStepOutput:
    video: object
    audio: object
    chunks: tuple
    writer: object
    audio_sample_rate: int = 32000

    def save(self, path):
        # The released writer still owns its 16-frame views, audio normalization,
        # encoder parameters and atomic MP4 publication. Only consumption time
        # changes: the decoder is fully exhausted before this call.
        self.writer(iter(self.chunks), 8, None, path)


def collect_output(chunks, audio, writer, *, torch_module, frame_count=121,
                   height=768, width=1344, audio_samples=161333):
    """Copy each decoder yield before advancing its temporal overlap state.

    Never retain CUDA yields in a list: decoders may reuse backing storage and
    holding every GPU chunk would change their peak residency. Original BF16
    values feed the writer; FP32 TCHW values feed the numerical evaluator.
    """
    torch = torch_module
    staged = []
    frames = 0
    for chunk in chunks:
        if (chunk.ndim != 4 or tuple(chunk.shape[1:]) != (height, width, 3)
                or chunk.dtype != torch.bfloat16 or chunk.shape[0] <= 0):
            raise RuntimeError('Seven-step decoder must emit nonempty BF16 THWC RGB chunks')
        frames += chunk.shape[0]
        if frames > frame_count:
            raise RuntimeError('Seven-step decoder exceeded the fixed frame count')
        if not bool(torch.isfinite(chunk).all()):
            raise RuntimeError('Seven-step decoder returned NaN or Inf')
        # copy=True also protects CPU-yielding decoders that reuse their buffer.
        staged.append(chunk.detach().to(device='cpu', copy=True).contiguous())
    if frames != frame_count:
        raise RuntimeError(f'Seven-step decoder returned {frames} frames, expected {frame_count}')
    if (audio is None or audio.dtype != torch.float32
            or tuple(audio.shape) != (1, 2, audio_samples)
            or not bool(torch.isfinite(audio).all())):
        raise RuntimeError('Seven-step output requires finite FP32 native H3 stereo PCM')
    video = torch.cat(staged, dim=0).permute(0, 3, 1, 2).to(dtype=torch.float32).contiguous()
    audio = audio.detach().to(device='cpu', copy=True).contiguous()
    return SevenStepOutput(video=video, audio=audio, chunks=tuple(staged), writer=writer)

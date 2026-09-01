"""
StableVideoDiffusionInpaintingPipeline

Derived from TencentARC/StereoCrafter (pipelines/stereo_video_inpainting.py),
Copyright (C) 2024 THL A29 Limited (Tencent). Used and redistributed under the
StereoCrafter license (academic/research/education use only) — see
third_party/StereoCrafter-LICENSE.txt. Itself based on the Stable Video
Diffusion pipeline from HuggingFace diffusers (Apache-2.0).
Modifications: adapted for diffusers>=0.39 API.
"""

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


def _append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}")
    return x[(...,) + (None,) * dims_to_append]


def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    outputs = []
    for batch_idx in range(video.shape[0]):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        outputs.append(processor.postprocess(batch_vid, output_type))
    return outputs


@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    frames: Union[List[PIL.Image.Image], np.ndarray]


class StableVideoDiffusionInpaintingPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae, image_encoder=image_encoder, unet=unet,
            scheduler=scheduler, feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False,
            do_binarize=True, do_convert_grayscale=True)

    def _encode_image(self, image, device, num_videos_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype
        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)
        image = image * 2.0 - 1.0
        image = _resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0
        image = self.feature_extractor(
            images=image, do_normalize=True, do_center_crop=False,
            do_resize=False, do_rescale=False, return_tensors="pt",
        ).pixel_values
        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds.unsqueeze(1)
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)
        if do_classifier_free_guidance:
            image_embeddings = torch.cat([torch.zeros_like(image_embeddings), image_embeddings])
        return image_embeddings

    def _encode_vae_frames(self, frames, device, num_videos_per_prompt,
                           do_classifier_free_guidance, n_frames_per_time=5):
        frames = frames.to(device=device, dtype=self.vae.dtype)
        latent_list = []
        for i in range(0, frames.shape[0], n_frames_per_time):
            latent_list.append(self.vae.encode(frames[i:i+n_frames_per_time]).latent_dist.mode())
        frame_latents = torch.cat(latent_list, dim=0).unsqueeze(0)
        if do_classifier_free_guidance:
            frame_latents = torch.cat([torch.zeros_like(frame_latents), frame_latents])
        return frame_latents.repeat(num_videos_per_prompt, 1, 1, 1, 1)

    def _encode_mask_frames(self, frames_mask, device, num_videos_per_prompt,
                            do_classifier_free_guidance):
        frames_mask = frames_mask.to(device=device)
        frames_mask = torch.nn.functional.interpolate(frames_mask, scale_factor=1/self.vae_scale_factor)
        frames_mask = frames_mask.unsqueeze(0)
        if do_classifier_free_guidance:
            frames_mask = torch.cat([torch.zeros_like(frames_mask), frames_mask])
        return frames_mask.repeat(num_videos_per_prompt, 1, 1, 1, 1)

    def _get_add_time_ids(self, fps, motion_bucket_id, noise_aug_strength, dtype,
                          batch_size, num_videos_per_prompt, do_classifier_free_guidance):
        add_time_ids = torch.tensor([[fps, motion_bucket_id, noise_aug_strength]], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)
        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])
        return add_time_ids

    def decode_latents(self, latents, num_frames, decode_chunk_size=14):
        latents = latents.flatten(0, 1)
        latents = 1 / self.vae.config.scaling_factor * latents
        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i:i+decode_chunk_size].shape[0]
            decode_kwargs = {"num_frames": num_frames_in} if accepts_num_frames else {}
            frames.append(self.vae.decode(latents[i:i+decode_chunk_size], **decode_kwargs).sample)
        frames = torch.cat(frames, dim=0)
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        return frames.float()

    def check_inputs(self, image, height, width):
        if (not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image)
                and not isinstance(image, list)):
            raise ValueError(f"`image` must be tensor/PIL/list, got {type(image)}")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"height/width must be divisible by 8, got {height}x{width}")

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height,
                        width, dtype, device, generator, latents=None):
        shape = (batch_size, num_frames, num_channels_latents // 2,
                 height // self.vae_scale_factor, width // self.vae_scale_factor)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        return latents * self.scheduler.init_noise_sigma

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1
        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        frames: torch.FloatTensor,          # (f, c, h, w) in [0,1]
        frames_mask: torch.FloatTensor,     # (f, 1, h, w) in [0,1]
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.0,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        self.check_inputs(frames, height, width)
        batch_size = 1
        device = self._execution_device
        self._guidance_scale = max_guidance_scale

        image_embeddings = self._encode_image(
            frames[0:1], device, num_videos_per_prompt, self.do_classifier_free_guidance)
        fps = fps - 1

        frames = self.image_processor.preprocess(frames, height=height, width=width)
        noise = randn_tensor(frames.shape, generator=generator, device=frames.device, dtype=frames.dtype)
        frames = frames + noise_aug_strength * noise
        frames_mask = self.mask_processor.preprocess(frames_mask, height=height, width=width)

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        frame_latents = self._encode_vae_frames(
            frames, device, num_videos_per_prompt, self.do_classifier_free_guidance
        ).to(image_embeddings.dtype)
        mask_latents = self._encode_mask_frames(
            frames_mask, device, num_videos_per_prompt, self.do_classifier_free_guidance
        ).to(image_embeddings.dtype)

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        added_time_ids = self._get_add_time_ids(
            fps, motion_bucket_id, noise_aug_strength, image_embeddings.dtype,
            batch_size, num_videos_per_prompt, self.do_classifier_free_guidance,
        ).to(device)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt, num_frames, num_channels_latents,
            height, width, image_embeddings.dtype, device, generator, latents)

        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        self._guidance_scale = guidance_scale

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, frame_latents, mask_latents], dim=2)

                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()
        if not return_dict:
            return frames
        return StableVideoDiffusionPipelineOutput(frames=frames)


# ---- resize utils (from the original implementation) ----
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])
    sigmas = (max((factors[0] - 1.0) / 2.0, 0.001), max((factors[1] - 1.0) / 2.0, 0.001))
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]
    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1
    input = _gaussian_blur2d(input, ks, sigmas)
    return torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)


def _compute_padding(kernel_size):
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]
    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        pad_front = computed_tmp // 2
        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = computed_tmp - pad_front
    return out_padding


def _filter2d(input, kernel):
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]
    padding_shape = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    return output.view(b, c, h, w)


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])
    batch_size = sigma.shape[0]
    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype, device=input.device)
    else:
        sigma = sigma.to(dtype=input.dtype, device=input.device)
    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    return _filter2d(out_x, kernel_y[..., None])

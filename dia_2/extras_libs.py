from PIL import Image
import numpy as np
import torchaudio
import torch
from huggingface_hub import hf_hub_download
import dataclasses
from typing import Optional, Union, List
import inspect
from argparse import Namespace


def slerp(
    t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995
) -> torch.Tensor:
    """
    Helper function to spherically interpolate two arrays v1 v2.
    """
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > dot_threshold:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess an image for the model.
    """
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)

    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np[None].transpose(0, 3, 1, 2)

    image_torch = torch.from_numpy(image_np)

    return 2.0 * image_torch - 1.0


def spectrogram_from_image(
    image: Image.Image,
    power: float = 0.25,
    stereo: bool = False,
    max_value: float = 30e6,
) -> np.ndarray:
    """
    Compute a spectrogram magnitude array from a spectrogram image.

    This is the inverse of image_from_spectrogram, except for discretization error from
    quantizing to uint8.

    Args:
        image: (frequency, time, channels)
        power: The power curve applied to the spectrogram
        stereo: Whether the spectrogram encodes stereo data
        max_value: The max value of the original spectrogram. In practice doesn't matter.

    Returns:
        spectrogram: (channels, frequency, time)
    """
    # Convert to RGB if single channel
    if image.mode in ("P", "L"):
        image = image.convert("RGB")

    # Flip Y
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # Munge channels into a numpy array of (channels, frequency, time)
    data = np.array(image).transpose(2, 0, 1)
    if stereo:
        # Take the G and B channels as done in image_from_spectrogram
        data = data[[1, 2], :, :]
    else:
        data = data[0:1, :, :]

    # Convert to floats
    data = data.astype(np.float32)

    # Invert
    data = 255 - data

    # Rescale to 0-1
    data = data / 255

    # Reverse the power curve
    data = np.power(data, 1 / power)

    # Rescale to max value
    data = data * max_value

    return data


def get_inverter(n_fft, num_griffin_lim_iters, win_length, hop_length, device):
    inverse_spectrogram_func = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        n_iter=num_griffin_lim_iters,
        win_length=win_length,
        hop_length=hop_length,
        window_fn=torch.hann_window,
        power=1.0,
        wkwargs=None,
        momentum=0.99,
        length=None,
        rand_init=True,
    ).to(device)
    return inverse_spectrogram_func


def audio_from_spectrogram(
    spectrogram: np.ndarray,
    params,
    device,
    apply_filters: bool = True,
    normalize: bool = True, 
):
    """
    Reconstruct an audio segment from a spectrogram.

    Args:
        spectrogram: (batch, frequency, time)
        apply_filters: Post-process with normalization and compression

    Returns:
        audio: Audio segment with channels equal to the batch dimension
    """
    # Move to device
    amplitudes_mel = torch.from_numpy(spectrogram).to(device)

    inverse_mel_scaler = torchaudio.transforms.InverseMelScale(
        n_stft=params.n_fft // 2 + 1,
        n_mels=params.num_frequencies,
        sample_rate=params.sample_rate,
        f_min=params.min_frequency,
        f_max=params.max_frequency,
        mel_scale=params.mel_scale_type,
    ).to(device)
    # Reconstruct the waveform
    amplitudes_linear = inverse_mel_scaler(amplitudes_mel)
    
    inverter_func = get_inverter(params.n_fft, params.num_griffin_lim_iters, params.win_len, params.hop_len, device)
    waveform = inverter_func(amplitudes_linear).cpu().numpy()

    # Convert to audio segment
    if normalize:
        waveform *= np.iinfo(np.int16).max / np.max(np.abs(waveform))
        
    # apply filers:
    # compression (effects normalize)
    # compress dynamic range
    # librosa.mu_compress

    return waveform


def image2audio(
    image,
    device,
    max_value: float = 30e6,
    power_for_image: float = 0.25,
    stereo: bool = False,
    sample_rate=44100,
    padded_duration_ms: int = 400,
    window_duration_ms=100,
    step_size_ms=10,
):
    spectrogram = spectrogram_from_image(
        image,
        max_value=max_value,
        power=power_for_image,
        stereo=stereo,
    )

    params = Namespace(
        sample_rate=sample_rate,
        power_for_image=power_for_image,
        stereo=stereo,
        n_fft=int(padded_duration_ms / 1000.0 * sample_rate),
        num_frequencies=512,
        min_frequency=0,
        max_frequency=10000,
        max_mel_iters=200,
        mel_scale_norm=None,
        mel_scale_type="htk",
        num_griffin_lim_iters=32,
        win_len=int(window_duration_ms / 1000.0 * sample_rate),
        hop_len=int(step_size_ms / 1000.0 * sample_rate)
    )
    segment = audio_from_spectrogram(
        spectrogram,
        params,
        device,
        apply_filters=True,
    )
    return segment


class TracedUNet(torch.nn.Module):
        @dataclasses.dataclass
        class UNet2DConditionOutput:
            sample: torch.FloatTensor

        def __init__(self, unet_file, device, dtype):
            super().__init__()
            self.in_channels = device
            self.device = device
            self.dtype = dtype
            self.unet_file = unet_file

        def forward(self, latent_model_input, t, encoder_hidden_states):
            sample = self.unet_file(latent_model_input, t, encoder_hidden_states)[0]
            return self.UNet2DConditionOutput(sample=sample)
    
    
def get_unet_traced(CACHE_DIR, device, dtype):
    unet_file = hf_hub_download(
        "riffusion/riffusion-model-v1",
        subfolder="unet_traced",
        filename="unet_traced.pt",
        cache_dir=CACHE_DIR,
    )
    unet_traced = torch.jit.load(unet_file)

    return TracedUNet(unet_traced, device, dtype)


@torch.no_grad()
def interpolate_img2img(
    pipe,
    text_embeddings: torch.Tensor,
    init_latents: torch.Tensor,
    generator_a: torch.Generator,
    generator_b: torch.Generator,
    interpolate_alpha: float,
    mask: Optional[torch.Tensor] = None,
    strength_a: float = 0.8,
    strength_b: float = 0.8,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: int = 1,
    eta: Optional[float] = 0.0,
    output_type: Optional[str] = "pil",
    **kwargs,
):

    batch_size = text_embeddings.shape[0]

    # set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    bs_embed, seq_len, _ = text_embeddings.shape
    text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
    text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0
    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance:
        if negative_prompt is None:
            uncond_tokens = [""]
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError("The length of `negative_prompt` should be equal to batch_size.")
        else:
            uncond_tokens = negative_prompt

        # max_length = text_input_ids.shape[-1]
        uncond_input = pipe.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[0]

        # duplicate unconditional embeddings for each generation per prompt
        uncond_embeddings = uncond_embeddings.repeat_interleave(
            batch_size * num_images_per_prompt, dim=0
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents_dtype = text_embeddings.dtype

    strength = (1 - interpolate_alpha) * strength_a + interpolate_alpha * strength_b

    # get the original timestep using init_timestep
    offset = pipe.scheduler.config.get("steps_offset", 0)
    init_timestep = int(num_inference_steps * strength) + offset
    init_timestep = min(init_timestep, num_inference_steps)

    timesteps = pipe.scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor(
        [timesteps] * batch_size * num_images_per_prompt, device=pipe.device
    )

    # add noise to latents using the timesteps
    noise_a = torch.randn(
        init_latents.shape, generator=generator_a, device=pipe.device, dtype=latents_dtype
    )
    noise_b = torch.randn(
        init_latents.shape, generator=generator_b, device=pipe.device, dtype=latents_dtype
    )
    noise = slerp(interpolate_alpha, noise_a, noise_b)
    init_latents_orig = init_latents
    init_latents = pipe.scheduler.add_noise(init_latents, noise, timesteps)

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same args
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    accepts_eta = "eta" in set(inspect.signature(pipe.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    latents = init_latents.clone()

    t_start = max(num_inference_steps - init_timestep + offset, 0)

    # Some schedulers like PNDM have timesteps as arrays
    # It's more optimized to move all timesteps to correct device beforehand
    timesteps = pipe.scheduler.timesteps[t_start:].to(pipe.device)

    for i, t in enumerate(pipe.progress_bar(timesteps)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        if mask is not None:
            init_latents_proper = pipe.scheduler.add_noise(
                init_latents_orig, noise, torch.tensor([t])
            )
            # import ipdb; ipdb.set_trace()
            latents = (init_latents_proper * mask) + (latents * (1 - mask))

    latents = 1.0 / 0.18215 * latents
    image = pipe.vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()

    if output_type == "pil":
        image = pipe.numpy_to_pil(image)

    return dict(images=image, latents=latents, nsfw_content_detected=False)


def forward_riffuse_pipeline(
    pipe,
    device,
    dtype,
    start,
    end,
    mask=None,
    alpha=1.0,
    num_inference_steps=50
):
    guidance_scale = start.guidance * (1.0 - alpha) + end.guidance * alpha
    generator_start = torch.Generator(device=device).manual_seed(start.seed)
    generator_end = torch.Generator(device=device).manual_seed(end.seed)
    
    def embed_text(_text):
        text_input = pipe.tokenizer(
            _text,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embed = pipe.text_encoder(text_input.input_ids.to(device))[0]
        return embed
    embed_start = embed_text(start.prompt)
    embed_end = embed_text(end.prompt)
    
    text_embedding = embed_start + alpha * (embed_end - embed_start)
    
    # Init Image
    init_image = Image.open("seed_images/og_beat.png").convert("RGB")
    init_image_torch = preprocess_image(init_image).to(device, dtype)
    init_latent_dist = pipe.vae.encode(init_image_torch).latent_dist
    
    generator = torch.Generator(device=device).manual_seed(start.seed)
    init_latents = init_latent_dist.sample(generator=generator)
    init_latents = 0.18215 * init_latents
    
    outputs = interpolate_img2img(
            pipe,
            text_embeddings=text_embedding,
            init_latents=init_latents,
            mask=mask,
            generator_a=generator_start,
            generator_b=generator_end,
            interpolate_alpha=alpha,
            strength_a=start.denoising,
            strength_b=end.denoising,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
    return outputs["images"][0]

import blobfile as bf
import io
import wandb
import numpy as np
import torchvision
import itertools
import os
import torch
from torchvision.transforms.functional import to_pil_image
import sys
from vit import sViT
sys.path.append("glid-3-xl")
from encoders.modules import BERTEmbedder

from guided_diffusion.script_util import create_gaussian_diffusion, create_model_and_diffusion, model_and_diffusion_defaults


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def load_ldm_encoder(model, requires_grad=False):
    assert os.path.exists(model), "model not found, download from https://dall-3.com/models/glid-3-xl/kl-f8.pt"
    encoder = torch.load(model, map_location="cpu")
    encoder.eval()
    set_requires_grad(encoder, requires_grad)
    del encoder.loss
    return encoder


def load_ldm_bert(device, bert_path='bert.ckpt', requires_grad=False):
    assert os.path.exists(bert_path), "bert not found, download from https://dall-3.com/models/glid-3-xl/bert.pt"
    bert = BERTEmbedder(1280, 32, device=device)
    sd = torch.load(bert_path, map_location="cpu")
    bert.load_state_dict(sd)
    bert.eval()
    set_requires_grad(bert, requires_grad)
    return bert

def diffusion_options(use_fp16, timestep_respacing, num_heads=8, attention_resolution="32,16,8"):
    options = model_and_diffusion_defaults()
    options.update({
        "attention_resolutions": "32,16,8",
        "class_cond": False,
        "diffusion_steps": 1000,
        "image_size": 32,
        "learn_sigma": False,
        "noise_schedule": "linear",
        "num_channels": 192,
        "num_heads": num_heads,
        "num_res_blocks": 2,
        "resblock_updown": False,
        "timestep_respacing": timestep_respacing,
        "use_fp16": use_fp16,
        "use_scale_shift_norm": False,
        'clip_embed_dim': 768 # TODO: remove this
    })
    return options

def set_encoder_options(context_dim=1280):
    return {
        "image_size": 32,
        "patch_size": 16,
        "num_classes": context_dim,
        "channels": 4,
        "dim": 768,
        "depth": 6,
        "heads": 16,
        "mlp_dim": 2048,
        "dropout": 0.1,
        "emb_dropout": 0.1,
    }
def load_state_dict(path, **kwargs):
    with bf.BlobFile(path, "rb") as f: data = f.read()
    return torch.load(io.BytesIO(data), **kwargs)

def load_set_encoder(model_path, context_dim=1280):
    options = set_encoder_options(context_dim=context_dim)
    encoder = sViT(**options)
    if model_path != "":
        model_state_dict = load_state_dict(model_path, map_location="cpu")
        encoder.load_state_dict(model_state_dict, strict=False)
    return encoder

def load_model_and_diffusion(model_path, use_fp16=True, num_heads=8, attention_resolution="32,16,8"):
    if model_path != "":
        assert os.path.exists(model_path), "model not found, download from https://dall-3.com/models/glid-3-xl/diffusion.pt"
        resumed_state_dict = load_state_dict(model_path, map_location="cpu")
        model_state_dict = resumed_state_dict
        for k in model_state_dict:
            if k in model_state_dict:
                if model_state_dict[k].shape != resumed_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, " f"required shape: {model_state_dict[k].shape}, " f"loaded shape: {resumed_state_dict[k].shape}")
                    model_state_dict[k] = resumed_state_dict[k]
                    if k.endswith('weight'):
                        kb = k.replace('weight','bias')
                        model_state_dict[kb] = resumed_state_dict[kb]
            else:
                print(f"Dropping parameter {k}")
    options = diffusion_options(use_fp16, timestep_respacing=str(1000), num_heads=num_heads, attention_resolution=attention_resolution)
    # options.update({ 'clip_embed_dim': 768 }) # TODO add this back
    model, diffusion = create_model_and_diffusion(**options)
    if model_path != "":
        model.load_state_dict(model_state_dict, strict=False)
    if use_fp16:
        model.convert_to_fp16()
    if model_path != "":
        del model_state_dict
    return model, diffusion


@torch.inference_mode()
def sample_diffusion(text, set_data, set_encoder, ldm, model, clip_model, custom_clip, batch_size, device, prefix="output", timestep_respacing="50", ddpm=False, guidance_scale=10.0, shape=(256, 256), save_last=True, wandb_run=None, images_per_row=8, text_embedding=None):
    if len(text) > 128:
        text = text[:128]
    # from custom_clip import clip
    sampling_options = diffusion_options(False, timestep_respacing)

    sampling_diffusion = create_gaussian_diffusion(
        steps=sampling_options["diffusion_steps"],
        learn_sigma=sampling_options["learn_sigma"],
        noise_schedule=sampling_options["noise_schedule"],
        use_kl=sampling_options["use_kl"],
        predict_xstart=sampling_options["predict_xstart"],
        rescale_timesteps=sampling_options["rescale_timesteps"],
        rescale_learned_sigmas=sampling_options["rescale_learned_sigmas"],
        timestep_respacing=timestep_respacing,
    )
    os.makedirs(prefix, exist_ok=True)
    height, width = shape
    set_b, b, c, h, w = set_data.shape
    set_data = torch.reshape(set_data, (set_b*b, c, h, w))
    encoded_set = ldm.encode(set_data)
    uncond_set_data = torch.zeros((set_b*b, c, h, w))
    uncond_encoded_set = ldm.encode(uncond_set_data)
    # text_blank = bert.encode([""]*batch_size).to(device).float()
    kwargs['context_data'] = torch.cat([encoded_set, uncond_encoded_set], dim=0).float().to(device)
    kwargs['context_fn'] = set_encoder
    
    if clip_model:
        text = custom_clip.tokenize([text]*batch_size, truncate=True).to(device)
        text_clip_blank = custom_clip.tokenize([""]*batch_size, truncate=True).to(device)
        # custom_clip context
        text_emb_clip = clip_model.encode_text(text)
        text_emb_clip_blank = clip_model.encode_text(text_clip_blank)

        kwargs = {
            "clip_embed": torch.cat([text_emb_clip, text_emb_clip_blank], dim=0).float(),
        }
    else:
        kwargs = {
            "clip_embed": torch.cat([text_embedding['clip_embed'], text_embedding['clip_embed']]),
        }
    # Create a classifier-free guidance sampling function

    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    final_step = int(timestep_respacing.replace("ddim", "")) - 1
    if timestep_respacing.startswith('ddim'):
        sample_fn = sampling_diffusion.ddim_sample_loop_progressive
    elif ddpm:
        sample_fn = sampling_diffusion.p_sample_loop_progressive
    else:
        # PLMS sampling skips the first steps
        final_step = int(timestep_respacing.replace("ddim", "")) - 3
        sample_fn = sampling_diffusion.plms_sample_loop_progressive

    samples = sample_fn(
        model_fn,
        (batch_size*2, 4, int(height/8), int(width/8)),
        clip_denoised=False,
        model_kwargs=kwargs,
        cond_fn=None,
        device=device,
        progress=True,
    )
    def decode_sample(image):
        image /= 0.18215
        im = image.unsqueeze(0)
        decoded_image = ldm.decode(im)
        return decoded_image.squeeze(0).add(1).div(2).clamp(0, 1)


    output_path = os.path.join(prefix, f"grid.png")
    for timestep_idx, sample in enumerate(samples):
        batch_gn = sample['pred_xstart'][:batch_size]
        batch = [decode_sample(out) for out in batch_gn]
        grid = torchvision.utils.make_grid(batch, nrow=images_per_row)
        if save_last and timestep_idx == final_step:
            torchvision.utils.save_image(grid, output_path)
        else:
            torchvision.utils.save_image(grid, output_path)
    return output_path
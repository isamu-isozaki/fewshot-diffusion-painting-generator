"""
Strategy

1. Train autoencoder
2. Encode set images in vit after encoding in autoencoder.
3. Add this as context-> train model from scratch.
"""

from matplotlib.style import context
import numpy as np
import argparse
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset
import wandb
from dalle_pytorch import distributed_utils
from crowsonkb.adamw_ema import AdamWEMA
import gc

# import sys
# # sys.path.append("custom_clip")
from clip_custom import clip
from latent_diffusion_deepspeed.deepspeed_config import deepspeed_config_from_args
from latent_diffusion_deepspeed.image_text_datasets import load_data
from latent_diffusion_deepspeed.model_util import (load_set_encoder,
                                                   load_ldm_encoder,
                                                   load_model_and_diffusion,
                                                   sample_diffusion)
from latent_diffusion_deepspeed.train_util import save_model

# TODO: Have an option to just encode the empty string for clip and then get rid of the clip+bert model from memory
# TODO: Have option to just train the autoencoder
# TODO: Adapt to inpainting


def get_uncond_text_embedding(batch_size, clip_model, device, use_fp16):
    with torch.cuda.amp.autocast(enabled=use_fp16):
        model_kwargs = {}

        clip_text = clip.tokenize(['']*batch_size, truncate=True).to(device)
        clip_emb = clip_model.encode_text(clip_text)

        model_kwargs["clip_embed"] = clip_emb
    return model_kwargs


def ldm_encode_data_gn(dataloader, encoder, clip_model, device, use_fp16, text_embedding=None):
    with torch.cuda.amp.autocast(enabled=use_fp16):
        for text, set_batch, batch in dataloader:
            model_kwargs = {}
            # print(f"text: {text}")
            # remove random indices from set
            set_b, b, c, h, w = set_batch.shape
            set_batch_perm = torch.randperm(b)
            set_batch = set_batch[:, set_batch_perm]
            set_batch_remove_size = int(b*.2)
            set_batch = set_batch[:, :-set_batch_remove_size]
            set_b, b, c, h, w = set_batch.shape
            set_batch = torch.reshape(set_batch, (set_b*b, c, h, w))
            set_batch = set_batch.to(device)
            set_emb = encoder.encode(set_batch).sample()
            set_emb *= 0.18215
            _, c, h, w = set_emb.shape
            set_emb = torch.reshape(set_emb, (set_b, b, c, h, w))

            model_kwargs["context_data"] = set_emb
            if clip_model:
                clip_text = clip.tokenize(text, truncate=True).to(device)
                clip_emb = clip_model.encode_text(clip_text)

                model_kwargs["clip_embed"] = clip_emb
            else:
                model_kwargs["clip_embed"] = text_embedding["clip_embed"]

            batch = batch.to(device)
            emb = encoder.encode(batch).sample()
            emb *= 0.18215
            yield emb, model_kwargs

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def train_step(model, set_encoder, diffusion, x_start, device, model_kwargs={}):
    model_kwargs["clip_embed"].to(device)
    timesteps = torch.randint(
        0, len(diffusion.betas) - 1, (x_start.shape[0],), device=device)
    scaled_timesteps = diffusion._scale_timesteps(timesteps).to(device)
    model_kwargs["context"] = set_encoder(model_kwargs['context_data'], scaled_timesteps)
    noise = torch.randn_like(x_start, device=device)
    x_t = diffusion.q_sample(x_start, timesteps, noise=noise).to(device)
    epsilon = model(x_t.to(device), scaled_timesteps.to(
        device), **model_kwargs).to(device)
    return torch.nn.functional.mse_loss(epsilon, noise.detach())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention_resolution", type=str, default="32,16,8")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--kl_model", type=str, default="kl-f8.pt")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--random_crop", action="store_true")
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--min_lr", type=float, default=1e-8)
    parser.add_argument("--set_encoder_path", type=str, default="")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--sample_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--ga_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--use_webdataset", action="store_true")
    parser.add_argument("--use_captions", action="store_true")
    parser.add_argument("--deepspeed", "-deepspeed", action="store_true")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--local_rank", "-local_rank",
                        type=int, default=0)  # stub for distributed
    parser.add_argument("--wandb_project", type=str,
                        default="latent-diffusion-deepspeed")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--inpainting", action="store_true", default=False)
    parser.add_argument("--distributed_backend", type=str, default=None)
    parser.add_argument("--set_training", action="store_true", default=False)
    parser.add_argument("--set_size", type=int, default=15)
    parser.add_argument("--target_sets", type=str, default="Chi,Alfred_Sisley,Pablo_Picasso,William_Turner")

    args = parser.parse_args()

    data_dir = args.data_dir
    if args.data_dir.startswith("s3://"):
        data_dir = f"pipe:aws s3 cp {args.data_dir} -"
        args.use_webdataset = True  # force webdataset

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Setup deepspeed distributed training
    print(f"Initializing distributed training with local rank {args.local_rank}")
    distr_backend = distributed_utils.set_backend_from_args(args)
    distr_backend.initialize()
    distr_backend.check_batch_size(args.batch_size)
    is_root_rank = distr_backend.is_local_root_worker()
    wandb_run = None
    if is_root_rank:
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                               sync_tensorboard="finetune-ldm-logs/finetune-ldm", tensorboard=True)
    
    # from clip_custom import clip # make clip end up on the right device

    print(f"Loading CLIP.")
    clip_model, _ = clip.load('ViT-L/14', device=device, jit=False)
    clip_model.eval().requires_grad_(False)

    del clip_model.visual

    # Load backbone models
    # requires a device bc bug in latent-diffusion
    print(f"Loading set encoder.")
    set_encoder = load_set_encoder(args.set_encoder_path, context_dim=1280)
    
    print(f"Loading LDM first stage encoder with local rank {args.local_rank}")
    encoder = load_ldm_encoder(args.kl_model, requires_grad=False)
    if args.use_fp16:  # Careful, this needs to be done _before_ loading the dataset
        set_encoder = set_encoder.half()
        encoder = encoder.half()
    else:
        clip_model.float() # CLIP is half precision, so we need to convert the encoder to FP32
    set_encoder.to(device)
    encoder.to(device)
    text_embedding = None
    if not args.use_captions:
        text_embedding = get_uncond_text_embedding(args.batch_size, clip_model, device, args.use_fp16)
        del clip_model
        clip_model = None
    gc.collect()
    torch.cuda.empty_cache() 

    # Load data
    print(f"Loading data with local rank {args.local_rank} from {data_dir}")
    dataset = load_data(
        distr_backend,
        data_dir,
        args.batch_size,
        args.image_size,
        dataset_length=args.max_steps,  # TODO
        random_crop=args.random_crop,
        random_flip=args.random_flip,
        use_webdataset=args.use_webdataset,
        num_workers=args.num_workers,
        inpainting=args.inpainting,
        set_data = args.set_training,
        set_size=args.set_size,
    )

    # Load the diffusion model (will be converted to fp16 if necessary)
    print(f"Loading diffusion model with local rank {args.local_rank}")
    model, diffusion = load_model_and_diffusion(
        model_path=args.resume_ckpt, use_fp16=args.use_fp16, num_heads=args.num_heads, attention_resolution=args.attention_resolution)
    model.to(device)

    print(f"Loading optimizer with local rank {args.local_rank}")
    optimizer = AdamWEMA(list(model.parameters())+list(set_encoder.parameters()), lr=args.lr, weight_decay=args.weight_decay, ema_decay=args.ema_decay, ema_power=1.)

    print(f"Distributing model with local rank {args.local_rank}")
    # Prepare pytorch vs. deepspeed optimizer, dataloader, model
    dataloader = None
    if args.deepspeed:
        deepspeed_config = deepspeed_config_from_args(args)

        (model, optimizer, dataloader, _) = distr_backend.distribute(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=[x for x in model.parameters() if x.requires_grad],
            training_data=None if args.use_webdataset else dataset,
            lr_scheduler=None, # TODO: allow for pytorch scheduler
            config_params=deepspeed_config,
        )
        # dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True) # required for non-distributed
        # wds_urls = braceexpand("pipe:aws s3 cp s3://laion-watermark/clear/{00000..00160}.tar -")
        # number_of_batches = (dataset_length // batch_size // distr_backend.get_world_size())
        # dl.length = number_of_batches
        # print(f"Loaded webdataset with {number_of_batches} batches on {distr_backend.get_world_size()} gpus")
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True) # required for non-distributed
    if args.use_webdataset:
        import webdataset as wds
        dataloader = wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=2) # TODO remove param


    print(f"Starting training with local rank {args.local_rank}")
    # Train loop
    
    for epoch in range(args.num_epochs):
        gc.collect()
        torch.cuda.empty_cache() 
        print(f"Starting epoch {epoch} with local rank {args.local_rank}")
        data = ldm_encode_data_gn(dataloader, encoder, clip_model, device, args.use_fp16, text_embedding=text_embedding)

        for i, (x_start, model_kwargs) in enumerate(data):
            with torch.cuda.amp.autocast(enabled=args.use_fp16):
                loss = train_step(model, set_encoder, diffusion, x_start,
                                  device=device, model_kwargs=model_kwargs)
            del encoder
            gc.collect()
            torch.cuda.empty_cache() 
            if args.deepspeed:
                model.backward(loss)
                model.step()
                accumulated_loss = distr_backend.average_all(loss)
            else:
                loss.backward()
                optimizer.step()
                model.zero_grad()
                accumulated_loss = loss
            encoder = load_ldm_encoder(args.kl_model, requires_grad=False)
            if args.use_fp16:  # Careful, this needs to be done _before_ loading the dataset
                encoder = encoder.half()
            
            if i % args.log_interval == 0 and is_root_rank:
                print(f"epoch {epoch} step {i} loss {accumulated_loss.item()}")

            if i % args.sample_interval == 0 and is_root_rank:
                # For inference 2 things
                # 1. Pass set_data specific to chi.
                # 2. Pass set_data specific to another random artist
                gc.collect()
                torch.cuda.empty_cache()
                log_dict = {}
                for target_set in args.target_sets.split(','):
                    target_set_data = torch.Tensor(dataset.get_set_imgs(target_set, ''))
                    target_paintings = sample_diffusion(text="", set_data=target_set_data, set_encoder=set_encoder, ldm=encoder, model=model, clip_model=clip_model, custom_clip=clip, batch_size=4, prefix="finetune-samples", device=device,
                                                        timestep_respacing="30", ddpm=False, guidance_scale=1.0, shape=(256, 256), save_last=True, wandb_run=wandb_run, images_per_row=4, text_embedding=text_embedding)
                    log_dict[target_set] = wandb.Image(target_paintings)
                    gc.collect()
                    torch.cuda.empty_cache() 
                if wandb_run is not None:
                    wandb_run.log(log_dict)
            if i % args.save_interval == 0: #TODO
                save_model(model=model, path=args.log_dir, is_root=is_root_rank,
                            epoch=epoch, using_deepspeed=args.deepspeed, opt=optimizer)
                print(f"saved model to {args.log_dir} at epoch {epoch} step {i}")

    save_model(model=model, path=args.log_dir, is_root=is_root_rank,
               epoch=epoch, using_deepspeed=args.deepspeed, opt=optimizer)
    print(f"Finished training. saved model to {args.log_dir} at epoch {epoch} step {i}")


if __name__ == "__main__":
    main()

from dataclasses import dataclass
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
import argparse
import wandb
from datetime import datetime
import pytz
import torch
import torch.nn.functional as F
import datasets
from datasets import load_dataset
from torchvision import transforms
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler

import deepspeed
from deepspeed.accelerator import get_accelerator

datasets.config.HF_DATASETS_CACHE=Path('/home/ds_debug/data')
datasets.config.DOWNLOADED_DATASETS_PATH = Path('/home/ds_debug/data')

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1, help="local rank for distributed training on gpus")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

if args.local_rank == -1:
    device = torch.device("cuda")
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    deepspeed.init_distributed()

def is_main_process():
    if get_accelerator().device_count() == 1:
        return True

    # if deepspeed.comm.get_global_rank() == 0:
    if get_accelerator().current_device() == 0:
        return True
    return False

def dic_to(dic, device):
    for key, value in dic.items():
        dic[key] = dic[key].to(device)
    return dic

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    num_workers = 1
    train_batch_size = 1
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 3
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    save_step = 5
    # mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    mixed_precision = "bf16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    # push_to_hub = False  # whether to upload the saved model to the HF Hub
    # hub_model_id = "chengming-zhang/debug-diffu"  # the name of the repository to create on the HF Hub
    # hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 66
    zero_stage = 2

config = TrainingConfig()

# torch.manual_seed(config.seed)

dtype = torch.float32
if config.mixed_precision == "fp16":
    dtype = torch.float16
elif config.mixed_precision == "bf16":
    dtype = torch.bfloat16

# config.dataset_name = "huggan/smithsonian_butterflies_subset"
config.dataset_name = "huggan/inat_butterflies_top10k"
dataset = load_dataset(config.dataset_name, split="train")
preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}
dataset.set_transform(transform)
if args.local_rank != -1:
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
else:
    train_sampler = None
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=(train_sampler is None), num_workers=config.num_workers, sampler=train_sampler)

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    # block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block 113M
    # block_out_channels=(256, 256, 512, 512, 1024, 1024),  # the number of output channels for each UNet block 454M
    block_out_channels=(512, 512, 1024, 1024, 1024, 1024),  # the number of output channels for each UNet block 771M
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)
num_p = count_parameters(model)
print(f'------ Total parameters {num_p / (10**6)} M ------')

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def get_ds_config(cfg):
    """Get the DeepSpeed configuration dictionary."""
    ds_config = {
        "train_batch_size": cfg.train_batch_size*cfg.gradient_accumulation_steps*get_accelerator().device_count(),
        "train_micro_batch_size_per_gpu": cfg.train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "steps_per_print": 2000,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "bf16": {"enabled": cfg.mixed_precision == "bf16"},
        "fp16": {
            "enabled": cfg.mixed_precision == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15,
        },
        "checkpoint": {
            "load_universal": False,
        },
        "communication_data_type": config.mixed_precision,
        "data_types": {"grad_accum_dtype": "fp32"},
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": cfg.zero_stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 500000000,
            "reduce_bucket_size":1000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False,
        },
    }
    return ds_config

new_york_t = pytz.timezone("America/New_York") 
if is_main_process():
    timestamp = datetime.now(new_york_t)
    run = wandb.init(
        # Set the project where this run will be logged
        project="debug-diffu",
        entity="xxx",
        name=f"run-zero-{config.zero_stage}-10k-batch{config.train_batch_size}-bucket1e6-accumfp32-{timestamp.minute}"
        # name=f"run-zero-{config.zero_stage}-ts-{timestamp.day}-{timestamp.hour}-{timestamp.minute}"
        # Track hyperparameters and run metadata
        # config={
        #     "learning_rate": lr,
        #     "epochs": epochs,
        # },
    )

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    if is_main_process():
        pass
        # if config.output_dir is not None:
        #     os.makedirs(config.output_dir, exist_ok=True)
        # if config.push_to_hub:
        #     repo_id = create_repo(
        #         repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
        #     ).repo_id

    model = model.to(get_accelerator().current_device_name(), dtype=dtype)
    parameters = model.parameters()
    ds_config = get_ds_config(config)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args, model=model, optimizer=optimizer, model_parameters=parameters, lr_scheduler=lr_scheduler, config=ds_config, dist_init_required=True)

    global_step = 0
    train_loss = 0.0
    for epoch in range(config.num_epochs):
        # progress_bar = tqdm(total=len(train_dataloader), disable=not is_main_process())
        # progress_bar.set_description(f"Epoch {epoch}")
        if is_main_process():
            print(f'------ current epoch {epoch} ------')
        for step, batch in enumerate(train_dataloader):
            batch = dic_to(batch, get_accelerator().current_device_name())
            clean_images = batch["images"].to(dtype)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device, dtype=dtype)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            model.backward(loss)
            model.step()

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = loss.repeat(get_accelerator().device_count(), config.train_batch_size)
            deepspeed.comm.all_gather_into_tensor(avg_loss, loss.repeat(config.train_batch_size))
            avg_loss = avg_loss.mean()
            train_loss += avg_loss.item() / config.gradient_accumulation_steps

            # progress_bar.update(1)
            # logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if is_main_process() and (global_step % config.save_step == 0):
                wandb.log({"avg_loss": avg_loss.detach().item(), "loss": loss.detach().item()}, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        # pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
        # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
        #     evaluate(config, epoch, pipeline)

        # if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
        #     if config.push_to_hub:
        #         upload_folder(
        #             repo_id=repo_id,
        #             folder_path=config.output_dir,
        #             commit_message=f"Epoch {epoch}",
        #             ignore_patterns=["step_*", "epoch_*"],
        #         )
        #     else:
        #         pipeline.save_pretrained(config.output_dir)

if __name__ == '__main__':
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

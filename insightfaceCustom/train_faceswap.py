import argparse
import logging
import os
import random
from datetime import datetime

import cv2
import numpy as np
import torch
import lpips
from backbones import get_model
from dataset import get_dataloader
from lr_scheduler import PolynomialLRWarmup
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        # Sign in to wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        # Initialize wandb
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                sync_tensorboard=True,
                resume=cfg.wandb_resume,
                name=run_name,
                notes=cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")
    
    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers
    )

    # Loading arcface model for face embeddings only
    arcface_model = get_model(
        cfg.afnetwork, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    arcface_model = torch.nn.parallel.DistributedDataParallel(
        module=arcface_model, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=False)
    arcface_model.register_comm_hook(None, fp16_compress_hook)
    arcface_model.eval()
    for p in arcface_model.module.parameters():
        p.requires_grad = False

    if cfg.resume:
        backbone_path = os.path.join(cfg.output, "ms1mv3_arcface_r50_fp16_backbone.pth")
        if os.path.exists(backbone_path):
            arcface_model.module.load_state_dict(torch.load(backbone_path, weights_only=True))
            print("===================================================")
            print(f"{backbone_path} LOADED correctly for arcface_model")
            print("===================================================")
        else:
            raise FileNotFoundError(f"Resume enabled but backbone file not found: {backbone_path}")

    # Loading faceswap model for training
    faceswap_model = get_model(
        cfg.fsnetwork, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    faceswap_model = torch.nn.parallel.DistributedDataParallel(
        module=faceswap_model, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=False)
    faceswap_model.register_comm_hook(None, fp16_compress_hook)
    faceswap_model.train()

    faceswap_model._set_static_graph()

    # pos_weight = torch.ones([512])
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.cuda())

    pixel_criterion = lpips.LPIPS(net='vgg').cuda()
    for p in pixel_criterion.parameters():
        p.requires_grad = False

    if cfg.optimizer == "sgd":
        opt_fs = torch.optim.SGD(
            params=faceswap_model.parameters(),
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        opt_fs = torch.optim.AdamW(
            params=faceswap_model.parameters(),
            lr=cfg.lr, 
            weight_decay=cfg.weight_decay,
            eps=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler_fs = PolynomialLRWarmup(
        optimizer=opt_fs,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    start_epoch = 0
    global_step = 0

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec,
        summary_writer=summary_writer, wandb_logger=wandb_logger
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer
    )

    loss_fs_am = AverageMeter()
    fs_amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    # Training
    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)

        for _, (imgs, local_labels) in enumerate(train_loader):
            global_step += 1

            # FaceSwap training
            with torch.no_grad():
                local_embeddings = arcface_model(imgs)
            
            fs_inputs = [imgs, local_embeddings]
            fs_fakes = faceswap_model(*fs_inputs)
            # print(f"Check imgs value range: [{imgs.min().item():.3f}, {imgs.max().item():.3f}]")
            # print(f"Check fs_fakes value range: [{fs_fakes.min().item():.3f}, {fs_fakes.max().item():.3f}]")
            # import sys; sys.exit()  # Stop the script.  We simply want to know the values.

            norm_fakes = (fs_fakes * 2.0) - 1.0
            
            fs_embeddings = arcface_model(norm_fakes)

            cos_sim = torch.nn.functional.cosine_similarity(fs_embeddings.float(), local_embeddings.float(), eps=1e-6)
            id_loss = 1.0 - cos_sim.mean()

            pixel_loss = pixel_criterion(norm_fakes.float(), imgs.float()).mean()

            fs_loss = id_loss + (10.0 * pixel_loss)

            if cfg.fp16:
                fs_amp.scale(fs_loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    fs_amp.unscale_(opt_fs)
                    torch.nn.utils.clip_grad_norm_(faceswap_model.parameters(), 1.0)
                    fs_amp.step(opt_fs)
                    fs_amp.update()
                    opt_fs.zero_grad()
            else:
                fs_loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(faceswap_model.parameters(), 1.0)
                    opt_fs.step()
                    opt_fs.zero_grad()
            lr_scheduler_fs.step()

            with torch.no_grad():
                if wandb_logger:
                    wandb_logger.log({
                        'FaceSwap Loss/Step Loss': fs_loss.item(),
                        'FaceSwap Loss/Train Loss': loss_fs_am.avg,
                        'FaceSwap Process/Step': global_step,
                        'FaceSwap Process/Epoch': epoch
                    })

                loss_fs_am.update(fs_loss.item(), 1)
                callback_logging(global_step, loss_fs_am, epoch, cfg.fp16, lr_scheduler_fs.get_last_lr()[0], fs_amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, faceswap_model)

        # Save checkpoint after each epoch
        if rank == 0:
            path_faceswap = os.path.join(cfg.output, "faceswap_model.pt")
            torch.save(faceswap_model.module.state_dict(), path_faceswap)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"faceswap_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_faceswap)
                wandb_logger.log_artifact(model)

        if cfg.dali:
            train_loader.reset()

    # Save final model
    if rank == 0:
        path_faceswap = os.path.join(cfg.output, "faceswap_model.pt")
        torch.save(faceswap_model.module.state_dict(), path_faceswap)

        if wandb_logger and cfg.save_artifacts:
            artifact_name = "faceswap_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_faceswap)
            wandb_logger.log_artifact(model)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="FaceSwap Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())

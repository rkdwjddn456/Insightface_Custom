import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

import argparse
import os
import onnxruntime
from torch2onnx import convert_onnx_main
from evaluation_metrics import main_evaluation_metrics

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
        init_method="tcp://127.0.0.1:12589",
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
    writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
    
    
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
                entity = cfg.wandb_entity, 
                project = cfg.wandb_project, 
                sync_tensorboard = True,
                resume=cfg.wandb_resume,
                name = run_name, 
                notes = cfg.notes) if rank == 0 or cfg.wandb_log_all else None
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

    student_backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    student_backbone = torch.nn.parallel.DistributedDataParallel(
        module=student_backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    
    student_backbone.register_comm_hook(None, fp16_compress_hook)
    
    student_backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    student_backbone._set_static_graph()

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": student_backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": student_backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    print(world_size)
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        # dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"model_{rank}.pt"))
        # start_epoch = dict_checkpoint["epoch"]
        # global_step = dict_checkpoint["global_step"]
        # student_backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        student_backbone.module.load_state_dict(dict_checkpoint)
        # module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        # opt.load_state_dict(dict_checkpoint["state_optimizer"])
        # lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, 
        summary_writer=summary_writer, wandb_logger = wandb_logger
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    def distillation_loss(student_output, teacher_output, teacher_norm, student_norm):
        distill_mse = F.mse_loss(student_output/student_norm, teacher_output/teacher_norm)
        
        return distill_mse
    
    ort_session = onnxruntime.InferenceSession(cfg.teacher_model_path, providers=['CUDAExecutionProvider'])

    for epoch in range(start_epoch, cfg.num_epoch):
        
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1

            local_embeddings = student_backbone(img)

            student_norm = torch.norm(local_embeddings)
            
            ort_inputs = {ort_session.get_inputs()[0].name: img.cpu().numpy()}

            teacher_outputs = ort_session.run(None, ort_inputs)
            teacher_outputs = np.array(teacher_outputs)

            teacher_embeddings = torch.from_numpy(teacher_outputs).to(student_norm.device).squeeze(0)
            teacher_norm = torch.norm(teacher_embeddings)

            kd_loss = distillation_loss(local_embeddings, teacher_embeddings, teacher_norm, student_norm)
            
            classification_loss = module_partial_fc(local_embeddings, local_labels)

            loss = classification_loss + cfg.kd_weight * kd_loss

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(student_backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(student_backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Process/Step': global_step,
                        'Process/Epoch': epoch
                    })
                    
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, student_backbone)

                    if rank == 0:
                        path_module = os.path.join(cfg.output, str(global_step) + "_model.pt")
                        onnx_module = os.path.join(cfg.output, str(global_step) + "_model.onnx")
                        save_result_path = os.path.join(cfg.output, str(global_step))
                        torch.save(student_backbone.module.state_dict(), path_module)

                        convert_onnx_main(input=path_module, output = onnx_module, network="mbf")        

                        average_f1_score, average_roc_auc = main_evaluation_metrics(onnx_module, save_result_path)

                        if rank == 0:        
                            writer.add_scalar("average_roc_auc", average_roc_auc, global_step)
                            writer.add_scalar("average_f1_score", average_f1_score, global_step)
                            writer.add_scalar("cfg.kd_weight * kd_loss", cfg.kd_weight * kd_loss, global_step)
                            writer.add_scalar("classification_loss", classification_loss, global_step)

                        student_backbone.train()

                    if cfg.save_all_states:
                        checkpoint = {
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "state_dict_backbone": student_backbone.module.state_dict(),
                            "state_dict_softmax_fc": module_partial_fc.state_dict(),
                            "state_optimizer": opt.state_dict(),
                            "state_lr_scheduler": lr_scheduler.state_dict()
                        }
                        torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

                    if rank == 0:
                        path_module = os.path.join(cfg.output, "model.pt")
                        torch.save(student_backbone.module.state_dict(), path_module)

                        if wandb_logger and cfg.save_artifacts:
                            artifact_name = f"{run_name}_E{epoch}"
                            model = wandb.Artifact(artifact_name, type='model')
                            model.add_file(path_module)
                            wandb_logger.log_artifact(model)        
                
        if cfg.dali:
            train_loader.reset()
    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(student_backbone.module.state_dict(), path_module)
        
        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
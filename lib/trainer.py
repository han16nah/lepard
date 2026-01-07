import gc
import os

import torch
import torch.nn as nn
import numpy as np

from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.timer import AverageMeter
from lib.utils import Logger, validate_gradient, check_gradients
from lib.tictok import Timers
try:
    from torch.amp import autocast, GradScaler
    autocast_kwargs = {'device_type': 'cuda', 'dtype': torch.float16}
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    autocast_kwargs = {'dtype': torch.float16}


class Trainer(object):
    def __init__(self, args):
        self.config = args
        # parameters
        self.start_epoch = 1
        self.max_epoch = args.max_epoch
        self.save_dir = args.save_dir
        self.device = args.device
        self.verbose = args.verbose

        self.model = args.model
        self.model = self.model.to(self.device)


        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_freq = args.scheduler_freq
        self.snapshot_dir = args.snapshot_dir

        self.iter_size = args.iter_size
        self.verbose_freq = args.verbose_freq // args.batch_size + 1
        if 'overfit' in self.config.exp_dir:
            self.verbose_freq = 1
        self.loss = args.desc_loss
        self.grad_invalid_count = 0
        self.grad_norm = None
        self.max_grad_norm = getattr(self.config, 'max_grad_norm', 1.0)
        self.grad_clip_enabled = getattr(self.config, 'grad_clip_enabled', True)
        self.epoch_unstable = False
        self.last_stable_ckpt = None
        self.lr_multiplier = 1.0

        self.best_loss = 1e5
        self.best_recall = -1e5
        self.summary_writer = SummaryWriter(log_dir=args.tboard_dir)
        self.logger = Logger(args.snapshot_dir)
        self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()]) / 1000000.} M\n')

        if args.finetune is True:
            self.finetune = True
        else:
            self.finetune = False
        if (args.pretrain != ''):
            self._load_pretrain(args.pretrain)
            self.last_stable_ckpt = args.pretrain


        self.loader = dict()
        self.loader['train'] = args.train_loader
        self.loader['val'] = args.val_loader
        self.loader['test'] = args.test_loader
        self.scaler = GradScaler()

        self.timers = args.timers
        if args.val_loader_full is not None:
            self.loader['val_full'] = args.val_loader_full
        else:
            self.loader['val_full'] = None


        with open(f'{args.snapshot_dir}/model', 'w') as f:
            f.write(str(self.model))
        f.close()

    def _snapshot(self, epoch, name=None): 
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'best_recall': self.best_recall
        }
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
            self.last_stable_ckpt = filename
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')
        self.logger.write(f"Save model to {filename}\n")
        torch.save(state, filename, _use_new_zipfile_serialization=False)

    def _load_pretrain(self, resume):
        print ("loading pretrained", resume)
        if os.path.isfile(resume):
            state = torch.load(resume, weights_only=True)
            self.model.load_state_dict(state['state_dict'], strict=False)
            self.start_epoch = state['epoch']
            self.max_epoch += self.start_epoch  # start counting from the loaded epoch
            if not self.finetune:
                self.scheduler.load_state_dict(state['scheduler'])
                self.optimizer.load_state_dict(state['optimizer'])
                self.best_loss = state['best_loss']
                self.best_recall = state['best_recall']

            self.logger.write(f'Successfully load pretrained model from {resume}!\n')
            if not self.finetune:
                self.logger.write(f'Current best loss {self.best_loss}\n')
                self.logger.write(f'Current best recall {self.best_recall}\n')
        else:
            raise ValueError(f"=> no checkpoint found at '{resume}'")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']

    def set_trainable_parameters(self, epoch):
        if self.config.pretrain == '' or not self.finetune:
            return  # training from scratch/resume training: all parameters trainable

        # Fine tuning: Stage 1: freeze entire backbone
        if epoch <= self.start_epoch + 5:
            for p in self.model.backbone.parameters():
                p.requires_grad = False
        
        # Stage 2: unfreeze decoder + last encoder block
        elif epoch <= self.start_epoch + 7:
            for name, p in self.model.backbone.named_parameters():
                if (
                    "decoder_blocks" in name or
                    "encoder_blocks.3" in name   # last block (adjust index)
                ):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            # modify gradient clipping
            self.max_grad_norm = 0.5

        # Stage 3: unfreeze all except first KPConv block
        else:
            for name, p in self.model.backbone.named_parameters():
                if "encoder_blocks.0" in name:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            # modify gradient clipping
            self.max_grad_norm = 0.3

    def inference_one_batch(self, inputs, phase):
        assert phase in ['train', 'val', 'test']
        inputs ['phase'] = phase


        if (phase == 'train'):
            with autocast(**autocast_kwargs):
                self.model.train()
                if self.timers: self.timers.tic('forward pass')
                data = self.model(inputs, timers=self.timers)  # [N1, C1], [N2, C2]
                if self.timers: self.timers.toc('forward pass')
                

                # NaN / Inf guard
                conf = data.get('conf_matrix_pred', None)
                if conf is not None and not torch.isfinite(conf).all():
                    self.logger.write("⚠️ NaN/Inf in conf_matrix_pred — skipping batch")
                    return None

                if self.timers: self.timers.tic('compute loss')
                loss_info = self.loss(data)
                true_loss = loss_info['loss']
                scaled_loss = true_loss / self.iter_size

                loss_info['true_loss'] = true_loss
                loss_info['scaled_loss'] = scaled_loss
                if self.timers: self.timers.toc('compute loss')
            
            if self.timers: self.timers.tic('backprop')
            self.scaler.scale(scaled_loss).backward()
            if self.timers: self.timers.toc('backprop')


        else:
            self.model.eval()
            with torch.no_grad():
                data = self.model(inputs, timers=self.timers)  # [N1, C1], [N2, C2]
                loss_info = self.loss(data)


        return loss_info


    def inference_one_epoch(self, epoch, phase):
        gc.collect()
        assert phase in ['train', 'val', 'test']
        self.grad_invalid_count = 0

        # init stats meter
        stats_meter = None #  self.stats_meter()

        num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size) # drop last incomplete batch
        c_loader_iter = self.loader[phase].__iter__()
        

        self.optimizer.zero_grad()
        self.set_trainable_parameters(epoch)

        # Rebuild optimizer ONLY when stage changes (or after instability)
        if self.config.finetune:
            if epoch == self.start_epoch or epoch == self.start_epoch + 5 or epoch == self.start_epoch + 7 or self.epoch_unstable:
                self.build_optimizer()
        
        self.epoch_unstable = False
        for c_iter in tqdm(range(num_iter)):  # loop through this epoch

            if self.timers: self.timers.tic('one_iteration')

            ##################################
            if self.timers: self.timers.tic('load batch')
            inputs = next(c_loader_iter)
            # for gpu_div_i, _ in enumerate(inputs):
            for k, v in inputs.items():
                if type(v) == list:
                    inputs [k] = [item.to(self.device) for item in v]
                elif type(v) in [ dict, float, int, type(None), np.ndarray]:
                    pass
                else:
                    inputs [k] = v.to(self.device)
            if self.timers: self.timers.toc('load batch')
            ##################################


            if self.timers: self.timers.tic('inference_one_batch')
            loss_info = self.inference_one_batch(inputs, phase)
            if loss_info is None:
                self.logger.write("Skipping batch due to NaNs in conf_matrix_pred")
                self.optimizer.zero_grad(set_to_none=True)
                continue  # skip this batch due to NaN/Inf
            if self.timers: self.timers.toc('inference_one_batch')


            ###################################################
            # run optimisation
            # if self.timers: self.timers.tic('run optimisation')
            if (max(c_iter, 1) % self.iter_size == 0 and phase == 'train'):
                gradient_valid = validate_gradient(self.model)
                grad_norm = check_gradients(self.model)
                self.grad_norm = grad_norm
                if (gradient_valid):
                    # For AMP: unscale gradients first
                    self.scaler.unscale_(self.optimizer)

                    # Clip gradients before optimizer step
                    if self.grad_clip_enabled:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            max_norm=self.max_grad_norm,
                            norm_type=2
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # self.logger.write('gradient not valid\n')
                    self.grad_invalid_count += 1
                self.optimizer.zero_grad(set_to_none=True)
            # if self.timers: self.timers.toc('run optimisation')
            ###############################

            if stats_meter is None:
                stats_meter = dict()
                for key, _ in loss_info.items():
                    stats_meter[key] = AverageMeter()
            for key, value in loss_info.items():
                if torch.is_tensor(value):
                    value = value.detach().cpu().item()  # .item() if it's a scalar loss
                stats_meter[key].update(value)

            if phase == 'train' :
                if max(c_iter, 1) % self.verbose_freq == 0 and self.verbose  :
                    curr_iter = num_iter * (epoch - 1) + c_iter
                    for key, value in stats_meter.items():
                        self.summary_writer.add_scalar(f'{phase}/{key}', value.avg, curr_iter)

                    dump_mess=True
                    if dump_mess:
                        message = f'{phase} Epoch: {epoch} [{c_iter + 1:4d}/{num_iter}]'
                        for key, value in stats_meter.items():
                            message += f'{key}: {value.avg:.2f}\t'
                        self.logger.write(message + '\n')
                        invalid_rate = self.grad_invalid_count / max(c_iter, 1)
                        self.logger.write(f'Gradient invalid count: {self.grad_invalid_count}\tInvalid rate: {invalid_rate:.4f}\tGrad Norm: {self.grad_norm}\n')
                        if invalid_rate > 0.5:
                            self.logger.write('All gradients invalid, stopping this epoch early.\n')
                            self.epoch_unstable = True
                            break

            if self.timers: self.timers.toc('one_iteration')

        if self.epoch_unstable:
            self.logger.write("Reloading model from last snapshot due to instability.\n")
            self._load_pretrain(self.last_stable_ckpt)

            # and reduce learning rate
            self.lr_multiplier *= 0.5
            
            self.build_optimizer()

        # report evaluation score at end of each epoch
        elif phase in ['val', 'test']:
            for key, value in stats_meter.items():
                self.summary_writer.add_scalar(f'{phase}/{key}', value.avg, epoch)
            if epoch % 3 == 0 and 'val_full' in self.loader and phase == 'val':
                self.test_val_full()

        message = f'{phase} Epoch: {epoch}'
        for key, value in stats_meter.items():
            message += f'{key}: {value.avg:.2f}\t'
        self.logger.write(message + '\n')

        return stats_meter

    def build_optimizer(self):
        backbone_params = []
        head_params = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(p)
            else:
                head_params.append(p)
        # hardcoded learning rates for finetuning
        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": 1e-5*self.lr_multiplier},
                {"params": head_params, "lr": 1e-4*self.lr_multiplier},
            ],
            weight_decay=1e-4
        )

        self.scaler = GradScaler()


    def train(self):
        print('start training...')
        print(f'Training for {self.max_epoch} epochs. Start from epoch {self.start_epoch}.')
        for epoch in range(self.start_epoch, self.max_epoch):
            print(f'epoch {epoch}/{self.max_epoch} : ')
            with torch.autograd.set_detect_anomaly(False):  # True
                if self.timers: self.timers.tic('run one epoch')
                stats_meter = self.inference_one_epoch(epoch, 'train')
                if self.timers: self.timers.toc('run one epoch')

            if self.scheduler is not None and not self.epoch_unstable:
                self.scheduler.step()


            if  'overfit' in self.config.exp_dir :
                if stats_meter['loss'].avg < self.best_loss:
                    self.best_loss = stats_meter['loss'].avg
                    self._snapshot(epoch, 'best_loss')

                if self.timers: self.timers.print()

            else : # no validation step for overfitting
                
                # validation and saving only when epoch not unstable
                if self.config.do_valid and not self.epoch_unstable:
                    stats_meter = self.inference_one_epoch(epoch, 'val')
                    if stats_meter['loss'].avg < self.best_loss:
                        self.best_loss = stats_meter['loss'].avg
                        self._snapshot(epoch, 'best_loss')
                    self._snapshot(epoch)


                if self.timers: self.timers.print()

        # finish all epoch
        print("Training finish!")
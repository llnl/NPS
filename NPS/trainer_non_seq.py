import os, sys, time
from . import utility

import torch
# from torch.autograd import Variable
#from tqdm import tqdm
import numpy as np
from NPS_common.utils import a1line
from NPS_common.io_utils import save_traj


def make_trainer(args, loader, model, loss, checkpoint):
    return TrainerNonSequential(args, loader, model, loss, checkpoint)

def _get_attr(x, key):
    return x.get(key, None) if (key and hasattr(x, "get")) else (x[..., key] if isinstance(x, torch.Tensor) and isinstance(key, slice) else x)

from NPS.trainer import Trainer
class TrainerNonSequential(Trainer):
    def evaluate(self, predict_only, epoch=0):
        t0 = time.time()
        args = self.args
        mse_detail = []
        mae_detail = []
        losses = []
        self.model.eval()
        n_pd = 0
        pd_all = []
        gt_all = []
        symbols_all = []
        with torch.no_grad():
            for i, x in enumerate(self.loader_test):
                if args.nstep_valid >= 0 and i >= args.nstep_valid: break
                if not predict_only:
                    gt = _get_attr(x, self.args.key_out_gt) #x[self.args.node_y].cpu()
                    if isinstance(gt, torch.Tensor): gt = gt.detach().cpu()
                    if self.args.channel_first:
                        gt = utility.to_channellast(gt, self.dim)
                pd, loss_item = self.evaluate_batch(x.to(self.device))#.detach().cpu()
                if isinstance(pd, torch.Tensor): pd = pd.detach().cpu()
                if self.args.channel_first:
                    pd = utility.to_channellast(pd, self.dim)
                n_pd += len(pd)
                pd_all.append(pd)
                if args.suffix_out in ('xyz', 'extxyz') and (not isinstance(pd, (np.ndarray,torch.Tensor))):
                    symbols_all.extend(x.numbers)
                if not predict_only:
                    gt_all.append(gt)
                    if isinstance(pd, torch.Tensor) and isinstance(gt, torch.Tensor):
                        if len(self.args.spatial_dims) > 0:
                            mse_detail.append(torch.mean((pd-gt)**2, dim=self.args.spatial_dims, keepdim=False))
                            mae_detail.append(torch.mean(torch.abs(pd-gt), dim=self.args.spatial_dims, keepdim=False))
                        else:
                            mse_detail.append((pd-gt)**2)
                            mae_detail.append(torch.abs(pd-gt))
                    losses.append([self.loss(pd, gt)] if ((loss_item is None) or (loss_item==[])) else loss_item)

        if isinstance(pd, torch.Tensor):
            pd_all = torch.cat(pd_all)
        else:
            pd_all = [x for l in pd_all for x in l]
        try:
            pd_all_size = pd_all.shape
        except:
            pd_all_size = len(pd_all)
        if args.n_traj_out > 0:
            pd_all = pd_all[:args.n_traj_out]
        the_epoch = epoch if args.epoch_in_pd_file else ''
        save_traj(f'{args.dir}/{args.file_out}pd_{the_epoch}_{len(pd_all)}.{args.suffix_out}', pd_all)
        print(f'Predicted data of size {pd_all_size} time {time.time()-t0:7.3e}', flush=True)
        if predict_only:
            return
        else:
            if gt_all[-1] is None:
                return np.mean(losses), 0, 0
            elif isinstance(gt_all[-1], torch.Tensor):
                gt_all = torch.cat(gt_all)
            else:
                gt_all = [x for l in gt_all for x in l]
            if args.n_traj_out > 0:
                gt_all = gt_all[:args.n_traj_out]
            save_traj(f'{args.dir}/{args.file_out}gt.{args.suffix_out}', gt_all)
            if mse_detail:
                try:
                    mse_detail = np.concatenate(mse_detail, 0)
                    mae_detail = np.concatenate(mae_detail, 0)
                except:
                    mse_detail = np.array(mse_detail)[:,None]
                    mae_detail = np.array(mae_detail)[:,None]
                mse, mae = np.mean(mse_detail), np.mean(mae_detail)
                try:
                    print(f'valid per channel/seq: mse {a1line(np.mean(mse_detail,axis=(0,)))} / {a1line(np.mean(mse_detail,axis=(1,)))}')
                    print(f'valid per channel/seq: mae {a1line(np.mean(mae_detail,axis=(0,)))} / {a1line(np.mean(mae_detail,axis=(1,)))}')
                except:
                    print(f'valid per channel: mse {a1line(np.mean(mse_detail,axis=(0,)))}')
                    print(f'valid per channel: mae {a1line(np.mean(mae_detail,axis=(0,)))}')
            else:
                mse = 0; mae = 0
            if args.suffix_out == 'xyz' and isinstance(pd_all[0], torch.Tensor):
                if not predict_only:
                    save_traj(f'{self.args.dir}/{args.file_out}gt.xyz', gt_all, symbols=symbols_all, CoM=True)
                save_traj(f'{self.args.dir}/{args.file_out}pd.xyz', pd_all, symbols=symbols_all, CoM=True)
            return np.mean(losses, axis=0).tolist()+ [mse, mae]

    def train_batch(self, x, epoch=1, is_train=True):
        args = self.args
        if is_train: self.optimizer.zero_grad()
        # y = self.model(x)
        # if len(y)==2 and y[1] is None: y=y[0] # silly backward compatibility fix when model return x(t+1), None
        # loss = self.loss(y, x[self.args.node_y])
        y, loss, loss_item = self.model_y_loss(x, target=_get_attr(x,args.node_y), loss_from_model=args.loss_from_model)
        if is_train: loss.backward()
        if is_train: self.training_callback()
        if is_train and (not self.args.scheduler_at_epoch):
            if self.args.scheduler == 'plateau':
                print(f"warning: for training with reduce-on-plateau, set scheduler_at_epoch=1")
                self.scheduler.step(-1.)
            else:
                self.scheduler.step()
            try:
                current_lr = self.scheduler.get_last_lr()[0]  # get_last_lr() returns a list
                print(f"    LR: {current_lr:.3g}")
            except:
                pass
        return loss.item(), loss_item

    def evaluate_batch(self, x):
        args = self.args
        y, loss, loss_item = self.model_y_loss(x, target=_get_attr(x,args.node_y), loss_from_model=args.loss_from_model)
        return y, [loss.item()]+list(loss_item)

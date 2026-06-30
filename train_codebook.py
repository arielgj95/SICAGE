import pdb

import logging
logging.getLogger().setLevel(logging.INFO)

from torch.utils.data import DataLoader
from dataset import create_subject_dependent_split, load_samples
import time
import torch
import torch.nn as nn
import yaml
from pprint import pprint
from easydict import EasyDict
from vq_vae.configs.parse_args import parse_args
from vq_vae.vqvae import VQVAE
import os
import pickle
from torch import optim
import itertools
from datetime import datetime
import distutils
import distutils.version
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


args = parse_args()
mydevice = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

def set_requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag

# evaluation of the val set at the end of each epoch

def evaluate_valset(model, val_data_loader):
    start = time.time()
    model = model.eval()
    euclidean_errors = []

    with torch.no_grad():
        for iter_idx, data in enumerate(val_data_loader, 0):
            pose_seq_eval = data[0].to(mydevice)    # （batch, 60, 54）
            labels = data[1]                        # Not needed

            b, t, c = pose_seq_eval.size()
            output, loss, _ = model(pose_seq_eval)
            diff = (pose_seq_eval - output).view(b, t, c // 6, 6)
            euclidean_errors.append(torch.mean(torch.sqrt(torch.sum(diff ** 2, dim=3))))

    print('generation took {:.2f} s'.format(time.time() - start))
    model.train()

    euclidean_errors = torch.stack(euclidean_errors)
    return euclidean_errors.mean().cpu().numpy(), euclidean_errors.std().cpu().numpy()


def main(args, train_loader, val_loader):

    logging.info('len of train loader:{}, len of test loader:{}'.format(len(train_loader), len(val_loader)))
    model = VQVAE(args.VQVAE, 9 * 6).to(mydevice) # n_joints * n_chanels
    if torch.cuda.is_available() and args.get("gpus", None):
        model = nn.DataParallel(model, device_ids=args.gpus, output_device=args.gpus[0]).to(mydevice)


    best_val_loss = (1e+2, 0)  # value, epoch

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    params = model.module.parameters() if isinstance(model, nn.DataParallel) else model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, betas=args.betas)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    updates = 0
    total = len(train_loader)

    tb_path = args.name + '_' + str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    tb_writer = SummaryWriter(log_dir=str(Path(args.model_save_path).parent / 'tensorboard_runs' / tb_path))


    for epoch in range(1, args.epochs + 1):
        logging.info(f'Epoch: {epoch}')
        i = 0
        diff_mean, diff_std = evaluate_valset(model, val_loader)
        logging.info('diff mean on validation: {:.3f}, diff std on validation: {:.3f}'.format(diff_mean, diff_std))
        tb_writer.add_scalar('diff mean/validation', diff_mean, epoch)
        tb_writer.add_scalar('diff std/validation', diff_std, epoch)
        is_best = diff_mean < best_val_loss[0]
        if is_best:
            logging.info(' *** BEST VALIDATION LOSS : {:.3f}'.format(diff_mean))
            best_val_loss = (diff_mean, epoch)
        else:
            logging.info(' best validation loss so far: {:.3f} at EPOCH {}'.format(best_val_loss[0], best_val_loss[1]))

        if is_best or (epoch % args.save_per_epochs == 0):
            if is_best:
                save_name = '{}/{}_checkpoint_best.bin'.format(args.model_save_path, args.name)
            else:
                save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, epoch)

            torch.save({
                'args': args, "epoch": epoch, 'model_dict': model.state_dict()
            }, save_name)
            logging.info('Saved the checkpoint')

        # train model
        model = model.train()
        start = datetime.now()
        for batch_i, batch in enumerate(train_loader, 0):
            pose_seq = batch[0].to(mydevice)  # (b, 60, 9*6)
            # labels = batch[1]               # Not needed
            optimizer.zero_grad()
            output, loss, metrics = model(pose_seq)


            # write to tensorboard
            tb_writer.add_scalar('loss' + '/train', loss, updates)
            for key in metrics.keys():
                tb_writer.add_scalar(key + '/train', metrics[key], updates)
            loss.backward()
            optimizer.step()
            # log
            stats = {'updates': updates, 'loss': loss.item()}
            stats_str = ' '.join(f'{key}[{val:.8f}]' for key, val in stats.items())
            i += 1
            remaining = str((datetime.now() - start) / i * (total - i))
            remaining = remaining.split('.')[0]
            logging.info(f'> epoch [{epoch}] updates[{i}] {stats_str} eta[{remaining}]')
            # if i == total:
            #     logging.debug('\n')
            #     logging.debug(f'elapsed time: {str(datetime.now() - start).split(".")[0]}')
            updates += 1
        schedular.step()
    tb_writer.close()
    # print best losses
    logging.info('--------- Final best loss values ---------')
    logging.info('diff mean: {:.3f} at EPOCH {}'.format(best_val_loss[0], best_val_loss[1]))


if __name__ == '__main__':
    print("Loading data...")

    lmdb_path = args.data_path  #
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)
    config.no_cuda = args.gpu



    metadata_pkl = os.path.join(lmdb_path, "metadata", "metadata.pkl")
    with open(metadata_pkl, "rb") as f:
        meta = pickle.load(f)

    sample_keys = meta["sample_keys"]
    speaker_encodings = meta["speaker_encodings"]
    culture_encodings = meta["culture_encodings"]

    splits_pkl = os.path.join(lmdb_path, "metadata", "splits_subject_dependent.pkl")

    train_loader, val_loader, test_loader = create_subject_dependent_split(
        sample_keys,
        motion_only=True,
        batch_size=config.batch_size,
        save_path=splits_pkl,  # IMPORTANT: this must be a *file*, not a directory
        lmdb_path=lmdb_path,  # IMPORTANT: this is how GestureDataset finds LMDB
        speaker_encodings=speaker_encodings,
        culture_encodings=culture_encodings
    )
    print("Data Loaded!")

    main(config, train_loader, val_loader)

import os
import argparse
import h5py
import math

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import EventWarping
from loss.reconstruction import BrightnessConstancy
from loss.cross_entropy import cross_entropy_power_spectrum_DLDL_softmax2
from models.model import FireFlowNet, EVFlowNet
from models.model import FireNet, E2VID
from model_rppg import ViT_ST_ST_Compact3_TDC_gra_sharp
from utils.utils import load_model, create_model_dir, save_model
from utils.visualization import Visualization


def events_to_image(xs, ys, ps, sensor_size=(128, 128)):
    """
    Accumulate events into an image.
    """

    device = xs.device
    img_size = list(sensor_size)
    img = torch.zeros(img_size).to(device)

    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    img.index_put_((ys, xs), ps, accumulate=True)

    return img


def events_to_voxel(xs, ys, ts, ps, num_bins, sensor_size=(128, 128)):
    """
    Generate a voxel grid from input events using temporal bilinear interpolation.
    """

    assert len(xs) == len(ys) and len(ys) == len(ts) and len(ts) == len(ps)

    voxel = []
    ts = ts * (num_bins - 1)
    zeros = torch.zeros(ts.size())
    for b_idx in range(num_bins):
        weights = torch.max(zeros, 1.0 - torch.abs(ts - b_idx))
        voxel_bin = events_to_image(xs, ys, ps * weights, sensor_size=sensor_size)
        voxel.append(voxel_bin)

    return torch.stack(voxel)


class Neg_Pearson(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson, self).__init__()
        return

    def forward(self, preds, labels):  # all variable operation
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])  # x
            sum_y = torch.sum(labels[i])  # y
            sum_xy = torch.sum(preds[i] * labels[i])  # xy
            sum_x2 = torch.sum(torch.pow(preds[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i], 2))  # y^2
            N = preds.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

            loss += 1 - pearson

        loss = loss / preds.shape[0]
        return loss


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def train_rppg(args, data_path):
    device = 'cuda:0'
    settings = {'name': 'E2VID', 'base_num_channels': 32, 'kernel_size': 5}

    model_e2vid = eval('E2VID')(settings.copy(), 5).to(device)
    model_e2vid = load_model(args.prev_model, model_e2vid, device)
    model_e2vid.train()

    model_rppg = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(160, 128, 128), patches=(4, 4, 4), dim=96, ff_dim=144,
                                                  num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
    model_rppg = model_rppg.cuda()
    # model_rppg = load_model(args.prev_model, model_rppg, device)
    model_rppg.train()

    lr = args.lr
    optimizer1 = optim.Adam(model_rppg.parameters(), lr=lr, weight_decay=0.00005)
    # optimizer1 = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)
    criterion_Pearson = Neg_Pearson()

    a_start = 0.1
    b_start = 1.0
    exp_a = 0.5
    exp_b = 5.0

    # for epoch, data in enumerate(os.listdir(data_path)):  # epoch
    for epoch in range(2 * len(os.listdir(data_path))):
        data = np.random.choice(os.listdir(data_path))
        scheduler1.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        loss_rPPG_avg = AvgrageMeter()
        loss_peak_avg = AvgrageMeter()
        loss_kl_avg_test = AvgrageMeter()
        loss_hr_mae = AvgrageMeter()

        file_path = os.path.join(data_path, data)

        with h5py.File(file_path, 'r') as file:
            bvp = file['bvp']
            avg_hr = file['hr']
            fps = 30
            length = len(bvp)
            for i in range(12):
                start = np.random.randint(0, length - 160)
                end = start + 160
                label_ecg = torch.Tensor(bvp[start:end]).cuda()
                label_ecg = (label_ecg - torch.mean(label_ecg)) / torch.std(label_ecg)
                label_hr = torch.Tensor(avg_hr[start:end] - 40).cuda()

                for idx in range(start, end):
                    ts_all = file['/events/ts'][:]
                    condition = np.where((ts_all >= idx*1e6/fps) & (ts_all < (idx + 1)*1e6/fps))
                    xs = torch.Tensor(file['/events/xs'][:][condition])
                    ys = torch.Tensor(file['/events/ys'][:][condition])
                    ts = torch.Tensor(file['/events/ts'][:][condition])
                    ps = torch.Tensor(file['/events/ps'][:][condition])

                    input_voxel = events_to_voxel(xs, ys, ts, ps, 5).unsqueeze(0).to(device)
                    if idx == start:
                        input_cat = input_voxel
                    else:
                        input_cat = torch.cat([input_cat, input_voxel], dim=0)

                ev_out = model_e2vid(input_cat)
                rPPG = ev_out.view(1, 160)
                '''if idx == start:
                        ev_cat = ev_out
                    else:
                        ev_cat = torch.cat([ev_cat, ev_out], dim=1)'''

                # ev_cat = torch.cat([ev_cat.unsqueeze(0)] * 3, dim=1)
                # rPPG, Score1, Score2, Score3 = model_rppg(ev_out.unsqueeze(0).unsqueeze(0).to(device), gra_sharp=2.0)

                rPPG = (rPPG - torch.mean(rPPG)) / (torch.std(rPPG) + 1e-8)  # normalize2
                loss_rPPG = criterion_Pearson(rPPG, label_ecg)

                fre_loss = 0.0
                kl_loss = 0.0
                train_mae = 0.0

                loss_distribution_kl, fre_loss_temp, train_mae_temp = cross_entropy_power_spectrum_DLDL_softmax2(rPPG[0], label_hr[0], fps, std=1.0)
                fre_loss = fre_loss + fre_loss_temp
                kl_loss = kl_loss + loss_distribution_kl
                train_mae = train_mae + train_mae_temp

                if epoch > 25:
                    a = 0.05
                    b = 5.0
                else:
                    # exp descend
                    a = a_start * math.pow(exp_a, epoch / 25.0)
                    # exp ascend
                    b = b_start * math.pow(exp_b, epoch / 25.0)

                a = 0.1
                # b = 1.0

                loss = a * loss_rPPG + b * (fre_loss + kl_loss)

                loss.backward(retain_graph=True)
                optimizer1.step()

                n = 160
                loss_rPPG_avg.update(loss_rPPG.data, n)
                loss_peak_avg.update(fre_loss.data, n)
                loss_kl_avg_test.update(kl_loss.data, n)
                loss_hr_mae.update(train_mae, n)

            print(
                "Train Epoch: {:04d} [{:03d}/{:03d} ({:03d}%)] rPPG loss: {:.6f}, total loss: {:.6f}".format(
                    epoch,
                    epoch,
                    2 * len(os.listdir(data_path)),
                    int(50 * epoch / len(os.listdir(data_path))),
                    loss_rPPG,
                    loss,
                ),
                end="\r",
            )
            print(epoch, ' over')
            if (epoch + 1) % 10 == 0 or (epoch + 1) == len(os.listdir(data_path)):
                torch.save(model_rppg.state_dict(), 'weight/' + args.log + '/rppg' + '_%d.pkl' % epoch)
                torch.save(model_e2vid.state_dict(), 'weight/' + args.log + '/e2vid' + '_%d.pkl' % epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train_reconstruction.yml",
        help="training configuration",
    )
    parser.add_argument(
        "--path_models",
        default="trained_models/",
        help="location of trained models",
    )
    parser.add_argument(
        "--prev_model",
        default="",
        help="pre-trained model to use as starting point",
    )
    parser.add_argument(
        '--input_data', type=str,
        default="./data/UBFC_h5"
    )
    parser.add_argument(
        '--lr', type=float,
        default=0.0001,
        help='initial learning rate',
    )
    parser.add_argument(
        '--log', type=str,
        default="ev_rppg_UBFC",
        help='log and save model name',
    )
    parser.add_argument(
        '--step_size', type=int,
        default=50,
        help='stepsize of optim.lr_scheduler.StepLR, how many epochs lr decays once'
    )
    parser.add_argument(
        '--gamma', type=float,
        default=0.5,
        help='gamma of optim.lr_scheduler.StepLR, decay of lr'
    )

    args = parser.parse_args()

    # launch training
    input_data = args.input_data
    train_rppg(args, input_data)

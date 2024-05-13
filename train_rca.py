#!/usr/bin/python
# -*- encoding: utf-8 -*-
from PIL import Image
import numpy as np
from torch.autograd import Variable
from FusionNet import FusionNet
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger
from model_TII import BiSeNet
from cityscapes import CityScapes
from loss import OhemCELoss, Fusionloss
from optimizer import Optimizer
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def train_fusion(num=0, logger=None):
    lr_start = 0.001
    epoch = args.epochs
    modelpth = './model'
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)
    fusionmodel = eval('FusionNet')(output=1)
    fusionmodel.cuda()
    fusionmodel.train()
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)

    train_dataset = Fusion_dataset('train')
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    criteria_fusion = Fusionloss()

    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')
    for epo in range(0, epoch):
        print('\n-------' + 'we are training the ' + str(epo) + '/' + str(epoch) + '-------')
        lr_start = 0.001
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo
        if mode_num == 2:
            for it, (image_vis, image_ir, name) in enumerate(train_loader):
                print('\r processing:  {}'.format(str(it)) + '/' + str(len(train_loader)) + ' in epoch ' + str(epo) ,end='')
                fusionmodel.train()
                image_vis = Variable(image_vis).cuda()
                image_vis_ycrcb = RGB2YCrCb(image_vis)
                image_ir = Variable(image_ir).cuda()
                logits = fusionmodel(image_vis_ycrcb, image_ir)
                fusion_ycrcb = torch.cat(
                    (logits, image_vis_ycrcb[:, 1:2, :, :],
                    image_vis_ycrcb[:, 2:, :, :]),
                    dim=1,
                )
                fusion_image = YCrCb2RGB(fusion_ycrcb)

                ones = torch.ones_like(fusion_image)
                zeros = torch.zeros_like(fusion_image)
                fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
                fusion_image = torch.where(
                    fusion_image < zeros, zeros, fusion_image)
                optimizer.zero_grad()
                loss_fusion, loss_in, loss_grad, loss_sim = criteria_fusion(
                    image_vis_ycrcb, image_ir, logits
                )
                loss_total = loss_fusion
                loss_total.backward()
                optimizer.step()
                ed = time.time()
                t_intv, glob_t_intv = ed - st, ed - glob_st
                now_it = train_loader.n_iter * epo + it + 1
                eta = int((train_loader.n_iter * epoch - now_it)
                        * (glob_t_intv / (now_it)))
                eta = str(datetime.timedelta(seconds=eta))
                if now_it % 10 == 0:
                    loss_seg=0
                    msg = ', '.join(
                        [
                            'step: {it}/{max_it}',
                            'loss_total: {loss_total:.4f}',
                            'loss_in: {loss_in:.4f}',
                            'loss_grad: {loss_grad:.4f}',
                            'loss_seg: {loss_seg:.4f}',
                            'eta: {eta}',
                            'time: {time:.4f}',
                        ]
                    ).format(
                        it=now_it,
                        max_it=train_loader.n_iter * epoch,
                        loss_total=loss_total.item(),
                        loss_in=loss_in.item(),
                        loss_grad=loss_grad.item(),
                        loss_seg=loss_seg,
                        time=t_intv,
                        eta=eta,
                    )
                    logger.info(msg)
                    st = ed

    fusion_model_file = os.path.join(modelpth, 'fusion_model.pth')
    torch.save(fusionmodel.state_dict(), fusion_model_file)
    logger.info("Fusion Model Save to: {}".format(fusion_model_file))
    logger.info('\n')

def run_fusion(type='train'):
    fusion_model_path = './model/Fusion/fusion_model.pth'
    fused_dir = os.path.join('./MSRS/Fusion', type, 'MSRS')
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    fusionmodel = eval('FusionNet')(output=1)
    fusionmodel.eval()
    if args.gpu >= 0:
        fusionmodel.cuda(args.gpu)
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    test_dataset = Fusion_dataset(type)
    print('done!')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir, name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            if args.gpu >= 0:
                images_vis = images_vis.cuda(args.gpu)
                images_ir = images_ir.cuda(args.gpu)
            images_vis_ycrcb = RGB2YCrCb(images_vis)
            logits = fusionmodel(images_vis_ycrcb, images_ir)
            fusion_ycrcb = torch.cat(
                    (logits, images_vis_ycrcb[:, 1:2, :,
                     :], images_vis_ycrcb[:, 2:, :, :]),
                    dim=1,
                )
            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(
                fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image)
                )
            fused_image = np.uint8(255.0 * fused_image)
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = image.squeeze()
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))

if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser(description='Train RCAFusion with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='RCAFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    parser.add_argument('--epochs', '-E', type=int, default=50)
    parser.add_argument('--mode_num', '-m', type=int, default=2)
    args = parser.parse_args()
    logpath='./logs'
    logger = logging.getLogger()
    mode_num = args.mode_num
    train_fusion(0, logger)
    print('Train Successfully')
    run_fusion('train')
    print('Finish Run')
    time_now = time.time()
    time_output = (time_now - time_start) / 60
    print(' Totally used ' + str(time_output) + 'min')
    print("training Done!")

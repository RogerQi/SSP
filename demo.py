from model.SSP_matching import SSP_MatchingNet
from dataset.transform import crop, hflip, normalize
from util.utils import count_params, set_seed, mIOU

import argparse
import os
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from IPython import embed

def parse_args():
    parser = argparse.ArgumentParser(description='Mining Latent Classes for Few-shot Segmentation')
    # basic arguments
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50', 'resnet101'],
                        default='resnet50',
                        help='backbone of semantic segmentation model')
    parser.add_argument('--refine', dest='refine', action='store_true', default=False)

    args = parser.parse_args()
    return args

def get_data(img_s_path_list, mask_s_path_list, img_q_path):
    assert len(img_s_path_list) == len(mask_s_path_list)
    n_shot = len(img_s_path_list)

    img_q = Image.open(img_q_path).convert('RGB')

    # support ids, images and masks
    img_s_list, mask_s_list = [], []

    for i in range(n_shot):
        img_s = Image.open(img_s_path_list[i]).convert('RGB')
        mask_s = Image.fromarray(np.array(Image.open(mask_s_path_list[i])))

        img_s_list.append(img_s)
        mask_s_list.append(mask_s)

    img_q = normalize(img_q)
    img_q = img_q.reshape((1,) + img_q.shape)

    for k in range(n_shot):
        img_s_list[k], mask_s_list[k] = normalize(img_s_list[k], mask_s_list[k])
        img_s_list[k] = img_s_list[k].reshape((1,) + img_s_list[k].shape)
        mask_s_list[k] = mask_s_list[k].reshape((1,) + mask_s_list[k].shape)

    return img_s_list, mask_s_list, img_q

def evaluate(model):
    img_s_list, mask_s_list, img_q = get_data(
        ["frame-000174.jpg"],
        ["frame-000174.png"],
        "frame-001279.jpg"
    )
    img_q = img_q.cuda()
    for k in range(len(img_s_list)):
        img_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), mask_s_list[k].cuda()

    with torch.no_grad():
        pred_bchw = model(img_s_list, mask_s_list, img_q, None)[0]
        pred_bhw = torch.argmax(pred_bchw, dim=1)
    
    for b in range(pred_bhw.shape[0]):
        pred_np_hw = pred_bhw[b].cpu().numpy().astype(np.uint8)
        Image.fromarray(pred_np_hw * 255).save(f"{b}.png")

def main():
    args = parse_args()
    print('\n' + str(args))
    refine = True
    model = SSP_MatchingNet(args.backbone, args.refine)
    checkpoint_path = "outdir/models/coco/fold_1/coco_1_resnet101_5shot_47.01.pth"

    print('Evaluating model:', checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    #print(model)
    print('\nParams: %.1fM' % count_params(model))

    best_model = DataParallel(model).cuda()

    model.eval()
    evaluate(best_model)

if __name__ == '__main__':
    main()


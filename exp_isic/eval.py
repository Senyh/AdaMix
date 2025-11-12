import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from skimage import io
from skimage import color
from data.dataset import ISICDataset
from models import networks
from utils import measure_img, ensure_dir
from medpy import metric
sep = '\\' if sys.platform[:3] == 'win' else '/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(known=False):
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    parser.add_argument('--project', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/runs/AdaMix_ST', help='project path for saving results')
    parser.add_argument('--backbone', type=str, default='UNet', choices=['UNet'], help='segmentation backbone')
    parser.add_argument('--data_path', type=str, default='YOUR_DATA_PATH', help='path to the data')
    parser.add_argument('--labeled_percentage', type=float, default=0.1, help='the percentage of labeled data')
    parser.add_argument('--image_size', type=int, default=256, help='the size of images for training and testing')
    parser.add_argument('--patch_size', type=int, default=8, help='the size of patches for adamix (256 // patchsize)')
    parser.add_argument('--topk', type=int, default=16, help='the number of patches for adamix')
    parser.add_argument('--batch_size', type=int, default=1, help='number of inputs per batch')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers to use for dataloader')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    parser.add_argument('--num_classes', type=int, default=2, help='number of target categories')
    parser.add_argument('--model_weights', type=str, default='best.pth', help='model weights')
    parser.add_argument('--visualization', type=bool, default=True, help='qualitative results')
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


def get_data(args):
    test_set = ISICDataset(image_path=args.data_path, stage='test', image_size=args.image_size, is_augmentation=False)
    test_dataloder = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    return test_dataloder, len(test_set)


def load_model(model_weights, in_channels, num_classes, backbone):
    model = networks.__dict__[backbone](in_channels=in_channels, out_channels=num_classes).to(device)
    print('#parameters:', sum(param.numel() for param in model.parameters()))
    model.load_state_dict(torch.load(model_weights))
    return model

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd


def eval(is_debug=False):
    args = get_args()
    # Project Saving Path
    project_path = args.project + '_{}_patchsize_{}_topk_{}_label_{}/'.format(args.backbone, args.patch_size, args.topk, args.labeled_percentage)
    # Load Data
    test_dataloader, length = get_data(args=args)
    iters = len(test_dataloader)
    iter_test_dataloader = iter(test_dataloader)
    if is_debug:
        pbar = range(10)
        length = 10 * args.batch_size
    else:
        pbar = range(iters)
    # Load model
    weights_path = project_path + 'weights/' + args.model_weights
    model = load_model(model_weights=weights_path, in_channels=args.in_channels, num_classes=args.num_classes, backbone=args.backbone)
    model.eval()
    ############################
    # Evaluation
    ############################
    print('start evaluation')
    results = {i: [] for i in range(4)}
    with torch.no_grad():
        for idx in tqdm(pbar):
            image, label = next(iter_test_dataloader)
            image, label = image.to(device), label.to(device)
            pred = model(image)['out']
            B, C, H, W = label.shape
            pred = F.interpolate(pred, size=[H, W], mode='bilinear', align_corners=False)
            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1).cpu().data.numpy()
            label = label.squeeze(1).long().cpu().numpy()
            # Post-Processing
            if len(pred[pred > 0]) == 0:
                pred[0, 0, 0] = 1
            pred_copy = copy.deepcopy(pred)
            pred_u = measure_img(pred, t_num=1).astype('bool')
            pred = pred_u * pred_copy
            if len(pred[pred > 0]) == 0:
                pred[0, 0, 0] = 1
            result = calculate_metric_percase(pred=pred, gt=label)
            for i in range(4):
                results[i].append(result[i])
            if args.visualization:
                un_norm = Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
                image = un_norm(image)
                image = image.squeeze(0).cpu().numpy() * 255.
                image = image.transpose([1, 2, 0])
                label = label[0]
                pred = pred[0]
                save_path = project_path + 'predictions/'
                ensure_dir(save_path)
                io.imsave(save_path + str(idx) + '_img.png', image.astype('uint8'))
                io.imsave(save_path + str(idx) + '_lbl.png',
                        (color.label2rgb(label, colors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 255.).astype('uint8'))
                io.imsave(save_path + str(idx) + '_prd.png',
                        (color.label2rgb(pred, colors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 255.).astype('uint8'))
    # save results
    data_frame = pd.DataFrame(
        data={i: results[i] for i in range(4)},
        index=range(1, length + 1))
    data_frame.to_csv(project_path + '/' + 'evaluation.csv', index_label='Index')
    result = data_frame.values
    avg_score = np.mean(result, axis=0)
    avg_std = np.std(result, axis=0, ddof=1)
    print('AVG Score:{}, Std:{}\n'.format(avg_score, avg_std))
    with open(project_path+'/performance.txt', 'w') as f:
        f.writelines('metric is {} \n'.format(avg_score))
        f.writelines('standard deviation is {}\n'.format(avg_std))
    print('AVG Score:', avg_score)
    print('EVAL FINISHED!', project_path)


if __name__ == '__main__':
    eval()
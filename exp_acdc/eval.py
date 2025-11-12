import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import argparse
import sys
import pandas as pd
import numpy as np
import h5py
import copy
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from exp_acdc.data.dataset import ACDCDataset
from models import networks
from utils import calculate_metric_percase, ensure_dir, measure_img
sep = '\\' if sys.platform[:3] == 'win' else '/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(known=False):
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    parser.add_argument('--project', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/runs/AdaMix_CT', help='project path for saving results')
    parser.add_argument('--backbone', type=str, default='UNet', choices=['UNet'], help='segmentation backbone')
    parser.add_argument('--data_path', type=str, default='YOUR_DATA_PATH', help='path to the data')
    parser.add_argument('--labeled_percentage', type=float, default=.1, help='the percentage of labeled data')
    parser.add_argument('--image_size', type=int, default=256, help='the size of images for training and testing')
    parser.add_argument('--patch_size', type=int, default=8, help='the size of patches for adamix (256 // patchsize)')
    parser.add_argument('--topk', type=int, default=16, help='the number of patches for adamix')
    parser.add_argument('--batch_size', type=int, default=1, help='number of inputs per batch')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers to use for dataloader')
    parser.add_argument('--in_channels', type=int, default=1, help='input channels')
    parser.add_argument('--num_classes', type=int, default=4, help='number of target categories')
    parser.add_argument('--model_weights', type=str, default='best.pth', help='model weights')
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


def get_data(args):
    test_set = ACDCDataset(image_path=args.data_path, stage='test', image_size=args.image_size, is_augmentation=False)
    test_dataloder = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    return test_dataloder, len(test_set)


def load_model(model_weights, in_channels, num_classes, backbone='UNet'):
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


def test_single_volume(case, net, test_save_path, args):
    h5f = h5py.File(args.data_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=3)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)['out']
            if len(out_main)>1:
                out_main=out_main[0]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    # Post-Processing
    prediction_copy = copy.deepcopy(prediction)
    prediction_u = measure_img(prediction, t_num=1).astype('bool')
    prediction = prediction_u * prediction_copy

    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    if np.sum(prediction == 2)==0:
        second_metric = 0,0,0,0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)

    if np.sum(prediction == 3)==0:
        third_metric = 0,0,0,0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def eval():
    args = get_args()
    # Project Saving Path
    project_path = args.project + '_{}_patchsize_{}_topk_{}_label_{}/'.format(args.backbone, args.patch_size, args.topk, args.labeled_percentage)
    # Load Data
    with open(args.data_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    # Load model
    weights_path = project_path + 'weights/' + args.model_weights
    model = load_model(model_weights=weights_path, in_channels=args.in_channels, num_classes=args.num_classes, backbone=args.backbone)
    model.eval()
    test_save_path = project_path + '/predictions/'
    ensure_dir(test_save_path)
    ############################
    # Evaluation
    ############################
    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    results = {i: [] for i in range(4)}
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(case, model, test_save_path, args)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
        for i in range(4):
            total_metric = (np.asarray(first_metric)+np.asarray(second_metric)+np.asarray(third_metric)) / 3.
            results[i].append(total_metric[i])
    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    data_frame = pd.DataFrame(
        data={i: results[i] for i in range(4)},
        index=range(1, len(image_list) + 1))
    data_frame.to_csv(project_path + '/' + 'evaluation.csv', index_label='Index')
    result = data_frame.values
    avg_score = np.mean(result, axis=0)
    avg_std = np.std(result, axis=0, ddof=1)
    print('AVG Score:{}, Std:{}\n'.format(avg_score, avg_std))
    return avg_metric, test_save_path, avg_score, avg_std


if __name__ == '__main__':
    metric, test_save_path, avg_score, avg_std = eval()
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
    with open(test_save_path+'../performance.txt', 'w') as f:
        f.writelines('metric is {} \n'.format(metric))
        f.writelines('average metric is {}\n'.format((metric[0]+metric[1]+metric[2])/3))
        f.writelines('average metric is {}\n'.format(avg_score))
        f.writelines('standard deviation is {}\n'.format(avg_std))
    print(test_save_path)

    
    
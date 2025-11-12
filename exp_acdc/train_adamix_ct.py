import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from easydict import EasyDict
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from exp_acdc.data.dataset import ACDCDataset
from utils import ensure_dir
from models import networks
from wheels.loss_functions import DSCLossH
from wheels.logger import logger as logging
from wheels.mask_generator import BoxMaskGenerator, AddMaskParamsToBatch, SegCollate
from wheels.torch_utils import seed_torch
from wheels.model_init import init_weight
from wheels.adamix_utils import AdaMix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(known=False):
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--project', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/runs/AdaMix_CT', help='project path for saving results')
    parser.add_argument('--backbone', type=str, default='UNet', choices=['DeepLabv3p', 'UNet'], help='segmentation backbone')
    parser.add_argument('--data_path', type=str, default='YOUR_DATA_PATH', help='path to the data')
    parser.add_argument('--image_size', type=int, default=256, help='the size of images for training and testing')
    parser.add_argument('--patch_size', type=int, default=8, help='the size of patches for adamix (256 // patchsize)')
    parser.add_argument('--topk', type=int, default=16, help='the number of patches for adamix')
    parser.add_argument('--labeled_percentage', type=float, default=0.1, help='the percentage of labeled data')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='number of inputs per batch')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers to use for dataloader')
    parser.add_argument('--in_channels', type=int, default=1, help='input channels')
    parser.add_argument('--num_classes', type=int, default=4, help='number of target categories')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--log_freq', type=int, default=10, help='logging frequency of metrics accord to the current iteration')
    parser.add_argument('--save_freq', type=int, default=10, help='saving frequency of model weights accord to the current epoch')
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


def get_data(args):
    val_set = ACDCDataset(image_path=args.data_path, stage='val', image_size=args.image_size, is_augmentation=False)
    labeled_train_set = ACDCDataset(image_path=args.data_path, stage='train', image_size=args.image_size, is_augmentation=True, labeled=True, percentage=args.labeled_percentage)
    unlabeled_train_set = ACDCDataset(image_path=args.data_path, stage='train', image_size=args.image_size, is_augmentation=True, labeled=False, percentage=args.labeled_percentage)
    train_set = ConcatDataset([labeled_train_set, unlabeled_train_set])

    # repeat the labeled set to have a equal length with the unlabeled set (dataset)
    print('before: ', len(train_set), len(labeled_train_set), len(val_set))
    labeled_ratio = len(train_set) // len(labeled_train_set)
    labeled_train_set = ConcatDataset([labeled_train_set for i in range(labeled_ratio)])
    labeled_train_set = ConcatDataset([labeled_train_set,
                                       Subset(labeled_train_set, range(len(train_set) - len(labeled_train_set)))])
    print('after: ', len(train_set), len(labeled_train_set), len(val_set))
    assert len(labeled_train_set) == len(train_set)
    train_labeled_dataloder = DataLoader(dataset=labeled_train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    train_unlabeled_dataloder = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_dataloder = DataLoader(dataset=val_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    mask_generator = BoxMaskGenerator(prop_range=(0.25, 0.5),
                                        n_boxes=3,
                                        random_aspect_ratio=True,
                                        prop_by_area=True,
                                        within_bounds=True,
                                        invert=True)

    add_mask_params_to_batch = AddMaskParamsToBatch(mask_generator)
    mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)
    train_unlabeled_aux_dataloder = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=mask_collate_fn)
    mask_generator = BoxMaskGenerator(prop_range=(0.25, 0.5),
                                        n_boxes=3,
                                        random_aspect_ratio=True,
                                        prop_by_area=True,
                                        within_bounds=True,
                                        invert=True)
    add_mask_params_to_batch = AddMaskParamsToBatch(mask_generator)
    mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)
    train_labeled_aux_dataloder = DataLoader(dataset=labeled_train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=mask_collate_fn)
    return train_labeled_dataloder, train_unlabeled_dataloder, val_dataloder, train_labeled_aux_dataloder, train_unlabeled_aux_dataloder


def main(is_debug=False):
    args = get_args()
    seed_torch(args.seed)
    # Project Saving Path
    project_path = args.project + '_{}_patchsize_{}_topk_{}_label_{}/'.format(args.backbone, args.patch_size, args.topk, args.labeled_percentage)
    ensure_dir(project_path)
    save_path = project_path + 'weights/'
    ensure_dir(save_path)
    ensure_dir(project_path + 'images/')

    # Tensorboard & Statistics Results & Logger
    tb_dir = project_path + '/tensorboard{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    writer = SummaryWriter(tb_dir)
    metrics = EasyDict()
    metrics.train_loss = []
    metrics.train_s_loss = []
    metrics.train_u_loss = []
    metrics.train_x_loss = []
    metrics.val_loss = []
    logger = logging(project_path + 'train_val.log')
    logger.info('PyTorch Version {}\n Experiment{}'.format(torch.__version__, project_path))

    # Load Data
    train_labeled_dataloader, train_unlabeled_dataloader, val_dataloader, train_labeled_aux_dataloader, train_unlabeled_aux_dataloader = get_data(args=args)
    iters = len(train_labeled_dataloader)
    val_iters = len(val_dataloader)

    # Load Model & EMA
    student1 = networks.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)
    init_weight(student1.net.classifier, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')
    student2 = networks.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)
    init_weight(student2.net.classifier, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')
    best_epoch = 0
    best_loss = 100
    total_iters = iters * args.num_epochs
    conf_threshold = 0.95

    # Criterion & Optimizer & LR Schedule
    criterion_s = DSCLossH(num_classes=args.num_classes, device=device)
    criterion_u = DSCLossH(num_classes=args.num_classes, device=device)
    optimizer1 = optim.AdamW(student1.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    optimizer2 = optim.AdamW(student2.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    # Init AdaMix
    s1_adamix_l = AdaMix(total_steps=total_iters // 4, num_classes=args.num_classes, image_size=args.image_size, patch_size=args.patch_size, topk=args.topk, mode='hard', p=0.5)
    s1_adamix_u = AdaMix(total_steps=total_iters // 4, num_classes=args.num_classes, image_size=args.image_size, patch_size=args.patch_size, topk=args.topk, mode='hard', p=0.5)
    s2_adamix_l = AdaMix(total_steps=total_iters // 4, num_classes=args.num_classes, image_size=args.image_size, patch_size=args.patch_size, topk=args.topk, mode='hard', p=0.5)
    s2_adamix_u = AdaMix(total_steps=total_iters // 4, num_classes=args.num_classes, image_size=args.image_size, patch_size=args.patch_size, topk=args.topk, mode='hard', p=0.5)

    # Train
    since = time.time()
    logger.info('start training')
    for epoch in range(1, args.num_epochs + 1):
        epoch_metrics = EasyDict()
        epoch_metrics.train_loss = []
        epoch_metrics.train_s_loss = []
        epoch_metrics.train_u_loss = []
        epoch_metrics.train_x_loss = []
        if is_debug:
            pbar = range(10)
        else:
            pbar = range(iters)
        iter_train_labeled_dataloader = iter(train_labeled_dataloader)
        iter_train_unlabeled_dataloader = iter(train_unlabeled_dataloader)
        iter_train_labeled_aux_dataloader = iter(train_labeled_aux_dataloader)
        iter_train_unlabeled_aux_dataloader = iter(train_unlabeled_aux_dataloader)
        
        ############################
        # Train
        ############################
        student1.train()
        student2.train()
        for idx in pbar:
            # labeled data
            image, label, imageA1, imageA2 = next(iter_train_labeled_dataloader)
            image, label, imageA1, imageA2 = image.to(device), label.to(device), imageA1.to(device), imageA2.to(device)
            # unlabel data
            uimage, _, uimageA1, uimageA2 = next(iter_train_unlabeled_dataloader)
            uimage, uimageA1, uimageA2 = uimage.to(device), uimageA1.to(device), uimageA2.to(device)
            # labeled auxiliary data
            laimage, lalabel, laimageA1, laimageA2, lamask = next(iter_train_labeled_aux_dataloader)
            laimage, lalabel, laimageA1, laimageA2, lamask = laimage.to(device), lalabel.to(device), laimageA1.to(device), laimageA2.to(device), lamask.to(device)
            # unlabeled auxiliary data
            uaimage, _, uaimageA1, uaimageA2, uamask = next(iter_train_unlabeled_aux_dataloader)
            uaimage, uaimageA1, uaimageA2, uamask = uaimage.to(device), uaimageA1.to(device), uaimageA2.to(device), uamask.to(device)
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            with torch.no_grad():
                # student 1
                s1_pred_a = student1(torch.cat([laimage, uaimage]))
                s1_pred_la_logits, s1_pred_ua_logits = s1_pred_a['out'].chunk(2)
                s1_pred_la_probs = torch.softmax(s1_pred_la_logits, dim=1)
                s1_pred_la_pseudo = torch.argmax(s1_pred_la_probs, dim=1).clone()
                s1_pred_la_conf = s1_pred_la_probs.max(dim=1)[0].clone()
                s1_pred_ua_probs = torch.softmax(s1_pred_ua_logits, dim=1)
                s1_pred_ua_pseudo = torch.argmax(s1_pred_ua_probs, dim=1).clone()
                s1_pred_ua_conf = s1_pred_ua_probs.max(dim=1)[0].clone()

                s1_pred_lu = student1(torch.cat([image, uimage]))
                s1_pred_l_logits, s1_pred_u_logits = s1_pred_lu['out'].chunk(2)
                s1_pred_l_probs = torch.softmax(s1_pred_l_logits, dim=1) # 8 4 256 256
                s1_pred_l_pseudo = torch.argmax(s1_pred_l_probs, dim=1) # 8 256 256
                s1_pred_l_conf = s1_pred_l_probs.max(dim=1)[0]
                s1_pred_u_probs = torch.softmax(s1_pred_u_logits, dim=1) # 8 4 256 256
                s1_pred_u_pseudo = torch.argmax(s1_pred_u_probs, dim=1) # 8 256 256
                s1_pred_u_conf = s1_pred_u_probs.max(dim=1)[0]

                s1_pred_uA1_logits = student1(uimageA1)['out']

                # student 2
                s2_pred_a = student2(torch.cat([laimage, uaimage]))
                s2_pred_la_logits, s2_pred_ua_logits = s2_pred_a['out'].chunk(2)
                s2_pred_la_probs = torch.softmax(s2_pred_la_logits, dim=1)
                s2_pred_la_pseudo = torch.argmax(s2_pred_la_probs, dim=1).clone()
                s2_pred_la_conf = s2_pred_la_probs.max(dim=1)[0].clone()
                s2_pred_ua_probs = torch.softmax(s2_pred_ua_logits, dim=1)
                s2_pred_ua_pseudo = torch.argmax(s2_pred_ua_probs, dim=1).clone()
                s2_pred_ua_conf = s2_pred_ua_probs.max(dim=1)[0].clone()

                s2_pred_lu = student2(torch.cat([image, uimage]))
                s2_pred_l_logits, s2_pred_u_logits = s2_pred_lu['out'].chunk(2)
                s2_pred_l_probs = torch.softmax(s2_pred_l_logits, dim=1) # 8 4 256 256
                s2_pred_l_pseudo = torch.argmax(s2_pred_l_probs, dim=1) # 8 256 256
                s2_pred_l_conf = s2_pred_l_probs.max(dim=1)[0]
                s2_pred_u_probs = torch.softmax(s2_pred_u_logits, dim=1) # 8 4 256 256
                s2_pred_u_pseudo = torch.argmax(s2_pred_u_probs, dim=1) # 8 256 256
                s2_pred_u_conf = s2_pred_u_probs.max(dim=1)[0]

                s2_pred_uA1_logits = student2(uimageA1)['out']
            
            # AdaMix
            s1_imageA1, s1_label, s1_pred_l_conf = s1_adamix_l(oimage=imageA1, aimage=laimageA1, olabel=label.squeeze(1).long(), alabel=lalabel.squeeze(1).long(), oconf=s1_pred_l_conf, aconf=s1_pred_la_conf, prediction=s1_pred_l_logits, cur_step=idx + len(pbar) * (epoch-1))
            s1_uimageA1, s2_pred_u_pseudo, s2_pred_u_conf = s1_adamix_u(oimage=uimageA1, aimage=uaimageA1, olabel=s2_pred_u_pseudo, alabel=s2_pred_ua_pseudo, oconf=s2_pred_u_conf, aconf=s2_pred_ua_conf, prediction=s1_pred_uA1_logits, cur_step=idx + len(pbar) * (epoch-1))
            
            s2_imageA1, s2_label, s2_pred_l_conf = s2_adamix_l(oimage=imageA1, aimage=laimageA1, olabel=label.squeeze(1).long(), alabel=lalabel.squeeze(1).long(), oconf=s2_pred_l_conf, aconf=s2_pred_la_conf, prediction=s2_pred_l_logits, cur_step=idx + len(pbar) * (epoch-1))
            s2_uimageA1, s1_pred_u_pseudo, s1_pred_u_conf = s2_adamix_u(oimage=uimageA1, aimage=uaimageA1, olabel=s1_pred_u_pseudo, alabel=s1_pred_ua_pseudo, oconf=s1_pred_u_conf, aconf=s1_pred_ua_conf, prediction=s2_pred_uA1_logits, cur_step=idx + len(pbar) * (epoch-1))
            
            s1_pred_l_logits = student1(s1_imageA1)['out']
            s1_pred_uA1_logits = student1(s1_uimageA1)['out']
            s2_pred_l_logits = student2(s2_imageA1)['out']
            s2_pred_uA1_logits = student2(s2_uimageA1)['out']


            # supervised path
            loss_s = (criterion_s(s1_pred_l_logits, s1_label.long()) + criterion_s(s2_pred_l_logits, s2_label.long())) / 2. 
            # unsupervised path
            loss_u = (criterion_u(s1_pred_uA1_logits, s2_pred_u_pseudo.detach(), pixel_mask=(s2_pred_u_conf >= conf_threshold).float()) \
                    + criterion_u(s2_pred_uA1_logits, s1_pred_u_pseudo.detach(), pixel_mask=(s1_pred_u_conf >= conf_threshold).float())) / 2.

            loss = (loss_s + loss_u) / 2.
            
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            included_ratio = (s1_pred_u_conf >= conf_threshold).sum() / s1_pred_u_conf.numel()

            writer.add_scalar('train_s_loss', loss_s.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_u_loss', loss_u.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('included_ratio', included_ratio, idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_loss', loss.item(), idx + len(pbar) * (epoch-1))
            if idx % args.log_freq == 0:
                logger.info("Train: Epoch/Epochs {}/{}, "
                            "iter/iters {}/{}, "
                            "loss {:.3f}, loss_s {:.3f}, loss_u {:.3f}, included_ratio {:.3f}"
                            .format(epoch, args.num_epochs, idx, len(pbar), loss.item(), loss_s.item(), loss_u.item(),
                                    included_ratio))
            epoch_metrics.train_loss.append(loss.item())
            epoch_metrics.train_s_loss.append(loss_s.item())
            epoch_metrics.train_u_loss.append(loss_u.item())
        metrics.train_loss.append(np.mean(epoch_metrics.train_loss))
        metrics.train_s_loss.append(np.mean(epoch_metrics.train_s_loss))
        metrics.train_u_loss.append(np.mean(epoch_metrics.train_u_loss))
        ############################
        # Validation
        ############################
        epoch_metrics.val_loss = []
        iter_val_dataloader = iter(val_dataloader)
        if is_debug:
            val_pbar = range(10)
        else:
            val_pbar = range(val_iters)
        student1.eval()
        with torch.no_grad():
            for idx in val_pbar:
                image, label = next(iter_val_dataloader)
                image, label = image.to(device), label.to(device)
                pred = student1(image)['out']
                loss = criterion_s(pred, label.squeeze(1).long())
                writer.add_scalar('val_loss', loss.item(), idx + len(val_pbar) * (epoch-1))
                if idx % args.log_freq == 0:
                    logger.info("Val: Epoch/Epochs {}/{}\t"
                                "iter/iters {}/{}\t"
                                "loss {:.3f}".format(epoch, args.num_epochs, idx, len(val_pbar),
                                                     loss.item()))
                epoch_metrics.val_loss.append(loss.item())
        metrics.val_loss.append(np.mean(epoch_metrics.val_loss))

        # Save Model
        if np.mean(epoch_metrics.val_loss) <= best_loss:
            best_epoch = epoch
            best_loss = np.mean(epoch_metrics.val_loss)
            torch.save(student1.state_dict(), save_path + 'best.pth'.format(best_epoch))
        torch.save(student1.state_dict(), save_path + 'last.pth'.format(best_epoch))
        logger.info("Average: Epoch/Epoches {}/{}, "
            "train epoch loss {:.3f}, "
            "val epoch loss {:.3f}, "
            "best loss {:.3f} at {}\n".format(epoch, args.num_epochs, np.mean(epoch_metrics.train_loss),
                                                np.mean(epoch_metrics.val_loss), best_loss, best_epoch))
    ############################
    # Save Metrics
    ############################
    data_frame = pd.DataFrame(
        data={'loss': metrics.train_loss,
              'loss_s': metrics.train_s_loss,
              'loss_u': metrics.train_u_loss,
              'val_loss': metrics.val_loss},
        index=range(1, args.num_epochs + 1))
    data_frame.to_csv(project_path + 'train_val_loss.csv', index_label='Epoch')
    plt.figure()
    plt.title("Loss During Training and Validating")
    plt.plot(metrics.train_loss, label="Train")
    plt.plot(metrics.val_loss, label="Val")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(project_path + 'train_val_loss.png')

    print(project_path)
    time_elapsed = time.time() - since
    logger.info('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('TRAINING FINISHED!')


if __name__ == '__main__':
    main()

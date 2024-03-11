import timm
import sys
import torch
import torch.utils.data
import torchvision.models

import argparse
import numpy as np
import torch.nn as nn
import random
from trainer import Trainer


parser = argparse.ArgumentParser(description='DYSON: Dynamic Feature Space Self-Organization for Online Task-Free '
                                             'Class Incremental Learning')

# Exp. settings.
parser.add_argument('--batch_size', default=50, type=int, help='Batch size for training')
parser.add_argument('--epoch', default=1, type=int, help='epoch for training')
parser.add_argument('--gpu', default='9', type=str, help='GPU id to use')
parser.add_argument('--save_path', default='model_saved_check/', type=str, help='save files directory')

parser.add_argument('--learning_rate', default=2e-5, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=0.000005, type=float, help='weight_decay')

# dataset setting
parser.add_argument('--data_name', default='cifar10', choices=['cifar100', 'cifar10', 'cub200', 'core50'],
                    help='Dataset name to use')
parser.add_argument('--schedule_type', default='gaussian', choices=['split', 'gaussian'], help='schedule type')
parser.add_argument('--split_num', default=10, type=int, help='if schedule is split, session number')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')

# backbone
parser.add_argument('--vit_backbone', default=1, type=int, help='VIT as backbone')
parser.add_argument('--k_nearest', default=20, type=int, help='number of neighbours in NCM')

# pseudo setting
parser.add_argument('--pseudo_weight', default=1, type=float, help='protoAug loss weight, 0 means no pseudo method')
parser.add_argument('--pseudo_mode', default='follow_bs', choices=['follow_bs', 'all', 'customize'],
                    help='mode of pseudo number')
parser.add_argument('--pass_pseudo', default=0, type=float, help='protoAug loss with PASS or not')
parser.add_argument('--pseudo_rate', default=1, type=float, help='protoAug number')
parser.add_argument('--customize_rate', default=0.1, type=float, help='usage rate of pseudo samples')

# loss weight setting
parser.add_argument('--proto_align_weight', default=5, type=float, help='p_weight')
parser.add_argument('--pseudo_align_weight', default=5, type=float, help='pseudo_weight')

# memory
parser.add_argument('--mem_size', default=0, type=int, help='number of samples stored per-class')
parser.add_argument('--mem_lr', default=5e-4, type=float, help='memory samples loss weight')
parser.add_argument('--temp', default=1, type=float, help='trianing time temperature')

# few-shot
parser.add_argument('--lowdata_rate', default=1, type=float, help='few-shot learning')

# variance
parser.add_argument('--var_enforce', default=0, type=int, help='use variance enforce or not')

args = parser.parse_args()
print(args)


def main():
    SEED = 1993
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
    file_name = args.data_name + '_' + str(args.data_name)

    if args.data_name == 'cifar100':
        if args.vit_backbone:
            feature_extractor = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
            feature_dim = 384
        else:
            feature_extractor = torchvision.models.resnet50(pretrained=True)
            # feature_extractor = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            feature_extractor.fc = nn.Identity()
            feature_dim = 2048
        args.total_nc = 100
    elif args.data_name == 'cub200':
        if args.vit_backbone:
            feature_extractor = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
            feature_dim = 384
        else:
            feature_extractor = torchvision.models.resnet50(pretrained=True)
            feature_extractor.fc = nn.Identity()
            feature_dim = 2048
        args.total_nc = 200
    elif args.data_name == 'cifar10':
        # online backbone -- resnet 18
        # feature_extractor = torchvision.models.resnet18(pretrained=True)
        # feature_extractor.fc = nn.Identity()
        # feature_dim = 512
        if args.vit_backbone:
            feature_extractor = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
            feature_dim = 384
        else:
            feature_extractor = torchvision.models.resnet50(pretrained=True)
            # feature_extractor = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            feature_extractor.fc = nn.Identity()
            feature_dim = 2048
        args.total_nc = 10
    elif args.data_name == 'core50':
        feature_extractor = torchvision.models.resnet18(pretrained=True)
        feature_extractor.fc = nn.Identity()
        feature_dim = 512
        args.total_nc = 50
    else:
        sys.exit(0)
    feature_extractor.eval()
    model = Trainer(args, file_name, feature_extractor, device, feature_dim)
    model.setup_data(shuffle=True, seed=1993)
    model.beforeTrain()
    model.train()
    model.afterTrain()


if __name__ == "__main__":
    main()

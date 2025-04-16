import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, UAVCLDataset
from models.resnet_cl4nav import ResNetCL4Nav
from cl4nav import CL4Nav

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CL4Nav')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-rgb-path', metavar='RGB_DIR', default='./datasets/RGB',
                    help='path to RGB dataset')
parser.add_argument('-depth-path', metavar='DEPTH_DIR', default='./datasets/Depth',
                    help='path to Depth dataset')
parser.add_argument('-img-size', type=int, default='224', help='input image size')
parser.add_argument('-dataset-name', default='uav',
                    help='dataset name', choices=['stl10', 'cifar10', 'uav'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-dir', type=str, default='runs', help='Directory to save training logs and models')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--pretrained-model-path', default="path/to/your/pretrained/model", help='Load pretrained model')


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # ContrastiveLearningDataset is a configurable dataset class for loading either STL-10 or CIFAR-10 depending on user-specified arguments.
    # dataset = ContrastiveLearningDataset(args.data)
    train_dataset = UAVCLDataset(args.img_size, args.rgb_path, args.depth_path)

    # train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetCL4Nav(base_model=args.arch, out_dim=args.out_dim)

    # Load pretrained weights (optional)
    if args.pretrained_model_path:
        checkpoint = torch.load(args.pretrained_model_path, map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded pretrained model from {args.pretrained_model_path}")

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        cl4nav = CL4Nav(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        cl4nav.train(train_loader)


if __name__ == "__main__":
    main()

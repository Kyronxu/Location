# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2024/11/16 15:13
# Author     ：XuJ1E
# version    ：python 3.8
# File       : main_fer.py
"""
import sys
import time
sys.path.append('..')
import torch
import yaml
import argparse
import numpy as np
from loguru import logger
from engine.dataset import DualDataset, custom_collate
from engine.evaluate_epoch import evaluate
from engine.train_epoch import train
from modules.models import prepare_model
from datasets import load_dataset
from sklearn.metrics import confusion_matrix
from engine.loss import LabelSmoothingCrossEntropy, CenterLoss
from torchvision import transforms, datasets


parser = argparse.ArgumentParser(description="Train a model using MixUp and CutMix augmentations")
parser.add_argument('--save_path', type=str, required=True, help='Path to save the model checkpoints')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
parser.add_argument('--ema_decay', type=float, default=0.999, help='Number of workers for data loading')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--save_every', type=int, default=25, help='Save model checkpoints every specified number of steps')
parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
parser.add_argument('--transform_type', choices=['simple', 'hard'], default='simple', help='Choose simple or hard transformations')
parser.add_argument('--test_split_percent', type=float, default=0.2, help='Percentage of training set to keep aside for the test set')
parser.add_argument('--model_name', choices=['dinov2', 'vit','dinov2base', 'convnext'], default='convnext', help='Choose the model architecture: dinov2 or vit')
parser.add_argument('--num_classes', type=int, default=7, help='Number of output classes')
parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adamw'], default='adamw', help='Choose the optimizer: sgd, adam, or adamw')
parser.add_argument('--scheduler', choices=['cosineannealing', 'custom'], default='custom', help='Choose the scheduler: cosineannealing or custom')
parser.add_argument('--baseline', action='store_false', help='Evaluate the model on the validation set before training')
parser.add_argument('--evaluate', action='store_false', help='Evaluate and show accuracy, loss on saved model')
args = parser.parse_args()


def main(args):
    exp_path = os.path.join('experiments/', args.data_name + time.strftime("_%Y_%m_%d_%H_%M_%S", time.localtime()))
    os.makedirs(exp_path, exist_ok=True)
    logger.add(os.path.join(exp_path, 'log.txt'))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.transform_type == 'simple':
        transform1, transform2 = get_simple_transforms()
    elif agrs.transform_type == 'hard':
        transform1, transform2 = get_hard_transforms()
    else:
        raise ValueError('Invalid transform type')

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])])
    
train_data = DualDataset(load_dataset(args.data_path)['train'], transform1=transform1, transform2=transform2)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           collate_fn=custom_collate)
val_data = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transforms.Compose([transforms.Resize((224,224)),
                                                                                                transforms.ToTensor(),
                                                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                                std=[0.229, 0.224, 0.225])]))
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True)

student_model, teacher_model = prepare_models(args.model_name, num_classes=args.num_classes)
student_model = freeze_student_except_last_layer(student_model)

optimizer = build_optimizer(args.optimizer, student_model)
T_max = args.epochs * len(train_loader)
scheduler = build_scheduler(args.scheduler, optimizer, T_max, args.epochs, train_loader)
criterion = LabelSmoothingCrossEntropy()

writer = SummaryWriter(log_dir=args.exp_path)
logger.add(os.path.join(args.exp_path, 'log.txt'))
logger.info(f"Starting training for {args.epochs} epochs")
train(student_model, teacher_model, train_loader, val_loader,
                    optimizer, scheduler, criterion, device,
                    args.grad_accum_steps, args.ema_decay,
                    args.save_path, args.save_every, args.baseline,
                    args.evaluate, writer)

val_metric, val_loss, correct = evaluate(student_model, val_loader, device)
logger.info(f"Validation accuracy: {val_metric:.4f}%, Validation loss: {val_loss:.4f}")


if __name__ == '__main__':
    main(args)


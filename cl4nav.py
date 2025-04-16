import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint, save_onnx

torch.manual_seed(0)



class CL4Nav(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.output_dir = os.path.join(self.args.log_dir, f'0415_simulation_{self.args.epochs}_epochs_test2')
        os.makedirs(self.output_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.output_dir)
        logging.basicConfig(filename=os.path.join(self.output_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        # Defining the shape of the input tensor for exporting ONNX models
        self.input_tensor = torch.randn(1, 3, 224, 224).to(self.args.device)  # Example input, assuming input size (1, 3, 224, 224)

    def info_nce_loss(self, features):

        labels = torch.arange(self.args.batch_size)
        labels = labels.repeat_interleave(2)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.output_dir, self.args)

        n_iter = 0
        logging.info(f"Start CL4Nav training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for rgb_images, depth_images in train_loader:
                if depth_images.shape[1] == 1:
                    depth_images = depth_images.repeat(1, 3, 1, 1)
                rgb_images = rgb_images.to(self.args.device)
                depth_images = depth_images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    rgb_features = self.model(rgb_images)
                    depth_features = self.model(depth_images)

                    features = torch.zeros(2 * self.args.batch_size, self.args.out_dim).to(self.args.device)
                    features[0::2] = rgb_features
                    features[1::2] = depth_features
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

            if epoch_counter % 20 == 0:
                # save model checkpoints
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
                save_checkpoint({
                    'epoch': self.args.epochs,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
                logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

                onnx_filename = "model_{:03d}.onnx".format(epoch_counter)
                save_onnx(self.model, self.input_tensor, save_dir=self.writer.log_dir, filename=onnx_filename)
                logging.info(f"ONNX model saved at {self.writer.log_dir} for epoch {epoch_counter}.")

        logging.info("Training has finished.")


import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
from torch.utils import data as dt
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import warnings
warnings.simplefilter("ignore")

import pathlib
# BASE_DIR will be like '/home/jovyan/DemoExample/'
BASE_DIR = str(pathlib.Path(__file__).parent.absolute())
print(f"Working dir: {BASE_DIR}")
    
class CNNClassifier(nn.Module):
    """Custom module for a simple convnet classifier"""

    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

    
def train(epoch, clf, optimizer, train_loader, device, writer, rank):
    clf.train()  # set model in training mode (need this because of dropout)

    # dataset API gives us pythonic batching
    for batch_id, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)
        # forward pass, calculate loss and backprop!
        optimizer.zero_grad()
        preds = clf(data)
        if (batch_id == 0) and (epoch == 0) and (rank == 0):
            writer.add_graph(clf, data)
        loss = F.nll_loss(preds, target)
        loss.backward()

        optimizer.step()

        if (batch_id % 100 == 0) and (rank == 0):
            print(f'train loss = {loss.item()}')
            writer.add_scalar('Train', loss.item(), epoch * len(train_loader) + batch_id)


def test(epoch, clf, test_loader, device, writer, rank):
    clf.eval()  # set model in inference mode (need this because of dropout)
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = clf(data)
        test_loss += F.nll_loss(output, target).item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    
    test_loss = test_loss
    test_loss /= (len(test_loader) // dist.get_world_size())  # loss function already averages over batch size
    accuracy = 100. * correct / (len(test_loader.dataset) // dist.get_world_size())
    if rank == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset) / dist.get_world_size(),
            accuracy))

    
def run(rank, size, local_rank):
    model = CNNClassifier()
    print(f'local rank = {local_rank}, rank = {rank}')
    device = torch.device(f'cuda:{local_rank}')
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    data = np.load(BASE_DIR + '/mnist.npz')
    mnist_images_train = np.expand_dims(data['x_train'], 1)
    mnist_labels_train = data['y_train']

    mnist_images_test = np.expand_dims(data['x_test'], 1)
    mnist_labels_test = data['y_test']
    data.close()

    dataset_train = dt.TensorDataset(torch.Tensor(mnist_images_train), torch.Tensor(mnist_labels_train).long())
    dataset_test = dt.TensorDataset(torch.Tensor(mnist_images_test), torch.Tensor(mnist_labels_test).long())
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train, num_replicas=dist.get_world_size(), rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_test, num_replicas=dist.get_world_size(), rank=rank)
    
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=50, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=50, sampler=test_sampler)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    current_time = datetime.now().strftime("%Y%m%d-%H_%M")
    if rank == 0:
        writer = SummaryWriter(log_dir=BASE_DIR + 'logs/log_' + current_time)
    else:
        writer = None


    num_epochs = 3

    if rank == 0:
        print(f'Start train with num epoch = {num_epochs}')

    for epoch in range(num_epochs):
        if rank == 0:
            print("Epoch %d" % epoch)
        train(epoch, model, optimizer, train_loader, device, writer, rank)
        test(epoch, model, test_loader, device, writer, rank)
        if rank == 0:
            torch.save(model.state_dict(), BASE_DIR +'logs/log_' + current_time + f"/model_epoch_{epoch}.bin")
    if rank == 0:
        writer.close()
    

def init_processes(fn, local_rank, backend='nccl'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend)
    fn(dist.get_rank(), dist.get_world_size(), local_rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    init_processes(run, args.local_rank, backend='nccl')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



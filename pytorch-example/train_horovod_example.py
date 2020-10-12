import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data as dt
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import horovod.torch as hvd
from datetime import datetime
import numpy as np
import warnings

warnings.simplefilter("ignore")

import pathlib
# BASE_DIR will be like '/home/jovyan/DemoExample/'
BASE_DIR = str(pathlib.Path(__file__).parent.absolute())
print(f"Working dir: {BASE_DIR}")

data = np.load(BASE_DIR + '/mnist.npz', allow_pickle=True)
mnist_images_train = np.expand_dims(data['x_train'], 1)
mnist_labels_train = data['y_train']

mnist_images_test = np.expand_dims(data['x_test'], 1)
mnist_labels_test = data['y_test']
data.close()

dataset_train = dt.TensorDataset(torch.Tensor(mnist_images_train), torch.Tensor(mnist_labels_train).long())
dataset_test = dt.TensorDataset(torch.Tensor(mnist_images_test), torch.Tensor(mnist_labels_test).long())


# define CNN
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


# init horovod and GPU
hvd.init()
torch.cuda.set_device(hvd.local_rank())
device = torch.device(f'cuda:{hvd.local_rank()}')
print(f'process local rank = {hvd.local_rank()}, global rank =  {hvd.rank()}')

clf = CNNClassifier()
clf.to(device)

train_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset_train, num_replicas=hvd.size(), rank=hvd.rank())

test_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset_test, num_replicas=hvd.size(), rank=hvd.rank())


train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=50, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=50, sampler=test_sampler)

# add Tensorboard writer
current_time = datetime.now().strftime("%Y%m%d-%H_%M")
if hvd.rank() == 0:
    writer = SummaryWriter(log_dir=BASE_DIR + 'logs/log_' + current_time)
else:
    writer = None

optimizer = optim.SGD(clf.parameters(), lr=0.01, momentum=0.5)

# Add Horovod Distributed Optimizer
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=clf.named_parameters())
# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(clf.state_dict(), root_rank=0)


def train(epoch, clf, optimizer, writer, rank):
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


def test(epoch, clf, writer, rank):
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
    test_loss /= (len(test_loader) // hvd.size())  # loss function already averages over batch size
    accuracy = 100. * correct / (len(test_loader.dataset) // hvd.size())
    if rank == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset) / hvd.size(),
            accuracy))


num_epochs = 3

if hvd.rank() == 0:
    print(f'Start train with num epoch = {num_epochs}')

for epoch in range(num_epochs):
    if hvd.rank() == 0:
        print("Epoch %d" % epoch)
    train(epoch, clf, optimizer, writer, hvd.rank())
    test(epoch, clf, writer, hvd.rank())
    if hvd.rank() == 0:
        torch.save(clf.state_dict(), BASE_DIR + 'logs/log_' + current_time + f"/model_epoch_{epoch}.bin")
if hvd.rank() == 0:
    writer.close()

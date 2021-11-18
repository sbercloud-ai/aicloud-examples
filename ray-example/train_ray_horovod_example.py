import pathlib
import numpy as np
import horovod.torch as hvd
import mlflow.pytorch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import yaml
from torch.utils import data as dt
from ray import tune
from ray.tune.integration.horovod import DistributedTrainableCreator
from torchvision import datasets, transforms

CUDA = torch.cuda.is_available()
BASE_DIR = str(pathlib.Path(__file__).parent.absolute())

with open(f"{BASE_DIR}/config.yaml") as f:
    base_config = yaml.load(f)

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


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def training_function(config):
    # Horovod: initialize library. Init horovod and GPU
    hvd.init()

    if CUDA:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())

    # DATASET
    data = np.load(BASE_DIR + '/mnist.npz', allow_pickle=True)
    mnist_images_train = np.expand_dims(data['x_train'], 1)
    mnist_labels_train = data['y_train']

    mnist_images_test = np.expand_dims(data['x_test'], 1)
    mnist_labels_test = data['y_test']
    data.close()

    dataset_train = dt.TensorDataset(torch.Tensor(mnist_images_train), torch.Tensor(mnist_labels_train).long())
    dataset_test = dt.TensorDataset(torch.Tensor(mnist_images_test), torch.Tensor(mnist_labels_test).long())
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train, num_replicas=hvd.size(), rank=hvd.rank())

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_test, num_replicas=hvd.size(), rank=hvd.rank())
    
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=50, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=50, sampler=test_sampler)

    model = CNNClassifier()
    lr_scaler = hvd.size()

    if CUDA:
        # Move model to GPU.
        model.cuda()

    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.SGD(model.parameters(), lr=config["lr"] * lr_scaler, momentum=0.5)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm. hvd.Compression.fp16
    compression = hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(), compression=compression,
    )

    def train(epoch):
        model.train()
        # Horovod: set epoch to sampler for shuffling.
        train_sampler.set_epoch(epoch)
        for data, target in train_loader:
            if CUDA:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        train_avg_loss = metric_average(loss.item(), "train_avg_loss")
        if hvd.rank() == 0:
            print(f"Train Epoch: {epoch} Avg_loss: {train_avg_loss}")
            # TODO save data for tensorboard

    def test(epoch):
        model.eval()
        test_loss = 0.0
        test_accuracy = 0.0

        for data, target in test_loader:
            if CUDA:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

        # Horovod: use test_sampler to determine the number of examples in
        # this worker's partition.
        test_loss /= len(test_sampler)
        test_accuracy /= len(test_sampler)

        # Horovod: average metric values across workers.
        test_loss = metric_average(test_loss, "avg_loss")
        test_accuracy = metric_average(test_accuracy, "avg_accuracy")
        if hvd.rank() == 0:
            print(
                "\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
                    test_loss, 100.0 * test_accuracy
                )
            )
            # Horovod: print output only on first rank.
        if hvd.rank() == 0:
            tune.report(loss=(test_loss), accuracy=test_accuracy)
            mlflow.log_metric("Test loss", test_loss, step=epoch)  # add mlflow metrics
            mlflow.log_metric(
                "Accuracy", test_accuracy, step=epoch
            )  # add mlflow metrics

    if hvd.rank() == 0:
        mlflow.set_tracking_uri("file:/home/jovyan/mlruns")
        mlflow.set_experiment("mlflow_example_default_ray_final")
        mlflow.start_run(nested=False, run_name="y")

    for epoch in range(1, config["epochs"] + 1):
        train(epoch)
        test(epoch)

    if hvd.rank() == 0:
        mlflow.end_run()


if __name__ == "__main__":
    trainable = DistributedTrainableCreator(
        training_function, num_slots=base_config["num_slots"], use_gpu=True
    )
    if base_config["hyperparameters"]["type"] == "grid":
        analysis = tune.run(
            trainable,
            num_samples=1,
            metric="loss",
            mode="min",
            config={
                "epochs": tune.grid_search(base_config["hyperparameters"]["epochs"]),
                "lr": tune.grid_search(base_config["hyperparameters"]["lr"]),
                "mode": "square",
                "x_max": 1.0,
            },
            fail_fast=True,
        )
    else:
        print(
            "Incorrect hyperparameters tuning type! Set correct parameter into config file"
        )
        exit(1)

    print("Best hyperparameters found were: ", analysis.best_config)

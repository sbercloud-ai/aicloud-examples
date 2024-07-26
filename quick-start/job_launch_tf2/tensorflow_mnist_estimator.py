# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import horovod.tensorflow.keras as hvd
import errno
import mlflow
import numpy as np
import os
import tensorflow as tf
import pathlib

# BASE_DIR will be like '/home/jovyan/DemoExample/'
BASE_DIR = str(pathlib.Path(__file__).parent.absolute())
print(f'Working dir: {BASE_DIR}')


# Hack due to our Internet-policies
def load_data(path=f'{BASE_DIR}/mnist.npz'):
    """Loads the MNIST dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    # Horovod: initialize Horovod.
    hvd.init()

    # Keras automatically creates a cache directory in ~/.keras/datasets for
    # storing the downloaded MNIST data. This creates a race
    # condition among the workers that share the same filesystem. If the
    # directory already exists by the time this worker gets around to creating
    # it, ignore the resulting exception and continue.
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
    if not os.path.exists(cache_dir):
        try:
            os.mkdir(cache_dir)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(cache_dir):
                pass
            else:
                raise
    
    # Download and load MNIST dataset.
    (train_data, train_labels), (eval_data, eval_labels) = load_data()
    
    # The shape of downloaded data is (-1, 28, 28), hence we need to reshape it
    # into (-1, 784) to feed into our network. Also, need to normalize the
    # features between 0 and 1.
#     train_data = np.reshape(train_data, (-1, 784)) / 255.0
#     eval_data = np.reshape(eval_data, (-1, 784)) / 255.0
    train_data = train_data.reshape(-1, 28, 28, 1) / 255.0
    eval_data = eval_data.reshape(-1, 28, 28, 1) / 255.0

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    model_dir = f'{BASE_DIR}/checkpoints_tf/mnist_convnet_model' if hvd.rank() == 0 else None

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(train_data, tf.float32),
                 tf.cast(train_labels, tf.int64))
    )
    train_dataset = train_dataset.repeat().shuffle(10000).batch(128)

    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax', name='softmax_tensor')
    ])

    # Horovod: adjust learning rate based on number of GPUs.
    scaled_lr = 0.001 * hvd.size()
    opt = tf.optimizers.Adam(scaled_lr)

    # Horovod: add Horovod DistributedOptimizer.
#     opt = hvd.DistributedOptimizer(
#         opt, backward_passes_per_step=1, average_aggregated_gradients=True)
    opt = hvd.DistributedOptimizer(opt)

    # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses hvd.DistributedOptimizer() to compute gradients.
    mnist_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                        optimizer=opt,
                        metrics=['accuracy'],
                        experimental_run_tf_function=False)

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3),
        
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True,
            verbose=(1 if hvd.rank() == 0 else 0)
        ),
    ]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        checkpoint_path = f'{BASE_DIR}/checkpoints_tf/mnist_convnet_model/checkpoint.ckpt'
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=1
        ))

    # Horovod: write logs on worker 0.
    verbose = 1 if hvd.rank() == 0 else 0

    # Train the model.
    # Horovod: adjust number of steps based on number of GPUs.
    mnist_model.fit(train_dataset, steps_per_epoch=500 // hvd.size(), callbacks=callbacks, epochs=100, verbose=verbose)

    # Evaluate the model and print results
    eval_results = mnist_model.evaluate(tf.cast(eval_data, tf.float32), tf.cast(eval_labels, tf.int64), batch_size=128)
    print(eval_results)
    if hvd.rank() == 0:
        # Here we log metrics of our model to mlflow
        mlflow.set_tracking_uri('file:/home/jovyan/mlruns')
        mlflow.set_experiment("quick-start/job_launch_tf2")
        with mlflow.start_run(nested=True) as run:
            mlflow.log_metric("loss", eval_results[0])
            mlflow.log_metric("accuracy", eval_results[1])
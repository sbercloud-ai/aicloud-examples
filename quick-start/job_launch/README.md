# Пример обучения на GPU в регионе с помощью `client-lib` 

В этом примере происходит обучение сверточной нейронной сети с помощью библиотек `Horovod` и `Tensorflow 1` на датасете MNIST посредством создания и отправки задачи в регион Christofari.V100.

Пример включает в себя:

 * [mnist.npz](mnist.npz) — датасет с рукописными цифрами.
 * [quick-start.ipynb](quick-start.ipynb) — Jupyter-ноутбук.
 * [requirements.txt](requirements.txt) — файл с зависимостями, который используется для сборки кастомного контейнера.
 * [tensorflow_mnist_estimator.py](tensorflow_mnist_estimator.py) — код модели на `TensorFlow 1` и `Horovod`.

Для запуска загрузите ноутбук [quick-start.ipynb](quick-start.ipynb) в веб-интерфейс [Jupyter Server, созданного в ML Space](https://mlspace.aicloud.sbercloud.ru/mlspace/jupyter-server).
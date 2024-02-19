# Пример обучения на GPU в регионе с помощью `client-lib` 

В этом примере происходит обучение сверточной нейронной сети с помощью библиотек `Keras`, `TensorFlow 2` и `Horovod` на датасете MNIST посредством создания и отправки задачи в регионы Christofari.V100 и Christofari.A100.

Пример включает в себя:

 * [mnist.npz](mnist.npz) — датасет с рукописными цифрами.
 * [quick-start-v100.ipynb](quick-start-v100.ipynb) — Jupyter-ноутбук для загрузки на [сервер](https://mlspace.aicloud.sbercloud.ru/mlspace/jupyter-server).
 * [quick-start-a100.ipynb](quick-start-a100.ipynb) — Jupyter-ноутбук для загрузки на регион A100 [сервер](https://mlspace.aicloud.sbercloud.ru/mlspace/jupyter-server).
 * [requirements.txt](requirements.txt) — файл с зависимостями, который используется для сборки кастомного контейнера.
 * [tensorflow_mnist_estimator.py](tensorflow_mnist_estimator.py) — код модели на `Keras`, `TensorFlow 2` и `Horovod`.

Для запуска загрузите ноутбук [quick-start.ipynb](quick-start.ipynb) в веб-интерфейс [Jupyter Server, созданного в ML Space](https://mlspace.aicloud.sbercloud.ru/mlspace/jupyter-server).

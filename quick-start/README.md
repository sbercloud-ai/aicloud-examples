# Знакомство с сервисом AICloud

Текущий пример позволяет обучиться отправке задач для распределенной тренировки моделей машинного обучения.

Для запуска примера достаточно загрузить notebook [quick-start.ipynb](quick-start.ipynb) в веб-интерфейс [Jupyter Server'а внутри AICloud](https://aicloud.sbercloud.ru/_/jupyter/).

В этом примере происходит обучение сверточной нейронной сети с помощью библиотек `Horovod` и `Tensorflow 1` на датасете `Mnist` через создание задачи на кластере Cristofari.

Пример включает в себя:

 * [mnist.npz](mnist.npz) (датасет с рукописными цифрами)
 * [quick-start.ipynb](quick-start.ipynb) (Jupyter Notebook для загрузки на сервер aicloud.sbercloud.ru)
 * [requirements.txt](requirements.txt) (файл с зависимостями)
 * [tensorflow_mnist_estimator.py](tensorflow_mnist_estimator.py) (код модели на `TensorFlow 1` и `Horovod`)


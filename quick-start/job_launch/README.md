# Знакомство с сервисом AI Cloud от SberCloud

Пример из данного раздела позволит пользователям научиться отправке задач для распределенного обучения моделей.

Для запуска примера достаточно загрузить ноутбук [quick-start.ipynb](quick-start.ipynb) в веб-интерфейс [Jupyter-сервера внутри AI Cloud](https://aicloud.sbercloud.ru/_/jupyter/).

В этом примере происходит обучение сверточной нейронной сети с помощью библиотек `Horovod` и `Tensorflow 1` на датасете `Mnist` посредством создания и отправки задачи на кластер "Кристофари".

Пример включает в себя:

 * [mnist.npz](mnist.npz) (датасет с рукописными цифрами)
 * [quick-start.ipynb](quick-start.ipynb) (Jupyter-ноутбук для загрузки на сервер aicloud.sbercloud.ru)
 * [requirements.txt](requirements.txt) (файл с зависимостями, который используется для сборки кастомного контейнера)
 * [tensorflow_mnist_estimator.py](tensorflow_mnist_estimator.py) (код модели на `TensorFlow 1` и `Horovod`)


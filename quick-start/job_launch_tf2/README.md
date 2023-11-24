# Знакомство с сервисом ML Space от Cloud.ru

Пример из данного раздела позволит пользователям научиться отправке задач для распределенного обучения моделей.

Для запуска примера достаточно загрузить ноутбук [quick-start.ipynb](quick-start.ipynb) в веб-интерфейс [Jupyter-сервера внутри ML Space](https://k1-nb.ai.cloud.ru/).

В этом примере происходит обучение сверточной нейронной сети с помощью библиотек `Keras`, `TensorFlow 2` и `Horovod` на датасете `MNIST` посредством создания и отправки задачи на кластер "Кристофари".

Пример включает в себя:

 * [mnist.npz](mnist.npz) (датасет с рукописными цифрами)
 * [quick-start-v100.ipynb](quick-start-v100.ipynb) (Jupyter-ноутбук для загрузки на [сервер](https://k1-nb.ai.cloud.ru/))
 * [quick-start-a100.ipynb](quick-start-a100.ipynb) (Jupyter-ноутбук для загрузки на регион A100 [сервер](https://k1-nb.ai.cloud.ru/))
 * [requirements.txt](requirements.txt) (файл с зависимостями, который используется для сборки кастомного контейнера)
 * [tensorflow_mnist_estimator.py](tensorflow_mnist_estimator.py) (код модели на `Keras`, `TensorFlow 2` и `Horovod`)


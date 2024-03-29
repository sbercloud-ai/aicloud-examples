# Пример обучения на GPU в регионе с помощью `client-lib` 

Пример из данного раздела позволит пользователям научиться отправке задач для распределенного обучения моделей.

Для запуска примера [создайте](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__create-new-jupyter-server.html) или [подключитесь к уже существующему Jupyter Server](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__connect-to-exist.html).

В этом примере происходит обучение сверточной нейронной сети с помощью библиотек `Horovod` и `Tensorflow 1` на датасете `Mnist` посредством создания и отправки задачи на кластер "Кристофари".

После подключения к Jupyter Server необходимо загрузить файлы через веб-интерфейс Jupyter Server внутри ML Space:

 * [mnist.npz](mnist.npz) — датасет с рукописными цифрами.
 * [quick-start.ipynb](quick-start.ipynb) — Jupyter-ноутбук.
 * [requirements.txt](requirements.txt) — файл с зависимостями, который используется для сборки кастомного контейнера.
 * [tensorflow_mnist_estimator.py](tensorflow_mnist_estimator.py) — код модели на `TensorFlow 1` и `Horovod`.

Для запуска загрузите ноутбук [quick-start.ipynb](quick-start.ipynb) в веб-интерфейс [Jupyter Server, созданного в ML Space](https://mlspace.aicloud.sbercloud.ru/mlspace/jupyter-server).
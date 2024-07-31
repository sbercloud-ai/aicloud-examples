# Пример обучения моделей из Jupyter Server на выделенных GPU

Этот пример поможет научиться решать задачу классификации на учебном датасете MNIST. 

Использован [DataParallel](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html), experiment tracking осуществляется с помощью библиотеки `MLflow`.

Для запуска примера:

1. [Создайте новый Jupyter Server](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__create-new-jupyter-server.html) или [подключитесь к уже существующему Jupyter Server](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__connect-to-exist.html), используя документацию.

2. Создайте новую папку и загрузите следующие файлы [через веб-интерфейс Jupyter Server](https://mlspace.aicloud.sbercloud.ru/mlspace/jupyter-server) на платформе ML Space:

   * [pytorch_tensorboard_mlflow.ipynb](pytorch_tensorboard_mlflow.ipynb) — обучение модели из Jupyter-ноутбука, подключенного к GPU.

3. Запустите ноутбук [pytorch_tensorboard_mlflow.ipynb](pytorch_tensorboard_mlflow.ipynb) в интерфейсе Jupyter Server.

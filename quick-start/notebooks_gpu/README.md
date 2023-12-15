# Запуск обучения моделей из Jupyter Server, подключенного к GPU

В данном примере показано как использовать [DataParallel](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html).

Experiment tracking осуществляется с помощью библиотеки `MLflow`.

Для запуска примера [создайте](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__create-new-jupyter-server.html) или [подключитесь к уже существующему Jupyter Server](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__connect-to-exist.html).

После подключения к Jupyter Server необходимо загрузить файл через веб-интерфейс Jupyter Server внутри ML Space:

 * [pytorch_tensorboard_mlflow.ipynb](pytorch_tensorboard_mlflow.ipynb) - обучение модели из Jupyter-ноутбука, подключенного к GPU.

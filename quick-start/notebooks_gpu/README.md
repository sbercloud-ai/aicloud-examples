# Пример обучения моделей из Jupyter Server на выделенных GPU

В примере решается задача классификации на учебном датасете MNIST. Использован [DataParallel](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html), experiment tracking осуществляется с помощью библиотеки `MLflow`.

Для запуска примера загрузите в веб-интерфейс [Jupyter Server внутри AI Cloud](https://aicloud.sbercloud.ru/_/jupyter/) следующие файлы:

 * [pytorch_tensorboard_mlflow.ipynb](pytorch_tensorboard_mlflow.ipynb) — обучение модели из Jupyter-ноутбука, подключенного к GPU.
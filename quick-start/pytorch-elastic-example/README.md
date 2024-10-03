# Пример обучения на GPU с использованием PyTorch Elastic 

В этом примере рассмотрена отправка задачи обучения на GPU.

Модель обучается с помощью библиотеки `PyTorch`на датасете `MNIST` путем создания и отправки задачи обучения.

Для запуска примера:

1. [Создайте новый Jupyter Server](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__create-new-jupyter-server.html) или [подключитесь к уже существующему Jupyter Server](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__connect-to-exist.html), используя документацию.

2. Создайте новую папку и загрузите следующие файлы [через веб-интерфейс Jupyter Server](https://mlspace.aicloud.sbercloud.ru/mlspace/jupyter-server) на платформе ML Space:

   * [mnist.npz](mnist.npz)— датасет с рукописными цифрами;
   * [train_pt-elastic.ipynb](train_pt-elastic.ipynb) — Jupyter-ноутбук для запуска задачи обучения
   * [train_ddp_elastic_example-torch2.py](train_ddp_elastic_example-torch2.py) — код модели на `Pytorch`.

3. Запустите ноутбук [train_pt_elastic.ipynb](train_pt_elastic.ipynb) в интерфейсе Jupyter Server.

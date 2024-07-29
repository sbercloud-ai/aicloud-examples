# Пример обучения на GPU в регионе с помощью `client-lib` 

В этом примере рассмотрена отправка задачи обучения на GPU.

Модель обучается с помощью библиотеки `PyTorch`на датасете `MNIST` путем создания и отправки задачи обучения.

Для запуска примера:

1. [Создайте новый Jupyter Server](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__create-new-jupyter-server.html) или [подключитесь к уже существующему Jupyter Server](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__connect-to-exist.html), используя документацию.

2. Создайте новую папку и загрузите следующие файлы [через веб-интерфейс Jupyter Server](https://mlspace.aicloud.sbercloud.ru/mlspace/jupyter-server) на платформе ML Space:

   * [mnist.npz](mnist.npz)— датасет с рукописными цифрами;
   * [quick-start.ipynb](quick-start.ipynb) — Jupyter-ноутбук для загрузки на [сервер](https://console.cloud.ru/projects/);
   * [requirements.txt](requirements.txt) — файл с зависимостями, который используется для сборки кастомного контейнера;
   * [tensorflow_mnist_estimator.py](tensorflow_mnist_estimator.py) — код модели на `Pytorch`.

3. Запустите ноутбук [quick-start.ipynb](quick-start.ipynb) в интерфейсе Jupyter Server.

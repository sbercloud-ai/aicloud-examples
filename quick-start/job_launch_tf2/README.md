# Пример обучения на GPU в регионе с помощью `client-lib` 

Этот пример поможет научиться запускать задачи распределенного обучения моделей.

Обучается сверточная нейронная сеть с помощью библиотек `Keras`, `TensorFlow 2` и `Horovod` на датасете `MNIST` посредством создания и отправки задачи в регион размещения ресурсов.

Для запуска примера:

1. [Создайте новый Jupyter Server](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__create-new-jupyter-server.html) или [подключитесь к уже существующему Jupyter Server](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__connect-to-exist.html), используя документацию.

2. Создайте новую папку и загрузите следующие файлы [через веб-интерфейс Jupyter Server](https://mlspace.aicloud.sbercloud.ru/mlspace/jupyter-server) на платформе ML Space:

   * [mnist.npz](mnist.npz)— датасет с рукописными цифрами;
   * [quick-start-v100.ipynb](quick-start-v100.ipynb) — Jupyter-ноутбук для загрузки на [сервер](https://console.cloud.ru/projects/);
   * [quick-start-a100.ipynb](quick-start-a100.ipynb) — Jupyter-ноутбук для загрузки на регион A100 [сервер](https://console.cloud.ru/projects/);
   * [requirements.txt](requirements.txt) — файл с зависимостями, который используется для сборки кастомного контейнера;
   * [tensorflow_mnist_estimator.py](tensorflow_mnist_estimator.py) — код модели на `Keras`, `TensorFlow 2` и `Horovod`.

3. Запустите ноутбук [quick-start.ipynb](quick-start.ipynb) в интерфейсе Jupyter Server.

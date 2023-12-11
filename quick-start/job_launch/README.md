# Знакомство с сервисом ML Space от Cloud.ru

Пример из данного раздела позволит пользователям научиться отправке задач для распределенного обучения моделей.

Для запуска примера достаточно загрузить файл [quick-start.ipynb](quick-start.ipynb) в веб-интерфейс [Jupyter Server внутри ML Space](https://console.cloud.ru/projects/).

В этом примере происходит обучение сверточной нейронной сети с помощью библиотек `Horovod` и `Tensorflow 1` на датасете `Mnist` посредством создания и отправки задачи на кластер "Кристофари".

   - Если Jupyter Server не существует, создайте его по [инструкции](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__create-new-jupyter-server.html).
   - Если Jupyter Server создан, воспользуйтесь [инструкцией по подключению к существующему](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__connect-to-exist.html).

Пример включает в себя:

 * [mnist.npz](mnist.npz) (датасет с рукописными цифрами)
 * [quick-start.ipynb](quick-start.ipynb) (Jupyter Notebook)
 * [requirements.txt](requirements.txt) (файл с зависимостями, который используется для сборки кастомного контейнера)
 * [tensorflow_mnist_estimator.py](tensorflow_mnist_estimator.py) (код модели на `TensorFlow 1` и `Horovod`)


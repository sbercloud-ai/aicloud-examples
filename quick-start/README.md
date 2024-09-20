# Знакомство с платформой ML Space от Cloud.ru

В этом разделе приведены примеры создания и отправки задач для обучения моделей в ML Space. Примеры построены так, что для запуска их достаточно загрузить на платформу.

Схема дает представление о доступных способах обучения модели. На ней примеры сгруппированы по инструментам для обучения. Ниже находятся ссылки на каждый пример для Jupyter Notebook с кратким описанием. 

Использование Training Job API здесь не рассматривается. Подробнее об API ML Space, в том числе для задач обучения — в [быстром старте по API](../public-api-example/ml_space_public_api.ipynb) и [пользовательской документации](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/api.html).

![](../img/qs_training_types.png)

## 1. Обучение напрямую на выделенных GPU

При таком варианте обучения максимальное количество выделенных GPU — 16. Подходит для пользователей, не знакомых с библиотекой Horovod.

Оплата взимается, пока Jupyter Server не будет удален, даже если он не используется.

* [Обучение модели в Jupyter Server](notebooks_gpu) показывает, как обучать модель напрямую из Jupyter Notebook, подключенного к GPU, задействуя `Pytorch`, `Tensorboard` и `MLFlow`.

  В примере решается задача классификации на учебном датасете MNIST.

Подробнее о создании Jupyter Server — в [документации](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__create-new-jupyter-server.html).

## 2. Обучение в регионе с помощью `client_lib` 

При таком варианте обучения можно задействовать до 1000 GPU. Оплата происходит за фактическое время исполнения задачи: от старта до окончания обучения.

* [Обучение модели на GPU с применением PyTorch](job_launch_pt) показывает, как создать и запустить задачу распределенного обучения (training job) с помощью `pytorch.distributed` и `PyTorch 2`.

  В примере решается задача классификации на учебном датасете MNIST.

* [Обучение модели на GPU с применением Tensorflow 2](job_launch_tf2) показывает, как создать и запустить задачу распределенного обучения (training job) на `Keras`, `Horovod` и `TensorFlow 2`.

  В примере решается задача классификации на учебном датасете MNIST. 

  Рассмотрено сохранение контрольных точек обучения (чекпоинтов).

* [Обучение модели на CPU](job_launch_cpu) показывает, как создать и запустить задачу, не требующую GPU.

  В примере решается задача предсказания цен домов.

  ## 3. Другие примеры обучения моделей

Дополнительные примеры обучения моделей под разные задачи:

 * [pytorch-example](/pytorch-example) — задача распределенного обучения Pytorch-модели с двумя типами запуска: стандартный `horovod` и дополнительный `pytorch`, он же `Pytorch.Distributed`.
 * [hugging-face-llm-example](hugging-face-llm-example) — работа с языковой моделью методами LoRA и PEFT, а также распределенное обучение с PyTorch Distributed Data Parallel (DDP).
 * [lightning-example](lightning-example) — использование PyTorch и PyTorch Lightning для задачи классификации изображений.
 * [pytorch-elastic-example](pytorch-elastic-example) — обучение модели на PyTorch и Elastic Learning с сохранением контрольных точек обучения (чекпоинтов).

Подробнее о работе с `client_lib` — в [документации](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/client-lib.html).

# Примеры работы с платформой ML Space от Cloud.ru

В репозитории приведены примеры использования платформы ML Space от начального до продвинутого уровня для решения ML-задач.

## Быстрый старт в обучении моделей

Базовые примеры размещены в директории [quick-start](quick-start). Они иллюстрируют, как обучить модель:

1. Напрямую из Jupyter Server, подключенного к GPU
   
   [Обучение в ноутбуке с GPU](quick-start/notebooks_gpu).

2. В регионе с помощью `client-lib` 

   [Обучение на PyTorch](quick-start/job_launch_pt).

   [Обучение на TensorFlow 2](quick-start/job_launch_tf2).

   [Обучение на CPU](quick-start/job_launch_cpu).

## Другие примеры обучения моделей

Дополнительные примеры обучения моделей под разные задачи:

 * [pytorch-example](pytorch-example) — задача распределенного обучения Pytorch-модели с двумя типами запуска: стандартный `horovod` и дополнительный `pytorch`, он же `Pytorch.Distributed`.
 * [ddp-example](ddp-example) — распределенное обучение с PyTorch Distributed Data Parallel (DDP).
 * [hugging-face-llm-example](hugging-face-llm-example) — работа с языковой моделью методами LoRA и PEFT.
 * [lightning-example](lightning-example) — использование PyTorch и PyTorch Lightning для задачи классификации изображений.
 * [pytorch-elastic-example](pytorch-elastic-example) — обучение модели на PyTorch и Elastic Learning с сохранением контрольных точек обучения (чекпоинтов).

## Препроцессинг данных

* Загрузка/выгрузка данных на S3 в [базовых примерах quick start](quick-start).
* [Работа с Rapids](rapids) — библиотекой, ускоряющей обработку датасетов на GPU.
* В ноутбуке `Spark_preproc.ipynb` поясняется, как c использованием ресурсов кластера Spark создать SparkSession и SparkContext, загрузить данные на S3 и выполнить препроцессинг этих данных.

## AutoML

В [стартовом ноутбуке для AutoML](automl) содержится объяснение, как обновить библиотеки `autowoe` и `lightautoml`, а также как загрузить туториалы из открытых репозиториев на GitHub для знакомства с функционалом библиотек.

## Public API V2

В [ноутбуке public-api-example](public-api-example) содержатся примеры того, как взаимодействовать с публичным API V2.

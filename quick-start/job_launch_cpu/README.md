# Пример обучения модели на CPU

В этом примере рассмотрена отправка задачи обучения моделей на CPU.

Модель обучается с помощью библиотек `XGBoost` на маленьком датасете `California House Pricing` путем создания и отправки CPU-задачи.
Рекомендуется использовать `XGBoost` на GPU либо `Spark`, если датасет большой.

Для запуска примера:

1. [Создайте новый Jupyter Server](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__create-new-jupyter-server.html) или [подключитесь к уже существующему Jupyter Server](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__connect-to-exist.html), используя документацию.

2. Создайте новую папку и загрузите следующие файлы [через веб-интерфейс Jupyter Server](https://mlspace.aicloud.sbercloud.ru/mlspace/jupyter-server) на платформе ML Space:

   * [cal_housing_py3.pkz](cal_housing_py3.pkz) — датасет `California House Pricing`;
   * [quick-start.ipynb](quick-start.ipynb) — Jupyter-ноутбук, который запустит задачу обучения;
   * [requirements.txt](requirements.txt) — файл с зависимостями, который используется для сборки кастомного образа;
   * [xgboost-boston-house-price.py](xgboost-boston-house-price.py) — код модели, который запускается из [quick-start.ipynb](quick-start.ipynb).

3. Запустите ноутбук [quick-start.ipynb](quick-start.ipynb) в интерфейсе Jupyter Server.

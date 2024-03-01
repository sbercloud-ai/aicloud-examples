# Пример обучения модели на CPU

Пример из данного раздела позволит пользователям научиться отправлять задачи для обучения моделей на CPU.

Для запуска примера [создайте](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__create-new-jupyter-server.html) или [подключитесь к уже существующему Jupyter Server](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__connect-to-exist.html).

В этом примере происходит обучение с помощью библиотек `XGBoost` на маленьком датасете `California House Pricing` с помощью создания и отправки CPU-задачи.

Рекомендуется использовать `XGBoost` на GPU, либо использовать `Spark` при расчетах больших датасетов.

После подключения к Jupyter Server необходимо загрузить файлы через веб-интерфейс Jupyter Server внутри ML Space:

 * [cal_housing_py3.pkz](cal_housing_py3.pkz) — датасет `California House Pricing`.
 * [quick-start.ipynb](quick-start.ipynb) — Jupyter-ноутбук, который запустит задачу обучения.
 * [requirements.txt](requirements.txt) — файл с зависимостями, который используется для сборки кастомного образа.
 * [xgboost-boston-house-price.py](xgboost-boston-house-price.py) — код модели, который запускается из [quick-start.ipynb](quick-start.ipynb).

Для запуска загрузите ноутбук [quick-start.ipynb](quick-start.ipynb) в веб-интерфейс [Jupyter Server, созданного в ML Space](https://mlspace.aicloud.sbercloud.ru/mlspace/jupyter-server).

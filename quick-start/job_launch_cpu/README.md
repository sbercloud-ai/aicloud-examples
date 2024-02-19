# Пример Обучения модели на CPU

В этом примере происходит обучение с помощью библиотек `XGBoost` на небольшом датасете `California House Pricing` с помощью создания и отправки CPU-задачи. Для датасетов большего размера рекомендуется использовать `XGBoost` на GPU либо `Spark`.

Пример включает в себя:

 * [cal_housing_py3.pkz](cal_housing_py3.pkz) — датасет `California House Pricing`.
 * [quick-start.ipynb](quick-start.ipynb) — Jupyter-ноутбук, который запустит задачу обучения.
 * [requirements.txt](requirements.txt) — файл с зависимостями, который используется для сборки кастомного образа.
 * [xgboost-boston-house-price.py](xgboost-boston-house-price.py) — код модели, который запускается из [quick-start.ipynb](quick-start.ipynb).

Для запуска загрузите ноутбук [quick-start.ipynb](quick-start.ipynb) в веб-интерфейс [Jupyter Server, созданного в ML Space](https://mlspace.aicloud.sbercloud.ru/mlspace/jupyter-server).

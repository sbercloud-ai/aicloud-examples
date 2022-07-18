# Знакомство с сервисом ML Space от Cloud

Пример из данного раздела позволит пользователям научиться отправке задач для обучения моделей на CPU.

Для запуска примера достаточно открыть ноутбук [quick-start.ipynb](quick-start.ipynb) в веб-интерфейсе [Jupyter Server внутри ML Space](https://aicloud.sbercloud.ru/_/jupyter/).

В этом примере происходит обучение с помощью библиотек `XGBoost` на маленьком датасете `California House Pricing` посредством создания и отправки CPU-задачи.

Для датасетов большего объема в целях ускорения расчетов рекомендуется использовать `XGBoost` на GPU, либо использовать `Spark`

Пример включает в себя:

 * [cal_housing_py3.pkz](cal_housing_py3.pkz) (датасет `California House Pricing`)
 * [quick-start.ipynb](quick-start.ipynb) (Jupyter Notebook который запустит задачу обучения)
 * [requirements.txt](requirements.txt) (файл с зависимостями, который используется для сборки кастомного образа)
 * [xgboost-boston-house-price.py](xgboost-boston-house-price.py) (код модели, который запускается из [quick-start.ipynb](quick-start.ipynb))


# Знакомство с библиотекой Rapids

В данном примере показано, как можно ускорить препроцессинг данных на GPU, используя библиотеки [Rapids](https://rapids.ai/) и [Dask](https://dask.org).

Для запуска примера [создайте](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__create-new-jupyter-server.html) или [подключитесь к уже существующему Jupyter Server](https://cloud.ru/ru/docs/aicloud/mlspace/concepts/guides/guides__jupyter/environments__environments__jupyter-server__connect-to-exist.html).

После подключения к Jupyter Server необходимо загрузить файлы через веб-интерфейс Jupyter Server внутри ML Space:

 * rapids_preprocessing.ipynb (отправка задач на суперкомпьютер Christofari)
 * cupy_cudf_example.py (сравнение скорости выполнения операций `groupby`,`merge`,`apply` на GPU и CPU)
 * dask_example.py (использование до 16 GPU внутри одного DGX-2)
 
 * Запускайте [пример для препроцессинга данных с использованием библиотеки Rapids](RapidsJupyterNotebooks) в образе jupyter-rapids.

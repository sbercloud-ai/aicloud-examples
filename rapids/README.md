# Знакомство с библиотекой Rapids

В данном примере показано, как можно ускорить препроцессинг данных на GPU, используя библиотеки [Rapids](https://rapids.ai/) и [Dask](https://dask.org).

Для запуска примера загрузите в веб-интерфейс [Jupyter-сервера внутри AI Cloud](https://aicloud.sbercloud.ru/_/jupyter/) следующие файлы:

 * rapids_preprocessing.ipynb (отправка задач на кластер "Кристофари")
 * cupy_cudf_example.py (сравнение скорости выполнения операций `groupby`,`merge`,`apply` на GPU и CPU)
 * dask_example.py (использование до 16 GPU внутри одного DGX-2)
 
 * Запускайте [пример для препроцессинга данных с использованием библиотеки Rapids](RapidsJupyterNotebooks) в образе jupyter-rapids.

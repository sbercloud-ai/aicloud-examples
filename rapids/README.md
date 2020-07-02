# Знакомство с библиотекой Rapids

В данном примере показано как можно ускорить препроцессинг данных на GPU, используя библиотеку [Rapids](https://rapids.ai/) и [Dask](https://dask.org).

Для запуска примера закачайте файлы из списка ниже в веб-интерфейс [Jupyter Server'а внутри AICloud](https://aicloud.sbercloud.ru/_/jupyter/).

 * rapids_preprocessing.ipynb (файл, отправляющий задачи на кластер Christofari)
 * cupy_cudf_example.py (сравнение скорости выполнения операций `groupby`,`merge`,`apply` на GPU по сравннению с CPU)
 * dask_example.py (использование внутри одного DGX-2 до 16 GPU)
 
 * [Rapids в ноутбуке с GPU](RapidsJupyterNotebooks) в AI Cloud из Jupyter Notebook, подключенного к GPU доступен в образе jupyter-rapids

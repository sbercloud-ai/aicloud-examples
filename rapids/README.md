# Знакомство с библиотекой Rapids

В данном примере показано как можно ускорить препроцессинг данных на GPU, используя библиотеку [Rapids](https://rapids.ai/) и [Dask](https://dask.org).

Для запуска примера закачайте файлы из списка ниже в веб-интерфейс [Jupyter Server'а внутри AICloud](https://aicloud.sbercloud.ru/_/jupyter/).

 * rapids_preprocessing.ipynb (файл, отправляющий задачи на кластер Christofari)
 * cupy_cudf_example.py (сравнение скорости выполнения операций `groupby`,`merge`,`apply` на GPU по сравннению с CPU)
 * dask_example.py (использование внутри одного DGX-2 до 16 GPU)
 * dask_mpi_example.py (запуск на нескольких DGX-2, больше 16 GPU)

# Пример тюнинга гиперпараметров (raytune) с Horovod и Pytorch.distributed

 В данном примере показано, как выполнить тюнинг гиперпараметров с использованием одной из двух библиотек:
 * [Horovod](https://github.com/horovod/horovod)
 * [Pytorch.distributed](https://pytorch.org/tutorials/intermediate/dist_tuto.html) (*в разработке*)


 Для типа сервера "Instance":

 Для запуска примера загрузите в веб-интерфейс [Jupyter Server внутри AI Cloud](https://aicloud.sbercloud.ru/_/jupyter/) следующие файлы:

 * [train_ray_horovod_example.py](train_ray_horovod_example.py) (распределенное обучение с использованием бибилотеки `Horovod`)
 * [requirements.txt](requirements.txt) (файл с зависимостями)
 * [ray_example.ipynb](ray_example.ipynb) (ноутбук для загрузки датасета и запуска обучения)
 * [config.yaml](config.yaml) (конфигурационный файл тюнинга гиперпараметров)
 
 Для запуска скриптов из-под Jupyter следуйте инструкциям в ray_example.ipynb
 
 Запуск с Horovod (в терминале):
 ```
pip install -U --user --no-cache -r requirements.txt

python train_ray_horovod_example.py
 ```
 Запуск с Pytorch.distributed:
 
 *В разработке...*
 
 Для типа сервера "Free":

 *В разработке...*
# Знакомство с библиотеками Horovod и Pytorch.distributed

В данном примере показано как можно писать скрипты для распределенного обучения `Pytorch` модели, используя одну из двух библиотек на выбор:
 * [Horovod](https://github.com/horovod/horovod)
 * [Pytorch.distributed](https://pytorch.org/tutorials/intermediate/dist_tuto.html)

Обратите внимание на параметр запуска `type` в `client_lib.Job`, принимающий одно из двух значений:
 * `type="horovod"` для запуска через библиотку Horovod
 * `type="pytorch"` для запуска обучения с использованием `DistributedDataParallel` и `Pytorch`

Для запуска примера закачайте файлы из списка ниже в веб-интерфейс [Jupyter Server'а внутри AICloud](https://aicloud.sbercloud.ru/_/jupyter/).

 * [pytorch_example.ipynb](pytorch_example.ipynb) (файл, отправляющий задачи на кластер Cristofari)
 * [train_distributed_example.py](train_distributed_example.py) (распределенное обучение с использованием `DistributedDataParallel` из бибилотеки `Pytorch.distributed`)
 * [train_horovod_example.py](train_horovod_example.py) (распределенное обучение с использованием бибилотеки `Horovod`)
 
 
 Для запуска и отладки скриптов из под Jupyter Notebook:
 
 Выберите один из образов с пометкой *horovod* (прим. jupyter-horovod-tf15)
 
 Запуск с Horovod:
 ```
 mpirun -np {GPU count} python train_horovod_example.py
 ```
 Запуск с Pytorch.distributed:
 ```
 python -m torch.distributed.launch --nproc_per_node {GPU count} train_distributed_example.py
```

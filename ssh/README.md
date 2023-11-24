# Примеры работы с сервисом ML Space от Cloud.ru по ssh

Ниже привиден пример запуска задачи обучения при подключении через ssh.

В этом примере мы создадим скрипт запуска на Python, с помощью которого можно запустить задачу обучения на кластере и посмотреть логи ее выполения.

## Environments (обучение моделей)

Первое что необходимо сделать - подключиться к запущенному инстансу Jupyter Server (free или gpu) по ssh. Процесс подключения подробно описан в документации сервиса и в интерфейсе ML Space.

После подключения по ssh запустите процесс обучения моделей одним из указанных способов:

#### 1. Напрямую из Jupyter Server, инстанс с GPU.
   
   Запустим пример, который доступен по ссылке: [Обучение модели в ноутбуке с GPU](quick-start/notebooks_gpu), пример представляет из себя Jupyter Notebook, который находится в домашней директории пользователя (`~/quick-start/notebooks_gpu/pytorch_tensorboard_mlflow.ipynb`)

   Первое, что нужно сделать — сконвертировать файл с моделью из формата `.ipynb` в `.py`-скрипт: ```jupyter nbconvert --to python ~/quick-start/notebooks_gpu/pytorch_tensorboard_mlflow.ipynb```

   После этого его можно будет запустить командой: `ipython ~/quick-start/notebooks_gpu/pytorch_tensorboard_mlflow.ipynb`

#### 2. Посредством отправки задачи обучения на кластер из free Jupyter Server.

   Первое, что нужно сделать для запуска обучения из консоли - создать управляющий скрипт, который задействует библитеку client_lib (она предустановлена внутри кластера). Пример такого скрипта доступен по ссылке [start_job.py](start_job.py), но вы можете использовать свой.
   
   1. Загрузим этот скрипт, выполнив в консоли: `cd ~ && wget https://raw.githubusercontent.com/sbercloud-ai/aicloud-examples/master/ssh/start_job.py`
   2. С помощью такого скрипта можно стартовать задачу обучения, указав файл с моделью, необходимый базовый образ и ресурсы:
      
      ```bash
      python3 start_job.py jobs run --n-gpus=2 /home/jovyan/quick-start/job_launch/tensorflow_mnist_estimator.py registry.aicloud.sbcp.ru/base/horovod-cuda10.0-tf1.15.0
      ```
      
      Результатом выполнения будет идентификатор запущенной задачи, например:
      
      ``Job "lm-mpi-job-b604a982-6158-4f3d-b11c-efa1229ddb34" created``
   3. Далее следить за процессом обучения можно как из интерфейса ML Space, так и выполнив в консоли следующую команду: `python3 start_job.py jobs logs lm-mpi-job-b604a982-6158-4f3d-b11c-efa1229ddb34`
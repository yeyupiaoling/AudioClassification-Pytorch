# work in progress

old instructions do not apply anymore

metrics are modified for binary classification. multiclass classification needs further modifications

### need to install:

- mlflow
  - `conda install conda-forge::mlflow`

### need to start before training:

- edit .`./configs/.yml` files with current experiment parameters
- `rm -rf ../log && mkdir ../log && rm -rf ../models && mkdir ../models`
- `mlflow server --host CHANGEME --port 5008`
  - dashboard: http://CHANGEME:5008
  - make sure it's running:
    - `netstat -ln | grep 5008`
    - `ps -aux | grep mlflow`
  - kill it if necessary:
    - `ps -ef | grep 'mlflow.server' | grep -v grep | awk '{print $2}' | xargs -r kill -15`

### training

- `OMP_NUM_THREADS=4 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py`

### multi-node training

- master node 192.168.x.x:
  - `torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=192.168.x.x --master_port=1234 train.py`
- worker nodes:
  - `torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=192.168.x.x --master_port=1234 train.py`

### changes made to the code:

- in train.py added reading local rank from environment variable

  - moved configs\log\models folders outside the folder code

- in trainer.py

  - multi-node training
  - add tar option
  - added MLflow logging

- in checkpoint.py

  - checkpoints of all epochs are now saved (not just the last 3)

- in predict.py

  - edited predict and predict_batch functions to ALSO return the `result` variable, needed for my custom inference code

- in \*.yml

  - moved dataset folder outside the folder code
  - added MLflow parameters
  - added new parameters for changes in predict.py

- in reader.py

  drop short audios during validation also

- in .gitignore

  - added the default exclusions

### to do:

- also add jupyter notebook with
  - file list creation code
  - evaluation code
  - fix metrics for multilabel classification

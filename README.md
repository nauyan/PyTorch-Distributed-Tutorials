# PyTorch-Distributed-Tutorials

Detailed blog on various Distributed Training startegies can be read [here](https://medium.com/p/54ae933bb9f0).

To train standalone PyTorch script run:
```console
python train.py
```
To train DataParallel PyTorch script run:
```console
python train_dataparallel.py
```
To train DistributedDataParallel(DDP) PyTorch script run:
```console
torchrun --nnodes=1 --nproc-per-node=4 train_ddp.py
```
To train FullyShardedDataParallel(FSDP) PyTorch script run:
```console
torchrun --nnodes=1 --nproc-per-node=4 train_fsdp.py
```

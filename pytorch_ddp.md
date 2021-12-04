## DDP 多卡分布式训练

导入库
```
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
```

local_rank 设置
```
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0)
FLAGS = parser.parse_args()
local_rank = int(FLAGS.local_rank)
```
同步bn
```
net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
```
初始化
```
def init_ddp():
    torch.cuda.set_device(f'cuda:{local_rank}')
    dist.init_process_group(backend='nccl')
    dist.get_rank()
    dist.get_world_size()
```
model 设置 需要先上cuda，混合精度amp init之后
```
net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
```

dataset 设置
```
sampler = DistributedSampler(dataset) if is_distributed else None
loader = DataLoader(dataset, shuffle=(sampler is None),
                    sampler=sampler)
for epoch in range(start_epoch, n_epochs):
    if is_distributed:
        sampler.set_epoch(epoch)
    train(loader)
```
执行脚本
```
CUDA_VISIBLE_DEVICES=xxx python -m torch.distributed.launch --nproc_per_node [n] train.py # n为几张卡
```

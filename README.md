# PyTorch Note 
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

## apex 混合精度
导入库
```
import apex
from apex import amp
```
FusedAdam速度更快
```
optimizer = apex.optimizers.FusedAdam([
        {'params':img_encoder.parameters()},
        {'params':text_encoder.parameters()}
    ], lr=1e-5, betas=(0.9, 0.99))
```
混合精度 init
```
(img_encoder, text_encoder), optimizer = amp.initialize(
        [img_encoder, text_encoder], optimizer,
        opt_level='O1' #, loss_scale=128
    )
```
backward
```
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```
以下摘自[apex.readme](https://github.com/NVIDIA/apex)
```
# Initialization
opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

# Train your model
...
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
...

# Save checkpoint
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'amp': amp.state_dict()
}
torch.save(checkpoint, 'amp_checkpoint.pt')
...

# Restore
model = ...
optimizer = ...
checkpoint = torch.load('amp_checkpoint.pt')

model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
amp.load_state_dict(checkpoint['amp'])

# Continue training
...

```

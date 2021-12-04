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

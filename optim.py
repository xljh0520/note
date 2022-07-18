base_params = list(map(id, net.backbone.parameters()))
logits_params = filter(lambda p: id(p) not in base_params, net.parameters())
params = [
    {"params": logits_params, "lr": config.lr},
    {"params": net.backbone.parameters(), "lr": config.backbone_lr},
]
optimizer = torch.optim.SGD(params, momentum=config.momentum, weight_decay=config.weight_decay)

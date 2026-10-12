"""Optional (Level 5): fine-tune a pretrained CNN end to end - the way the competition teams did it.

Everything else in this course trains a classifier *on top of* fixed CNN features. Here the CNN's
own weights are adjusted as well. Use a GPU (Colab: Runtime -> Change runtime type -> T4 GPU).
Estimated time for 3 epochs of ResNet-18 on ~17,000 crops: minutes on a T4, hours on a laptop CPU
(only a small CPU smoke test was run when this course was built). Works with torchvision ResNets.
"""
import time

import numpy as np

from .features import _Crops, device, imagenet_transform


def finetune(train, test, backbone="resnet18", epochs=3, batch_size=64, lr=1e-3, size=224,
             workers=2, freeze_backbone=False, log=print):
    """Train on train['path'] / train['y'], return predicted labels for test (same order)."""
    import torch
    import torchvision
    import torchvision.transforms as T

    classes = sorted(train["y"].unique())
    idx = {c: i for i, c in enumerate(classes)}
    dev = device()
    net = getattr(torchvision.models, backbone)(weights="IMAGENET1K_V1")
    net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
    if freeze_backbone:
        for n, p in net.named_parameters():
            p.requires_grad = n.startswith("fc.")
    net = net.to(dev)

    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = T.Compose([T.RandomResizedCrop(size, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                          T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), norm])
    ds = _Crops(list(train["path"]), train_tf)
    targets = torch.tensor([idx[c] for c in train["y"]])
    loader = torch.utils.data.DataLoader(list(range(len(ds))), batch_size=batch_size, shuffle=True)

    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader))
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(workers * 2)
    for ep in range(epochs):
        net.train()
        t0, tot, n = time.time(), 0.0, 0
        for b in loader:
            xb = torch.stack(list(ex.map(ds.__getitem__, b.tolist()))).to(dev)
            yb = targets[b].to(dev)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot, n = tot + loss.item() * len(b), n + len(b)
        log(f"epoch {ep + 1}/{epochs}: training loss {tot / n:.3f}  ({time.time() - t0:.0f}s)")

    net.eval()
    test_ds = _Crops(list(test["path"]), imagenet_transform(size))
    preds = []
    with torch.no_grad():
        for s in range(0, len(test_ds), batch_size):
            xb = torch.stack(list(ex.map(test_ds.__getitem__, range(s, min(s + batch_size, len(test_ds))))))
            preds.append(net(xb.to(dev)).argmax(1).cpu().numpy())
    ex.shutdown()
    return np.array(classes)[np.concatenate(preds)]

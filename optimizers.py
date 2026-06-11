import torch
from torch import nn


def build_optimizers(m:nn.Module, args):
    adamw_allowed_weights = ("embedding.weight", "output_fc.weight")

    muon_params = []
    adamw_params = []

    for name, param in m.named_parameters():
        if not param.requires_grad:
            continue

        name = name.removeprefix("module.")

        if name in adamw_allowed_weights or "norm" in name.lower():
            adamw_params.append(param)
        elif param.ndim == 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    optimizer_muon = torch.optim.Muon(
        muon_params,
        lr=args.peak_lr,
        adjust_lr_fn="match_rms_adamw",
        ns_steps=5,
        weight_decay=args.weight_decay,
        momentum=0.95,
        nesterov=True
    )

    optimizer_adamw = torch.optim.AdamW(
        adamw_params,
        lr=args.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    return optimizer_muon, optimizer_adamw
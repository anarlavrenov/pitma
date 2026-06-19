import torch
from torch import nn


def multi_hot_cross_entropy_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:

    # Пример:
    # pred size: [1, 4, 100].
    # true size: [1, 12].

    B, l, V = pred.shape                                                                   # [1, 4, 100]
    true_B, L = true.shape                                                                 # [1, 12]
    assert L % l == 0
    superposition_bag_size = L // l                                                        # 3
    superposition_offset = superposition_bag_size - 1                                      # 2

    pred = pred.flatten(0, 1).float()                                                      # [4, 100]
    true = nn.functional.pad(true, [0, superposition_offset], mode="constant", value=-100) # [1, 14]
    true = true[..., superposition_offset:].view(B, l, superposition_bag_size)             # [1, 4, 3]

    total_loss = 0.0
    total_w = 0.0

    for i in range(superposition_bag_size):
        w = 1
        target = true[..., i].flatten(0, 1)
        loss = nn.functional.cross_entropy(pred, target)

        total_loss += loss
        total_w += w

    return total_loss / total_w
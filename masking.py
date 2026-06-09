import torch


def make_reset_position_ids(doc_ids: torch.Tensor) -> torch.Tensor:
    """Поскольку используется кросс-документное маскирование,
    нужно сбросить rope_cache, чтобы каждый документ начинался с позиции 0."""

    # doc_ids = tensor([[0, 0, 0, 1, 1, 1, 1, 2, 2],
    #                   [0, 0, 0, 1, 1, 1, 1, 2, 2]])

    # boundary = tensor([[ True, False, False,  True, False, False, False,  True, False],
    #                    [ True, False, False,  True, False, False, False,  True, False]])

    # seq_pos = tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8],
    #                   [0, 1, 2, 3, 4, 5, 6, 7, 8]])

    # doc_starts = tensor([[0, 0, 0, 3, 3, 3, 3, 7, 7],
    #                      [0, 0, 0, 3, 3, 3, 3, 7, 7]])

    # reset_position_ids = tensor([[0, 1, 2, 0, 1, 2, 3, 0, 1],
    #                              [0, 1, 2, 0, 1, 2, 3, 0, 1]])

    # size: (2, 9)
    B, L = doc_ids.size()

    # size: (2, 9), состоящий из False.
    boundary = torch.zeros(B, L, dtype=torch.bool, device=doc_ids.device)
    boundary[:, 0] = True
    boundary[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]

    # [2, 9]
    seq_pos = torch.arange(L, device=doc_ids.device).unsqueeze(0).expand(B, L)

    doc_starts = torch.where(boundary, seq_pos, torch.zeros_like(seq_pos))
    doc_starts = torch.cummax(doc_starts, dim=1)[0]

    reset_position_ids = seq_pos - doc_starts

    return reset_position_ids
from torch.optim.lr_scheduler import LambdaLR

def get_wsd_scheduler(
        optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int
):

  total_stable_end = warmup_steps + stable_steps

  def lr_lambda(step):
    if step < warmup_steps:
      return step / warmup_steps

    elif step < total_stable_end:
      return 1.0

    else:
      progress = min((step - total_stable_end) / decay_steps, 1.0)
      return 0.5 ** (4 * progress)

  return LambdaLR(optimizer, lr_lambda=lr_lambda)


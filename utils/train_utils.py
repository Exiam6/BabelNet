def weighted_mse_loss(pred, target, weight):
    target = target.long()
    weight = weight[target - 1].to(pred.dtype)
    loss = (pred - target.to(pred.dtype)).pow(2)
    return ((weight * loss).mean(), loss.mean())

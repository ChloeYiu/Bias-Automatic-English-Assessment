import torch
from scipy.stats import spearmanr, kendalltau

def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        print("No CUDA found")
        return torch.device('cpu')

def calculate_avg(x):
    return torch.mean(x)

def calculate_mse(x1, x2):
    squared_error = (x2-x1)**2
    return torch.mean(squared_error)

def calculate_rmse(x1, x2):
    return calculate_mse(x1, x2)**0.5


def calculate_pcc(y_pred_useful, y):
    size = y.size(0)
    vy = y - torch.mean(y)
    vyp = y_pred_useful - torch.mean(y_pred_useful)
    pcc = 1/((size-1)*torch.std(vy) *torch.std(vyp))*(torch.sum(vy*vyp))
    return pcc


def calculate_less_0105(y_pred_useful, y):
    size = y.size(0)
    total05 = 0
    total1 = 0
    for a,b in zip(y, y_pred_useful):
            diff = abs(a-b)
            if diff < 1:
                    total1 += 1
                    if diff < 0.5:
                            total05 += 1
    less1 = total1/size
    less05 = total05/size
    return (less05, less1)


def calculate_less1(y_pred_useful, y):
        less05, less1 = calculate_less_0105(y_pred_useful, y)
        return less1


def calculate_less05(y_pred_useful, y):
        less05, less1 = calculate_less_0105(y_pred_useful, y)
        return less05


def calculate_src(y_pred_useful, y):
    src, _ = spearmanr(y_pred_useful.cpu().numpy(), y.cpu().numpy())
    return torch.tensor(src)


def calculate_krc(y_pred_useful, y):
    krc, _ = kendalltau(y_pred_useful.cpu().numpy(), y.cpu().numpy())
    return torch.tensor(krc)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

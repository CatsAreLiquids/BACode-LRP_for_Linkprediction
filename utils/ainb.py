import torch

def ainb(a,b):
    """gets mask for tensor a showing which elements are also found in b
    
    example:
        a = [0,1,2]
        b = [2,3,4]

        return [False, False, True]
    """

    size = (b.size(0), a.size(0))

    if size[0] == 0: # Prevents error in torch.Tensor.max(dim=0)
        return torch.tensor([False]*a.size(0), dtype= torch.bool)
        
    a = a.expand((size[0], size[1]))
    b = b.expand((size[1], size[0])).T

    mask = a.eq(b).max(dim= 0).values

    return mask


def ainb_wrapper(a, b, splits = .72):
    if a.size(0) == 0 or b.size(0) == 0:
        return torch.tensor([False]*a.size(0), dtype= torch.bool)
    
    inds = round(a.size(0)**splits)
    slices = round(a.size(0)/inds)+1
    tmp = [ainb(a[i*inds:(i+1)*inds], b) for i in range(slices)]
    try:
        ret = torch.cat(tmp)
    except NotImplementedError as err:
        print('WARNING: ainb_wrapper returned empty list')
        return torch.tensor([False]*a.size(0), dtype= torch.bool)
    return ret
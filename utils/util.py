import torch


def pad(k, p=None, d=1):
    """
    Calculate padding for Conv2d to maintain spatial dimensions.
    
    Args:
        k: kernel size
        p: padding (if None, auto-calculate)
        d: dilation
    
    Returns:
        padding value
    """
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    Generate anchor points from features.
    
    Args:
        feats: list of feature maps
        strides: stride values for each feature level
        grid_cell_offset: offset for grid cell centers (default 0.5)
    
    Returns:
        tuple of (anchor_points, stride_tensor)
    """
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    
    return torch.cat(anchor_points), torch.cat(stride_tensor)

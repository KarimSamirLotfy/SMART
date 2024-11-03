
from typing import Any, List, Optional
import torch

def safe_list_index(ls: List[Any], elem: Any) -> Optional[int]:
    try:
        return ls.index(elem)
    except ValueError:
        return None

def index_tensor(data: torch.Tensor, indices: torch.Tensor, dim: int = 2) -> torch.Tensor:
    """
    Index a tensor using indices along a specified dimension
    
    Args:
        data: Input tensor of any shape
        indices: Index tensor. Must be compatible with data's shape excluding the indexed dimension
        dim: Dimension to index along (default 2)
    
    Returns:
        Indexed tensor with shape of data but without the indexed dimension
    """
    # Create meshgrid for all dimensions before the indexed dim
    grid_shapes = [data.shape[i] for i in range(dim)]
    if grid_shapes:
        grids = torch.meshgrid([torch.arange(s) for s in grid_shapes], indexing='ij')
        # Expand each grid and indices to match input dimensions
        expanded_grids = [g.expand(*indices.shape, *([1] * (len(data.shape) - dim))) for g in grids]
        expanded_indices = indices.expand(*indices.shape, *([1] * (len(data.shape) - dim)))
        # Combine all indices
        final_indices = [*expanded_grids, expanded_indices]
    else:
        final_indices = [indices]
        
    return data[final_indices]

def select_with_index(data, index):
    """
    Selects specific slices from 'data' along the third dimension based on 'index'.

    Parameters:
        data (torch.Tensor): The input tensor of shape (16, 18, 2048, 4, 2).
        index (torch.Tensor): The indices tensor of shape (16, 18) for selection along dimension 2.

    Returns:
        torch.Tensor: The selected data based on the index tensor.
    """
    # Expanding index tensor dimensions to match 'data' dimensions for indexing
    index_expanded = index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 2)
    
    # Use torch.gather to select along the third dimension
    result = torch.gather(data, 2, index_expanded)
    return result


if __name__ == '__main__':
    # data = torch.randn(3, 4, 5)
    # indices = torch.tensor([[0, 1, 2], [3, 2, 1]])
    # print(index_tensor(data, indices, dim=1))

    # Create text case 2 
    data = torch.randn(16, 18, 2048, 4, 2)
    index = torch.randint(0, 2047, (16, 18))
    # print(select_with_index(data, index))

    # Create a7a
    results = torch.zeros(16, 18, 4, 2)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            results[i, j] = data[i, j, index[i, j]]
    print(results.shape)
    # Use advanced indexing to select the values from data
    # We need to expand the index dimensions to match the data shape
    # The unsqueeze adds dimensions to enable broadcasting
# Use advanced indexing to select the values from data
    # Create the batch and sequence dimensions for indexing
    batch_indices = torch.arange(data.shape[0]).unsqueeze(1).unsqueeze(2)  # Shape: [16, 1, 1]
    seq_indices = torch.arange(data.shape[1]).unsqueeze(0).unsqueeze(2)    # Shape: [1, 18, 1]

    # Expand the dimensions of index to make it compatible with data
    selected_results = data[batch_indices, seq_indices, index.unsqueeze(-1)]

    print(selected_results.shape)  # This should output: torch.Size([16, 18, 4, 2])

    print(results.shape)  # This should output: torch.Size([16, 18, 4, 2])


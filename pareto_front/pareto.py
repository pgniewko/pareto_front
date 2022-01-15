from typing import List
import torch


def gpu_pareto_front(samples: torch.Tensor, fronts_number=None) -> List[torch.Tensor]:
    """
    Non-dominated sorting algorithm for GPU

    Args:
        samples (torch.Tensor): m x n tensor
             m: the number of samples
             n: the number of objectives.
        fronts_number (int): number of the top fronts.
             None for all the fronts.
    Returns:
        fronts (List): a list of ordered Pareto fronts
    """
    dominate_each = (samples.unsqueeze(1) >= samples.unsqueeze(0)).all(-1)
    dominate_some = (samples.unsqueeze(1) > samples.unsqueeze(0)).any(-1)
    dominate_each = (dominate_each & dominate_some).to(torch.int16)

    fronts = []
    while (dominate_each.diagonal() == 0).any():
        count = dominate_each.sum(dim=0)
        front = torch.where(count == 0)[0]
        fronts.append(front)
        dominate_each[front, :] = 0
        dominate_each[front, front] = -1
        if fronts_number and len(fronts) >= fronts_number:
            break
    return fronts

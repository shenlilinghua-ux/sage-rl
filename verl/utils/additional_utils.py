import torch
import torch

def get_first_occurrence_pos(tensor: torch.Tensor, target_id: int) -> torch.Tensor:
    """
    找到每个样本中目标id首次出现的位置，未出现则返回序列长度
    
    Args:
        tensor: 输入张量，形状为 (batch_size, seq_len)
        target_id: 目标ID值
        
    Returns:
        形状为 (batch_size,) 的张量，每个元素为对应样本中目标ID的首次出现位置，
        未出现则为序列长度（tensor.size(1)）
    """
    # 获取序列长度（每个样本的序列长度）
    seq_len = tensor.size(1)
    
    # 生成掩码：标记目标ID出现的位置 (batch_size, seq_len)，1表示出现，0表示未出现
    target_mask = (tensor == target_id).float()
    
    # 计算每个样本中目标ID首次出现的位置（argmax返回第一个1的索引，若全0则返回0）
    first_pos = target_mask.argmax(dim=1)  # 形状 (batch_size,)
    
    # 检查哪些样本中从未出现目标ID（掩码全为0）
    has_target = target_mask.any(dim=1)  # 形状 (batch_size,)，True表示出现过
    
    # 未出现目标ID的样本，位置设为序列长度seq_len
    first_pos = torch.where(has_target, first_pos, torch.tensor(seq_len, device=tensor.device))
    
    return first_pos.float().mean().item()


# def get_avg_first_occurrence_pos(tensor: torch.Tensor, target_id: int) -> float:
#     # 找到每个样本中目标id首次出现的位置（返回形状为(batch_size,)的张量）
#     # 若未出现则为整个序列长度
#     first_occurrences = torch.where(
#         (tensor == target_id).any(dim=1),  # 标记哪些样本包含目标id
#         (tensor == target_id).float().argmax(dim=1),  # 含目标id的样本取首次出现位置
#         torch.zeros(tensor.size(0), dtype=torch.long, device=tensor.device)  # 不含的样本置0
#     )
#     return first_occurrences.float().mean().item()  # 求平均并转为Python数值


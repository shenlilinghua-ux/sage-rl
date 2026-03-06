import torch

from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List

from verl.workers.rollout.vllm_rollout.llm import LLM
from verl.workers.rollout.vllm_rollout.pref_instance_cls import SAGEParams, OutputObject, SAGESequence, SAGEOutput


def generate_uniform_indicator_tensor(tensor):
    row_sum = tensor.sum(dim=1)
    S = tensor.shape[1]
    uniform_rows = (row_sum == 0) | (row_sum == S)
    row_indicator = (~uniform_rows)
    indicator_tensor = row_indicator.unsqueeze(1).expand_as(tensor).flatten()
    return indicator_tensor


def transform_AAA_to_A(tensor, n):
    batch_size = tensor.shape[0] // n
    indices = torch.arange(0, tensor.shape[0], step=n, device=tensor.device)
    compressed_tensor = tensor[indices, :]
    return compressed_tensor


def fetch_single_prompt_list(lst, n):
    batch_size = len(lst) // n
    compressed_lst = [lst[i * n] for i in range(batch_size)]
    return compressed_lst


def fetch_group_first_x(lst, n, x):
    batch_size = len(lst) // n
    compressed_lst = [lst[i * n + j] for i in range(batch_size) for j in range(x)]
    return compressed_lst


def fetch_group_first_elements(lst, n, repeat_times):
    batch_size = len(lst) // n
    compressed_lst=[]
    for i in range(batch_size):
        for j in range(repeat_times[i]):
            compressed_lst.append(lst[i * n + j])
    return compressed_lst


def compute_matrix_average(sim_matrix, include_diagonal=False):
    sim_matrix=torch.tensor(sim_matrix)
    if include_diagonal:
        return sim_matrix.mean().item()
    else:
        n = sim_matrix.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
        return sim_matrix[mask].mean().item()


def compute_semantic_similarity(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=True)
    similarity_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
    return similarity_matrix


def get_adaptive_merge_ratio(accuracy_list, acc_threshold=0.75):
    acc_array = np.array(accuracy_list)
    n = len(acc_array)
    if n == 0:
        return []
    sorted_indices = np.argsort(-acc_array)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(n)
    percentiles = (ranks + 1) / n
    for i in range(len(accuracy_list)):
        if accuracy_list[i] < acc_threshold:
            percentiles[i] = 1
    return percentiles.tolist()


def create_attention_mask(lengths, max_seq_len=10):
    B = len(lengths)
    if max_seq_len is None:
        max_seq_len = max(lengths)
    range_tensor = torch.arange(max_seq_len).unsqueeze(0).repeat(B, 1)
    lengths_tensor = torch.tensor(lengths).unsqueeze(1)
    attention_mask = (range_tensor < lengths_tensor)
    return attention_mask


def convert_generate_to_sage(vllm_outputs, k, max_len):
    sage_outputs= []
    current_beam = []
    for output in vllm_outputs:
        seq = output.outputs[0]
        seq.tokens = output.prompt_token_ids + seq.token_ids[:max_len]
        seq.logprobs = output.prompt_logprobs + seq.logprobs[:max_len]
        seq.prompt_length = len(output.prompt_token_ids)+1
        current_beam.append(seq)
        if len(current_beam) == k:
            sage_output = SAGEOutput(sequences=current_beam)
            sage_outputs.append(sage_output)
            current_beam = []
    return sage_outputs

convert_generate_to_sage = convert_generate_to_sage
from vllm.sequence import Logprob

from typing import Optional, Any, Union, TYPE_CHECKING
from dataclasses import dataclass
import msgspec

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalDataDict


from verl.workers.rollout.vllm_rollout.efficient_token_vocab import detect_tokens

# SAGE parameters for text generation
class SAGEParams(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):  # type: ignore[call-arg]
    """SAGE parameters for text generation."""
    exploration_width: int
    max_tokens: int
    ignore_eos: bool = False
    temperature: float = 0.0
    length_penalty: float = 1.0
    include_stop_str_in_output: bool = False
    max_think_tokens: int =1024
    token_penalty: bool = False
    diverse_penalty: bool = False
    diverse_length: int = 100
    insert_sot: bool = False
    enable_sort: bool = True
    min_think_token: int = -1
    non_sort_way: str = "greedy"
    required_completions: int = -1  
    our_desire_first: bool = True
    sort_freq: int = -1 # control the basic step-wise freq of conducting sort e.g. every 10 generate step and sort once
    norm_beam_scores: bool = False
    logprob_weight: float = 0.8
    token_weight: float = 0.1
    diverse_weight: float = 0.1
    log_mode: bool = False
    accept_ratio: int = 1.0
    max_step_num: int = 30
    think_search_width: int = 2
    per_step_token: int = 512
    token_constrain: bool = False


@dataclass
class SAGESequence:
    """A sequence for SAGE.
    It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user.
    """
    # The tokens includes the prompt.
    tokens: list[int]
    logprobs: list[dict[int, Logprob]]
    cum_logprob: float = 0.0
    text: Optional[str] = None
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None
    multi_modal_data: Optional["MultiModalDataDict"] = None
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    think_length: int = None
    res_length: int = None
    diverse_score: float = 0.0
    logprob_score: float = 0.0
    token_score: float = 0.0
    prompt_length: int = None
    step_num: int = 0




@dataclass
class SAGEOutput:
    sequences: list[SAGESequence]



@dataclass
class OutputObject:
    outputs: list[Any]  


class SAGEInstance:

    def __init__(
        self,
        prompt_tokens: list[int],
        logprobs: Optional[list[dict[int, Logprob]]] = None,
        required_completions: int= 4 , 
        **kwargs,
    ):
        self.beams: list[SAGESequence] = [
            SAGESequence(
                tokens=prompt_tokens,
                logprobs=[] if logprobs is None else list(logprobs),
                prompt_length=len(prompt_tokens),
                **kwargs,
            )
        ]
        self.completed: list[SAGESequence] = []
        self.required_completions = required_completions  
        self.is_finished: bool = False  




def get_sage_score_with_token_penalty(
    tokens: list[int],
    tokenizer: Any,
    cumulative_logprob: float,
    eos_token_id: int,
    length_penalty: float = 1.0,
    token_penalty_weight: float = 0.5
) -> float:
    seq_len = len(tokens)
    if tokens[-1] == eos_token_id:
        seq_len -= 1   

    token_penalty=detect_tokens(tokenizer, tokens[-3:])
    score=cumulative_logprob / (seq_len**length_penalty) * (1-token_penalty * token_penalty_weight)
    return score



def get_sage_score_with_token_diverse_penalty(
    tokens: list[int],
    tokenizer: Any,
    cumulative_logprob: float,
    eos_token_id: int,
    length_penalty: float = 1.0,
    token_penalty_weight: float = 0.5,
    diverse_score: float = 0.0, 
    token_penalty: bool = False
) -> float:
    seq_len = len(tokens)
    if tokens[-1] == eos_token_id:
        seq_len -= 1   

    score=cumulative_logprob / (seq_len**length_penalty) # factor 1 the cumulative logprob

    score += (1-diverse_score) # factor 2  the diverse score

    if token_penalty: # factor 3 the token penalty
        token_penalty_score=detect_tokens(tokenizer, tokens[-3:])
        score= score *  (1-token_penalty_score * token_penalty_weight) 
    return score



# lazy sage sort
def lazy_aggregate_sage_score(
    logprob_score: float = 0.0,
    token_score: float = 0.0,
    diverse_score: float = 0.0,
    logprob_weight: float = 0,
    token_weight: float = 0,
    diverse_weight: float = 1.0
) -> float:
    return logprob_score * logprob_weight + token_score * token_weight + diverse_score * diverse_weight


# SPDX-License-Identifier: Apache-2.0

import itertools
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Callable, ClassVar, Optional, Union, cast, overload

import cloudpickle
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing_extensions import TypeVar, deprecated

from vllm.sage import  get_sage_score
from vllm.config import CompilationConfig
from vllm.engine.arg_utils import (EngineArgs, HfOverrides, PoolerConfig,
                                   TaskOption)
from vllm.engine.llm_engine import LLMEngine

from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         ChatTemplateContentFormatOption,
                                         apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         parse_chat_messages,
                                         resolve_chat_template_content_format)
from vllm.entrypoints.score_utils import (_cosine_similarity,
                                          _validate_score_input_lens)
from vllm.inputs import PromptType, SingletonPrompt, TextPrompt, TokensPrompt
from vllm.inputs.parse import is_token_prompt, parse_and_batch_prompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding.guided_fields import (
    GuidedDecodingRequest, LLMGuidedOptions)
from vllm.outputs import (ClassificationRequestOutput, EmbeddingRequestOutput,
                          PoolingRequestOutput, RequestOutput,
                          ScoringRequestOutput)
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import (GuidedDecodingParams,
                                  RequestOutputKind, SamplingParams)
from vllm.transformers_utils.tokenizer import (AnyTokenizer, MistralTokenizer,
                                               get_cached_tokenizer)
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (Counter, Device, deprecate_args, deprecate_kwargs,
                        is_list_of)

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)

from verl.workers.rollout.vllm_rollout.llm_engine import LLMEngine

from verl.workers.rollout.vllm_rollout.pref_instance_cls import (SAGEInstance, 
                               SAGEOutput, 
                               SAGESequence,
                               SAGEParams, 
                               get_sage_score_with_token_diverse_penalty,
                               lazy_aggregate_sage_score
                                )
from verl.workers.rollout.vllm_rollout.efficient_token_vocab import detect_tokens


import numpy as np
import random


def update_mean(old_mean, new_value, n) :
    new_mean = old_mean + (new_value - old_mean) / (n + 1)
    return new_mean

def get_res_length(tokenizer, id_seq):
    query_response_length=len(id_seq)
    sot_token_id=tokenizer.convert_tokens_to_ids("<think>")
    if sot_token_id not in id_seq:
        return query_response_length, query_response_length
    sot_pos=id_seq.index(sot_token_id)

    eot_token_id=tokenizer.convert_tokens_to_ids("</think>")
    if eot_token_id not in id_seq:
        return query_response_length, query_response_length
    eot_pos=id_seq.index(eot_token_id)
    
    res_length= query_response_length-sot_pos
    think_length=eot_pos-sot_pos 
    return res_length, think_length

def get_vanilla_res_length(tokenizer, id_seq):
    query_response_length=len(id_seq)
    sot_pos=0
    eot_token_id=tokenizer.convert_tokens_to_ids("</think>")
    if eot_token_id not in id_seq:
        return query_response_length, query_response_length
    eot_pos=id_seq.index(eot_token_id)
    res_length= query_response_length-sot_pos
    think_length=eot_pos-sot_pos 
    return res_length, think_length


def get_diverse_score(seq1, seq2):
    max_length = max(len(seq1), len(seq2))
    min_length = min(len(seq1), len(seq2))
    length_diff = abs(len(seq1)-len(seq2))
    diverse_score=(torch.sum(torch.tensor(seq1[:min_length])==torch.tensor(seq2[:min_length])).item() + length_diff) / max_length # 计算和自己之前所有beam的
    return diverse_score

class LLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer. Expect valid prompt_token_ids and None for prompt
            from the input.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        allowed_local_media_path: Allowing API requests to read local images
            or videos from directories specified by the server file system.
            This is a security risk. Should only be enabled in trusted
            environments.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq", "gptq", and "fp8" (experimental).
            If None, we first check the `quantization_config` attribute in the
            model config file. If that is None, we assume the model weights are
            not quantized and use `dtype` to determine the data type of
            the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Noting that `best_of` is only supported in V0. Otherwise, too small
            values may cause out-of-memory (OOM) errors.
        cpu_offload_gb: The size (GiB) of CPU memory to use for offloading
            the model weights. This virtually increases the GPU memory space
            you can use to hold the model weights, at the cost of CPU-GPU data
            transfer for every forward pass.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode. Additionally for encoder-decoder models, if the
            sequence length of the encoder input is larger than this, we fall
            back to the eager mode.
        disable_custom_all_reduce: See :class:`~vllm.config.ParallelConfig`
        disable_async_output_proc: Disable async output processing.
            This may result in lower performance.
        hf_token: The token to use as HTTP bearer authorization for remote files
            . If `True`, will use the token generated when running
            `huggingface-cli login` (stored in `~/.huggingface`).
        hf_overrides: If a dictionary, contains arguments to be forwarded to the
            HuggingFace config. If a callable, it is called to update the
            HuggingFace config.
        compilation_config: Either an integer or a dictionary. If it is an
            integer, it is used as the level of compilation optimization. If it
            is a dictionary, it can specify the full compilation configuration.
        **kwargs: Arguments for :class:`~vllm.EngineArgs`. (See
            :ref:`engine-args`)

    Note:
        This class is intended to be used for offline inference. For online
        serving, use the :class:`~vllm.AsyncLLMEngine` class instead.
    """

    DEPRECATE_LEGACY: ClassVar[bool] = True
    """A flag to toggle whether to deprecate the legacy generate/encode API."""

    DEPRECATE_INIT_POSARGS: ClassVar[bool] = True
    """
    A flag to toggle whether to deprecate positional arguments in
    :meth:`LLM.__init__`.
    """

    @classmethod
    @contextmanager
    def deprecate_legacy_api(cls):
        cls.DEPRECATE_LEGACY = True

        yield

        cls.DEPRECATE_LEGACY = False

    @deprecate_args(
        start_index=2,  # Ignore self and model
        is_deprecated=lambda: LLM.DEPRECATE_INIT_POSARGS,
        additional_message=(
            "All positional arguments other than `model` will be "
            "replaced with keyword arguments in an upcoming version."),
    )
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: Optional[bool] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        disable_async_output_proc: bool = False,
        hf_token: Optional[Union[bool, str]] = None,
        hf_overrides: Optional[HfOverrides] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        # After positional args are removed, move this right below `model`
        task: TaskOption = "auto",
        override_pooler_config: Optional[PoolerConfig] = None,
        compilation_config: Optional[Union[int, dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        '''
        LLM constructor.

        Note: if enforce_eager is unset (enforce_eager is None)
        it defaults to False.
        '''

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if compilation_config is not None:
            if isinstance(compilation_config, (int, dict)):
                compilation_config_instance = CompilationConfig.from_cli(
                    str(compilation_config))
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = None

        engine_args = EngineArgs(
            model=model,
            task=task,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            disable_async_output_proc=disable_async_output_proc,
            hf_token=hf_token,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            override_pooler_config=override_pooler_config,
            compilation_config=compilation_config_instance,
            **kwargs,
        )


        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS)
        self.engine_class = type(self.llm_engine)

        self.request_counter = Counter()
        self.default_sampling_params: Union[dict[str, Any], None] = None


    def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        return self.llm_engine.get_tokenizer_group().get_lora_tokenizer(
            lora_request)

    def set_tokenizer(self, tokenizer: AnyTokenizer) -> None:
        tokenizer_group = self.llm_engine.get_tokenizer_group()

        # While CachedTokenizer is dynamic, have no choice but
        # compare class name. Misjudgment will arise from
        # user-defined tokenizer started with 'Cached'
        if tokenizer.__class__.__name__.startswith("Cached"):
            tokenizer_group.tokenizer = tokenizer
        else:
            tokenizer_group.tokenizer = get_cached_tokenizer(tokenizer)

    def get_default_sampling_params(self) -> SamplingParams:
        if self.default_sampling_params is None:
            self.default_sampling_params = (
                self.llm_engine.model_config.get_diff_sampling_param())
        if self.default_sampling_params:
            return SamplingParams.from_optional(**self.default_sampling_params)
        return SamplingParams()

    @overload
    def generate(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        /,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        *,
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None,
    ) -> list[RequestOutput]:
        ...

    @overload  # LEGACY: single (prompt + optional token ids)
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def generate(
        self,
        prompts: str,
        sampling_params: Optional[Union[SamplingParams,
                                        list[SamplingParams]]] = None,
        prompt_token_ids: Optional[list[int]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None,
    ) -> list[RequestOutput]:
        ...

    @overload  # LEGACY: multi (prompt + optional token ids)
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def generate(
        self,
        prompts: list[str],
        sampling_params: Optional[Union[SamplingParams,
                                        list[SamplingParams]]] = None,
        prompt_token_ids: Optional[list[list[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None,
    ) -> list[RequestOutput]:
        ...

    @overload  # LEGACY: single (token ids + optional prompt)
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def generate(
        self,
        prompts: Optional[str] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        list[SamplingParams]]] = None,
        *,
        prompt_token_ids: list[int],
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None,
    ) -> list[RequestOutput]:
        ...

    @overload  # LEGACY: multi (token ids + optional prompt)
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def generate(
        self,
        prompts: Optional[list[str]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        list[SamplingParams]]] = None,
        *,
        prompt_token_ids: list[list[int]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None,
    ) -> list[RequestOutput]:
        ...

    @overload  # LEGACY: single or multi token ids [pos-only]
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def generate(
        self,
        prompts: None,
        sampling_params: None,
        prompt_token_ids: Union[list[int], list[list[int]]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None,
    ) -> list[RequestOutput]:
        ...

    @deprecate_kwargs(
        "prompt_token_ids",
        is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
        additional_message="Please use the 'prompts' parameter instead.",
    )
    def generate(
        self,
        prompts: Union[Union[PromptType, Sequence[PromptType]],
                       Optional[Union[str, list[str]]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        prompt_token_ids: Optional[Union[list[int], list[list[int]]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None,
        priority: Optional[list[int]] = None,
    ) -> list[RequestOutput]:
        """Generates the completions for the input prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each prompts.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
                When it is a single value, it is applied to every prompt.
                When it is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.
            priority: The priority of the requests, if any.
                Only applicable when priority scheduling policy is enabled.

        Returns:
            A list of ``RequestOutput`` objects containing the
            generated completions in the same order as the input prompts.

        Note:
            Using ``prompts`` and ``prompt_token_ids`` as keyword parameters is
            considered legacy and may be deprecated in the future. You should
            instead pass them via the ``inputs`` parameter.
        """
        runner_type = self.llm_engine.model_config.runner_type
        if runner_type not in ["generate", "transcription"]:
            messages = [
                "LLM.generate() is only supported for (conditional) generation "
                "models (XForCausalLM, XForConditionalGeneration).",
            ]

            supported_runner_types = self.llm_engine.model_config \
                .supported_runner_types
            if "generate" in supported_runner_types:
                messages.append(
                    "Your model supports the 'generate' runner, but is "
                    f"currently initialized for the '{runner_type}' runner. "
                    "Please initialize vLLM using `--task generate`.")

            raise ValueError(" ".join(messages))

        if prompt_token_ids is not None:
            parsed_prompts = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, list[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            parsed_prompts = cast(Union[PromptType, Sequence[PromptType]],
                                  prompts)

        if isinstance(guided_options_request, dict):
            if len(guided_options_request) > 1:
                raise ValueError(
                    "You can only use one guided decoding but multiple is "
                    f"specified: {guided_options_request}")
            guided_options_request = GuidedDecodingRequest(
                **guided_options_request)

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = self.get_default_sampling_params()
        self._validate_and_add_requests(
            prompts=parsed_prompts,
            params=sampling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            guided_options=guided_options_request,
            priority=priority)
        # 运行llm engine进行推理
        outputs = self._run_engine(use_tqdm=use_tqdm)
        return self.engine_class.validate_outputs(outputs, RequestOutput)

    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict[str, Any]] = None) -> list[_R]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                :exc:`TimeoutError` on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.

        Returns:
            A list containing the results from each worker.
        
        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """

        return self.llm_engine.collective_rpc(method, timeout, args, kwargs)

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        """
        Run a function directly on the model inside each worker,
        returning the result for each of them.
        """
        executor = self.llm_engine.model_executor
        return executor.apply_model(func)



    def TSearch_with_Phi(
        self,
        prompts: list[Union[TokensPrompt, TextPrompt]],
        params: SAGEParams,
    ) -> list[SAGEOutput]:
        """
        Generate sequences using SAGE.

        Args:
            prompts: A list of prompts. Each prompt can be a string or a list
                of token IDs.
            params: The SAGE parameters.
        """
        # TODO: how does SAGE work together with length penalty,
        # frequency, penalty, and stopping criteria, etc.?
        exploration_width = params.exploration_width
        max_tokens = params.max_tokens
        temperature = params.temperature
        ignore_eos = params.ignore_eos
        length_penalty = params.length_penalty 
        max_think_tokens = min(params.max_think_tokens, max_tokens)
        token_penalty=params.token_penalty
        diverse_penalty = params.diverse_penalty
        diverse_length = params.diverse_length
        insert_sot = params.insert_sot
        enable_sort = params.enable_sort
        min_think_token = params.min_think_token
        non_sort_way = params.non_sort_way
        required_completions = params.required_completions if params.required_completions > 0 else exploration_width
        our_desire_first = params.our_desire_first
        sort_freq = params.sort_freq
        norm_beam_scores = params.norm_beam_scores 
        accept_ratio = params.accept_ratio

        # TODO set the freq, each stage we conduct sort for 90 times
        length_buckets = [
            (0, sort_freq * 10, 1),
            (sort_freq * 10, sort_freq * 100, sort_freq),
            (sort_freq * 100, sort_freq * 1000, sort_freq * 10),
            (sort_freq * 1000, sort_freq * 10000, sort_freq * 100),
            (sort_freq * 10000, sort_freq * 100000, sort_freq * 1000)
        ] 


        def sort_beams_key(x: SAGESequence) -> float:
            if norm_beam_scores:
                return lazy_aggregate_sage_score(logprob_score=x.logprob_score, 
                                            token_score=x.token_score,
                                            diverse_score=x.diverse_score,
                                            logprob_weight=params.logprob_weight,
                                            token_weight=params.logprob_weight,
                                            diverse_weight=params.diverse_weight) 
            else:
                return get_sage_score_with_token_diverse_penalty(tokens=x.tokens, tokenizer=tokenizer, 
                                                            cumulative_logprob=x.cum_logprob,
                                                            eos_token_id=tokenizer.eos_token_id, 
                                                            length_penalty=length_penalty, 
                                                            diverse_score=x.diverse_score, 
                                                            token_penalty=token_penalty) 
        
            
        def candidate_sort_beams_key(x: SAGESequence) -> float:
            return get_sage_score_with_token_diverse_penalty(tokens=x.tokens, tokenizer=tokenizer, 
                                                        cumulative_logprob=x.cum_logprob,
                                                        eos_token_id=tokenizer.eos_token_id, 
                                                        length_penalty=length_penalty, 
                                                        diverse_score=x.diverse_score, 
                                                        token_penalty=token_penalty) 

        def create_tokens_prompt(
                beam: SAGESequence) -> TokensPrompt:
            token_prompt_kwargs: TokensPrompt = {
                "prompt_token_ids": beam.tokens
            }
            if beam.multi_modal_data is not None:
                token_prompt_kwargs["multi_modal_data"] = beam.multi_modal_data

            if beam.mm_processor_kwargs is not None:
                token_prompt_kwargs[
                    "mm_processor_kwargs"] = beam.mm_processor_kwargs
            return TokensPrompt(**token_prompt_kwargs)

        tokenizer = self.get_tokenizer() 
        # TODO get \think token
        eot_token_id=tokenizer.convert_tokens_to_ids("</think>")
        sot_token_id=tokenizer.convert_tokens_to_ids("<think>")
        sage_params = SamplingParams(logprobs=2 * exploration_width,
                                            max_tokens=1,
                                            temperature=temperature)
        instances: list[SAGEInstance] = []
        for prompt in prompts: 
            mm_kwargs = {}
            if "multi_modal_data" in prompt:
                mm_kwargs["multi_modal_data"] = prompt["multi_modal_data"]
            if "mm_processor_kwargs" in prompt:
                mm_kwargs["mm_processor_kwargs"] = prompt[
                    "mm_processor_kwargs"]

            if is_token_prompt(prompt):
                prompt_tokens = prompt["prompt_token_ids"] 
            else:
                prompt_tokens = tokenizer.encode(prompt["prompt"]) 
            
            if insert_sot: 
                prompt_tokens.append(sot_token_id)

            instances.append(
                SAGEInstance(prompt_tokens, logprobs=None, required_completions=required_completions, **mm_kwargs)) 
        

        for t in range(max_think_tokens): 
            active_instances = [inst for inst in instances if not inst.is_finished]
            if not active_instances:
                break 
            all_beams: list[SAGESequence] = list(
                sum((instance.beams for instance in active_instances), [])) 
            pos = [0] + list(
                itertools.accumulate(
                    len(instance.beams) for instance in active_instances))
            instance_start_and_end: list[tuple[int, int]] = list(
                zip(pos[:-1], pos[1:]))  

            if len(all_beams) == 0:
                break

            prompts_batch = [
                create_tokens_prompt(beam) for beam in all_beams] 


            output = self.generate(prompts_batch,
                                   sampling_params=sage_params,
                                   use_tqdm=False) 


            for (start, end), instance in zip(instance_start_and_end,
                                              active_instances):
                instance_new_beams = []
                for i in range(start, end): 
                    current_beam = all_beams[i]
                    result = output[i]

                    if result.outputs[0].logprobs is not None:
                        # if `result.outputs[0].logprobs` is None, it means
                        # the sequence is completed because of the max-model-len
                        # or abortion. we don't need to add it to the new beams.
                        logprobs = result.outputs[0].logprobs[0] 
                        for token_id, logprob_obj in logprobs.items():
                            new_beam = SAGESequence(
                                tokens=current_beam.tokens + [token_id],
                                logprobs=current_beam.logprobs + [logprobs],
                                cum_logprob=current_beam.cum_logprob +
                                logprob_obj.logprob,
                                multi_modal_data=current_beam.multi_modal_data,
                                mm_processor_kwargs=current_beam.mm_processor_kwargs,
                                prompt_length=current_beam.prompt_length)
                            
                            # TODO add a constraint 
                            if token_id == eot_token_id and not ignore_eos:
                                token_rank = logprob_obj.rank
                                total_tokens = len(logprobs.items())
                                if token_rank < total_tokens * accept_ratio:
                                    instance.completed.append(new_beam)
                                  
                            else: 
                                instance_new_beams.append(new_beam)

                # compute the current seq
                cur_sort_freq = 1 
                if sort_freq: # if set to -1, every step we conduct sort
                    for min_step, max_step, freq in reversed(length_buckets): 
                        if t >= min_step and t<= max_step:
                            cur_sort_freq = freq
                            break         
                
                if enable_sort and t % cur_sort_freq == 0:
                    if diverse_penalty:
                        def get_overlap_score(tokens1: list[int], tokens2: list[int]) -> float:
                            set1 = set(tokens1)
                            set2 = set(tokens2)
                            intersection = len(set1 & set2) 
                            union = len(set1 | set2)       
                            return intersection / union 

                        for bid in range(1,len(instance_new_beams)):
                            ds = get_overlap_score(instance_new_beams[bid].tokens, instance_new_beams[bid-1].tokens)
                            instance_new_beams[bid].diverse_score = ds
                            # print(f"diverse score: {ds}")
                    
                    if norm_beam_scores:
                        # pre_compute the scores and record the min max value
                        min_logprob = float('inf')
                        max_logprob = -float('inf')
                        min_diverse = float('inf')
                        max_diverse = -float('inf')
                        min_token_score = float('inf')
                        max_token_score = -float('inf')

                        for beam in instance_new_beams:
                            # TODO compute logprob_score
                            beam.logprob_score = beam.cum_logprob / len(beam.tokens)
                            min_logprob = min(min_logprob, beam.logprob_score)
                            max_logprob = max(max_logprob, beam.logprob_score)

                            # TODO compute the token score
                            beam.token_score = detect_tokens(tokenizer, beam.tokens[-3:])
                            min_token_score = min(min_token_score, beam.token_score)
                            max_token_score = max(max_token_score, beam.token_score)

                            # TODO  compute the diverse score
                            beam.diverse_score = 1 - beam.diverse_score
                            min_diverse = min(min_diverse, beam.diverse_score)
                            max_diverse = max(max_diverse, beam.diverse_score)
                        
                        # apply min_max norm
                        def min_max_norm(raw, _min, _max):
                            return (raw - _min ) / (_max - _min + 1e-12) 

                        
                        for idx, beam in enumerate(instance_new_beams):
                            beam.logprob_score = min_max_norm(beam.logprob_score, min_logprob, max_logprob)
                            beam.token_score = min_max_norm(beam.token_score, min_token_score, max_token_score)
                            beam.diverse_score = min_max_norm(beam.diverse_score, min_diverse, max_diverse)
                            # print(f"beam: {idx+1}, logprob_score: {beam.logprob_score}, token_score: {beam.token_score}, diverse_score: {beam.diverse_score}")


                    sorted_beams = sorted(instance_new_beams,
                                        key=sort_beams_key,
                                        reverse=True) 
                    instance.beams = sorted_beams[:exploration_width] 
                
                else:
                    if "greedy" in non_sort_way :
                        instance.beams = instance_new_beams[:exploration_width]
                    elif "random" in non_sort_way:
                        instance.beams = random.sample(instance_new_beams, exploration_width)
                    
                
                # TODO remove the valid_completed logits
                if len(instance.completed) >= instance.required_completions:
                    instance.is_finished = True 
                    instance.beams = []  

        outputs = []
        for instance in instances:

            # TODO we force the output generated by us rank in the first
            if  our_desire_first:
                sorted_completed = sorted(instance.completed,
                                        key=sort_beams_key,
                                        reverse=True)
                sorted_completed.extend(sorted(instance.beams, key=sort_beams_key, reverse=True))

            else:
                instance.completed.extend(instance.beams) 
                sorted_completed = sorted(instance.completed, key=sort_beams_key, reverse=True) 


            best_beams = sorted_completed[:required_completions] 
            for b in best_beams:
                if eot_token_id not in b.tokens:
                    b.tokens+=[eot_token_id]
            
            best_beams_ids=[b.tokens for b in best_beams]
            sampling_params = SamplingParams(n=1,
                                            temperature=0,
                                            max_tokens=max(1, max_tokens-max_think_tokens)) 
            
            answer_outputs = self.generate(prompt_token_ids=best_beams_ids, sampling_params=sampling_params,
                                   use_tqdm=False) 


            for beam, output in zip(best_beams,answer_outputs):
                beam.tokens=beam.tokens + output.outputs[0].token_ids # <think>
                beam.res_length, beam.think_length = get_res_length(tokenizer, beam.tokens)
                beam.text = tokenizer.decode(beam.tokens)
        
            outputs.append(SAGEOutput(sequences=best_beams))


        return outputs
    
    

    def SAGE(
        self,
        prompts: list[Union[TokensPrompt, TextPrompt]],
        params: SAGEParams,
    ) -> list[SAGEOutput]:
        exploration_width = params.exploration_width
        max_tokens = params.max_tokens
        temperature = params.temperature
        ignore_eos = params.ignore_eos
        length_penalty = params.length_penalty 
        max_think_tokens = min(params.max_think_tokens, max_tokens) 
        token_penalty=params.token_penalty
        diverse_penalty = params.diverse_penalty
        diverse_length = params.diverse_length
        insert_sot = params.insert_sot
        enable_sort = params.enable_sort
        min_think_token = params.min_think_token
        non_sort_way = params.non_sort_way
        required_completions = params.required_completions if params.required_completions > 0 else exploration_width
        our_desire_first = params.our_desire_first
        sort_freq = params.sort_freq
        norm_beam_scores = params.norm_beam_scores 
        accept_ratio = params.accept_ratio
        # new params for SAGE
        max_step_num = params.max_step_num
        think_search_width = params.think_search_width
        per_step_token = params.per_step_token

        # TODO get step
        model_name_or_path=self.llm_engine.tokenizer.tokenizer_id

        token_wise_constrain = params.token_constrain
        if token_wise_constrain:
            per_step_token = 1024
            max_step_num = 200


        def sort_beams_key(x: SAGESequence) -> float:
            if norm_beam_scores:
                return lazy_aggregate_sage_score(logprob_score=x.logprob_score, 
                                            token_score=x.token_score,
                                            diverse_score=x.diverse_score,
                                            logprob_weight=params.logprob_weight,
                                            token_weight=params.logprob_weight,
                                            diverse_weight=params.diverse_weight) 
            else:
                return get_sage_score_with_token_diverse_penalty(tokens=x.tokens, tokenizer=tokenizer, 
                                                            cumulative_logprob=x.cum_logprob,
                                                            eos_token_id=tokenizer.eos_token_id, 
                                                            length_penalty=length_penalty, 
                                                            diverse_score=x.diverse_score, 
                                                            token_penalty=token_penalty) 
        
        def create_tokens_prompt(
                beam: SAGESequence) -> TokensPrompt:
            token_prompt_kwargs: TokensPrompt = {
                "prompt_token_ids": beam.tokens
            }
            if beam.multi_modal_data is not None:
                token_prompt_kwargs["multi_modal_data"] = beam.multi_modal_data

            if beam.mm_processor_kwargs is not None:
                token_prompt_kwargs[
                    "mm_processor_kwargs"] = beam.mm_processor_kwargs
            return TokensPrompt(**token_prompt_kwargs)

        tokenizer = self.get_tokenizer() 
        eot_token_id=tokenizer.convert_tokens_to_ids("</think>")
        eostep_token_id = 4710 
        sot_token_id=tokenizer.convert_tokens_to_ids("<think>")

        eostep_token_list = [eostep_token_id, eot_token_id]

        instances: list[SAGEInstance] = []
        for prompt in prompts: 
            mm_kwargs = {}
            if "multi_modal_data" in prompt:
                mm_kwargs["multi_modal_data"] = prompt["multi_modal_data"]
            if "mm_processor_kwargs" in prompt:
                mm_kwargs["mm_processor_kwargs"] = prompt[
                    "mm_processor_kwargs"]

            if is_token_prompt(prompt):
                prompt_tokens = prompt["prompt_token_ids"] 
            else:
                prompt_tokens = tokenizer.encode(prompt["prompt"]) 
            
            if insert_sot:
                prompt_tokens.append(sot_token_id)

            instances.append(
                SAGEInstance(prompt_tokens, logprobs=None, required_completions=required_completions, **mm_kwargs)) 
        
        for t in range(max_step_num): 
           
            active_instances = [inst for inst in instances if not inst.is_finished]

            if not active_instances:
                break 

            all_beams: list[SAGESequence] = list(
                sum((instance.beams for instance in active_instances), [])) 
            pos = [0] + list(
                itertools.accumulate(
                    len(instance.beams) for instance in active_instances))
            instance_start_and_end: list[tuple[int, int]] = list(
                zip(pos[:-1], pos[1:]))  
            if len(all_beams) == 0:
                break

            prompts_batch = [
                create_tokens_prompt(beam) for beam in all_beams
            ] 

            step_sampling_params = SamplingParams(n=exploration_width*2, 
                                                    temperature=1.0, logprobs=think_search_width,
                                                    top_p=1, max_tokens=per_step_token, 
                                                    stop_token_ids=eostep_token_list)

            output = self.generate(prompts_batch,
                                   sampling_params=step_sampling_params,
                                   use_tqdm=False) 
            
            for (start, end), instance in zip(instance_start_and_end,
                                              active_instances):
                
                instance_new_beams = []
                for i in range(start, end): 
                    current_beam = all_beams[i]
                    result = output[i]  

                    if result is not None:
                        for output_step in result.outputs:

                            new_beam = SAGESequence(
                                tokens=current_beam.tokens + output_step.token_ids,
                                logprobs=current_beam.logprobs + output_step.logprobs,
                                cum_logprob=current_beam.cum_logprob + output_step.cumulative_logprob,
                                multi_modal_data=current_beam.multi_modal_data,
                                mm_processor_kwargs=current_beam.mm_processor_kwargs,
                                prompt_length=current_beam.prompt_length,
                                step_num = t+1)
                            
                            last_token = new_beam.tokens[-1]
                            last_token_logprobs = output_step.logprobs[-1]

                            window_size = len(last_token_logprobs)
                            
                            cur_step = len(current_beam.tokens) - current_beam.prompt_length

                            
                            find_eot = False
                            eot_logpro_obj = None
                            for token_id, logprob_obj in last_token_logprobs.items(): 
                                if eot_token_id == token_id:
                                    find_eot = True
                                    eot_logprob_obj = logprob_obj

                            
                            if find_eot and eot_logprob_obj.rank < think_search_width: 
                                instance.completed.append(new_beam)
                            
                            elif last_token == eot_token_id:
                                instance.completed.append(new_beam)

                            elif token_wise_constrain and len(new_beam.tokens) - new_beam.prompt_length > max_think_tokens - per_step_token:
                                instance.completed.append(new_beam)
                            else: 
                                instance_new_beams.append(new_beam)
     
                if enable_sort:
                    if diverse_penalty:
                        def get_overlap_score(tokens1: list[int], tokens2: list[int]) -> float:
                            set1 = set(tokens1)
                            set2 = set(tokens2)
                            intersection = len(set1 & set2) 
                            union = len(set1 | set2)       
                            return intersection / union 

                        for bid in range(1,len(instance_new_beams)):
                            ds = get_overlap_score(instance_new_beams[bid].tokens, instance_new_beams[bid-1].tokens)
                            instance_new_beams[bid].diverse_score = ds
                            # print(f"diverse score: {ds}")
                    
                    if norm_beam_scores:
                        # pre_compute the scores and record the min max value
                        min_logprob = float('inf')
                        max_logprob = -float('inf')
                        min_diverse = float('inf')
                        max_diverse = -float('inf')
                        min_token_score = float('inf')
                        max_token_score = -float('inf')

                        for beam in instance_new_beams:
                            # TODO compute logprob_score
                            beam.logprob_score = beam.cum_logprob / len(beam.tokens)
                            min_logprob = min(min_logprob, beam.logprob_score)
                            max_logprob = max(max_logprob, beam.logprob_score)

                            # TODO compute the token score
                            beam.token_score = detect_tokens(tokenizer, beam.tokens[-3:])
                            min_token_score = min(min_token_score, beam.token_score)
                            max_token_score = max(max_token_score, beam.token_score)

                            # TODO  compute the diverse score
                            beam.diverse_score = 1 - beam.diverse_score
                            min_diverse = min(min_diverse, beam.diverse_score)
                            max_diverse = max(max_diverse, beam.diverse_score)
                        
                        # apply min_max norm
                        def min_max_norm(raw, _min, _max):
                            return (raw - _min ) / (_max - _min + 1e-12) 

                        
                        for idx, beam in enumerate(instance_new_beams):
                            beam.logprob_score = min_max_norm(beam.logprob_score, min_logprob, max_logprob)
                            beam.token_score = min_max_norm(beam.token_score, min_token_score, max_token_score)
                            beam.diverse_score = min_max_norm(beam.diverse_score, min_diverse, max_diverse)
                            # print(f"beam: {idx+1}, logprob_score: {beam.logprob_score}, token_score: {beam.token_score}, diverse_score: {beam.diverse_score}")


                    sorted_beams = sorted(instance_new_beams,
                                        key=sort_beams_key,
                                        reverse=True) 
                    instance.beams = sorted_beams[:exploration_width]
                
                else:
                    if "greedy" in non_sort_way :
                        instance.beams = instance_new_beams[:exploration_width]
                    elif "random" in non_sort_way:
                        instance.beams = random.sample(instance_new_beams, exploration_width)
                    
                
                # TODO remove the valid_completed logits
                if len(instance.completed) >= instance.required_completions:
                    instance.is_finished = True  
                    instance.beams = []  

        outputs = []
        for instance in instances: 

            # TODO we force the output generated by us rank in the first
            if  our_desire_first:
                sorted_completed = sorted(instance.completed,
                                        key=sort_beams_key,
                                        reverse=True) 
                sorted_completed.extend(sorted(instance.beams, key=sort_beams_key, reverse=True))

            else:
                instance.completed.extend(instance.beams) 
                sorted_completed = sorted(instance.completed, key=sort_beams_key, reverse=True) 


            # TODO exploration_width->sorted_completed
            # best_beams = sorted_completed[:exploration_width]                       
            best_beams = sorted_completed[:required_completions] 

            for b in best_beams:
                if eot_token_id not in b.tokens:
                    b.tokens+=[eot_token_id]

            best_beams_ids=[b.tokens for b in best_beams]
            sampling_params = SamplingParams(n=1,
                                            temperature=0,
                                            max_tokens=max(1, max_tokens-max_think_tokens)) 

            answer_outputs = self.generate(prompt_token_ids=best_beams_ids, sampling_params=sampling_params,
                                   use_tqdm=False) 

            for beam, output in zip(best_beams,answer_outputs):
                beam.tokens=beam.tokens + output.outputs[0].token_ids # <think>
                beam.res_length, beam.think_length = get_res_length(tokenizer, beam.tokens)
                beam.text = tokenizer.decode(beam.tokens)
        
            outputs.append(SAGEOutput(sequences=best_beams))



        return outputs
    


    
    def chat(
        self,
        messages: Union[list[ChatCompletionMessageParam],
                        list[list[ChatCompletionMessageParam]]],
        sampling_params: Optional[Union[SamplingParams,
                                        list[SamplingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        chat_template: Optional[str] = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: Optional[list[dict[str, Any]]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[RequestOutput]:
        """
        Generate responses for a chat conversation.

        The chat conversation is converted into a text prompt using the
        tokenizer and calls the :meth:`generate` method to generate the
        responses.

        Multi-modal inputs can be passed in the same way you would pass them
        to the OpenAI API.

        Args:
            messages: A list of conversations or a single conversation.

              - Each conversation is represented as a list of messages.
              - Each message is a dictionary with 'role' and 'content' keys.

            sampling_params: The sampling parameters for text generation.
                If None, we use the default sampling parameters. When it
                is a single value, it is applied to every prompt. When it
                is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The template to use for structuring the chat.
              If not provided, the model's default chat template will be used.
            chat_template_content_format: The format to render message content.

              - "string" will render the content as a string.
                Example: ``"Who are you?"``
              - "openai" will render the content as a list of dictionaries,
                similar to OpenAI schema.
                Example: ``[{"type": "text", "text": "Who are you?"}]``

            add_generation_prompt: If True, adds a generation template
                to each message.
            continue_final_message: If True, continues the final message in
                the conversation instead of starting a new one. Cannot be
                ``True`` if ``add_generation_prompt`` is also ``True``.
            mm_processor_kwargs: Multimodal processor kwarg overrides for this
                chat request. Only used for offline requests.

        Returns:
            A list of ``RequestOutput`` objects containing the generated
            responses in the same order as the input messages.
        """
        list_of_messages: list[list[ChatCompletionMessageParam]]

        # Handle multi and single conversations
        if is_list_of(messages, list):
            # messages is list[list[...]]
            list_of_messages = cast(list[list[ChatCompletionMessageParam]],
                                    messages)
        else:
            # messages is list[...]
            list_of_messages = [
                cast(list[ChatCompletionMessageParam], messages)
            ]

        tokenizer = self.get_tokenizer(lora_request)
        model_config = self.llm_engine.get_model_config()
        resolved_content_format = resolve_chat_template_content_format(
            chat_template,
            tools,
            chat_template_content_format,
            tokenizer,
            trust_remote_code=model_config.trust_remote_code,
        )

        prompts: list[Union[TokensPrompt, TextPrompt]] = []

        for msgs in list_of_messages:
            # NOTE: _parse_chat_message_content_parts() currently doesn't
            # handle mm_processor_kwargs, since there is no implementation in
            # the chat message parsing for it.
            conversation, mm_data = parse_chat_messages(
                msgs,
                model_config,
                tokenizer,
                content_format=resolved_content_format,
            )

            if isinstance(tokenizer, MistralTokenizer):
                prompt_token_ids = apply_mistral_chat_template(
                    tokenizer,
                    messages=msgs,
                    chat_template=chat_template,
                    tools=tools,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                )
            else:
                prompt_str = apply_hf_chat_template(
                    tokenizer,
                    trust_remote_code=model_config.trust_remote_code,
                    conversation=conversation,
                    chat_template=chat_template,
                    tools=tools,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                )
                # Special tokens are already included in chat templates so
                # should not be added by the tokenizer in this case.
                prompt_token_ids = tokenizer.encode(prompt_str,
                                                    add_special_tokens=False)

            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

            if mm_data is not None:
                prompt["multi_modal_data"] = mm_data

            if mm_processor_kwargs is not None:
                prompt["mm_processor_kwargs"] = mm_processor_kwargs

            prompts.append(prompt)

        return self.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
        )

    @overload
    def encode(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        /,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        *,
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> list[PoolingRequestOutput]:
        ...

    @overload  # LEGACY: single (prompt + optional token ids)
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def encode(
        self,
        prompts: str,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[list[int]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> list[PoolingRequestOutput]:
        ...

    @overload  # LEGACY: multi (prompt + optional token ids)
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def encode(
        self,
        prompts: list[str],
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[list[list[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> list[PoolingRequestOutput]:
        ...

    @overload  # LEGACY: single (token ids + optional prompt)
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def encode(
        self,
        prompts: Optional[str] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        *,
        prompt_token_ids: list[int],
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> list[PoolingRequestOutput]:
        ...

    @overload  # LEGACY: multi (token ids + optional prompt)
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def encode(
        self,
        prompts: Optional[list[str]] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        *,
        prompt_token_ids: list[list[int]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> list[PoolingRequestOutput]:
        ...

    @overload  # LEGACY: single or multi token ids [pos-only]
    @deprecated("'prompt_token_ids' will become part of 'prompts'")
    def encode(
        self,
        prompts: None,
        pooling_params: None,
        prompt_token_ids: Union[list[int], list[list[int]]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> list[PoolingRequestOutput]:
        ...

    @deprecate_kwargs(
        "prompt_token_ids",
        is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
        additional_message="Please use the 'prompts' parameter instead.",
    )
    def encode(
        self,
        prompts: Union[Union[PromptType, Sequence[PromptType]],
                       Optional[Union[str, list[str]]]] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[Union[list[int], list[list[int]]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> list[PoolingRequestOutput]:
        """Apply pooling to the hidden states corresponding to the input
        prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each prompts.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.

        Returns:
            A list of ``PoolingRequestOutput`` objects containing the
            pooled hidden states in the same order as the input prompts.

        Note:
            Using ``prompts`` and ``prompt_token_ids`` as keyword parameters is
            considered legacy and may be deprecated in the future. You should
            instead pass them via the ``inputs`` parameter.
        """
        runner_type = self.llm_engine.model_config.runner_type
        if runner_type != "pooling":
            messages = ["LLM.encode() is only supported for pooling models."]

            supported_runner_types = self.llm_engine.model_config \
                .supported_runner_types
            if "pooling" in supported_runner_types:
                messages.append(
                    "Your model supports the 'pooling' runner, but is "
                    f"currently initialized for the '{runner_type}' runner. "
                    "Please initialize vLLM using `--task embed`, "
                    "`--task classify`, `--task score` etc.")

            raise ValueError(" ".join(messages))

        if prompt_token_ids is not None:
            parsed_prompts = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, list[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            parsed_prompts = cast(Union[PromptType, Sequence[PromptType]],
                                  prompts)

        if pooling_params is None:
            # Use default pooling params.
            pooling_params = PoolingParams()
        elif isinstance(pooling_params, PoolingParams):
            pooling_params.verify(self.llm_engine.model_config)
        else:
            for pooling_param in pooling_params:
                pooling_param.verify(self.llm_engine.model_config)

        self._validate_and_add_requests(
            prompts=parsed_prompts,
            params=pooling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return self.engine_class.validate_outputs(outputs,
                                                  PoolingRequestOutput)

    def embed(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        /,
        *,
        use_tqdm: bool = True,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> list[EmbeddingRequestOutput]:
        """
        Generate an embedding vector for each prompt.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each prompts.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.

        Returns:
            A list of ``EmbeddingRequestOutput`` objects containing the
            embedding vectors in the same order as the input prompts.
        """
        if self.llm_engine.model_config.task != "embed":
            raise ValueError(
                "Embedding API is only enabled for `--task embed`")

        items = self.encode(prompts,
                            use_tqdm=use_tqdm,
                            pooling_params=pooling_params,
                            lora_request=lora_request,
                            prompt_adapter_request=prompt_adapter_request)

        return [EmbeddingRequestOutput.from_base(item) for item in items]

    def classify(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        /,
        *,
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> list[ClassificationRequestOutput]:
        """
        Generate class logits for each prompt.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each prompts.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.

        Returns:
            A list of ``ClassificationRequestOutput`` objects containing the
            embedding vectors in the same order as the input prompts.
        """
        if self.llm_engine.model_config.task != "classify":
            raise ValueError(
                "Classification API is only enabled for `--task classify`")

        items = self.encode(prompts,
                            use_tqdm=use_tqdm,
                            lora_request=lora_request,
                            prompt_adapter_request=prompt_adapter_request)

        return [ClassificationRequestOutput.from_base(item) for item in items]

    def _embedding_score(
        self,
        tokenizer: AnyTokenizer,
        text_1: list[Union[str, TextPrompt, TokensPrompt]],
        text_2: list[Union[str, TextPrompt, TokensPrompt]],
        truncate_prompt_tokens: Optional[int] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> list[ScoringRequestOutput]:

        encoded_output: list[PoolingRequestOutput] = self.encode(
            text_1 + text_2,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request)

        encoded_output_1: list[PoolingRequestOutput] = encoded_output[
            0:len(text_1)]
        encoded_output_2: list[PoolingRequestOutput] = encoded_output[
            len(text_1):]

        if len(encoded_output_1) == 1:
            encoded_output_1 = encoded_output_1 * len(encoded_output_2)

        scores = _cosine_similarity(tokenizer=tokenizer,
                                    embed_1=encoded_output_1,
                                    embed_2=encoded_output_2)

        items = self.engine_class.validate_outputs(scores,
                                                   PoolingRequestOutput)
        return [ScoringRequestOutput.from_base(item) for item in items]

    def _cross_encoding_score(
        self,
        tokenizer: AnyTokenizer,
        text_1: list[str],
        text_2: list[str],
        truncate_prompt_tokens: Optional[int] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> list[ScoringRequestOutput]:

        if isinstance(tokenizer, MistralTokenizer):
            raise ValueError(
                "Score API is only enabled for `--task embed or score`")

        if len(text_1) == 1:
            text_1 = text_1 * len(text_2)

        input_pairs = [(t1, t2) for t1, t2 in zip(text_1, text_2)]

        pooling_params = PoolingParams()

        tokenization_kwargs: dict[str, Any] = {}
        if truncate_prompt_tokens is not None:
            tokenization_kwargs["truncation"] = True
            tokenization_kwargs["max_length"] = truncate_prompt_tokens

        parsed_prompts = []

        for q, t in input_pairs:
            prompt_inputs = tokenizer(text=q,
                                      text_pair=t,
                                      **tokenization_kwargs)
            engine_prompt = TokensPrompt(
                prompt_token_ids=prompt_inputs["input_ids"],
                token_type_ids=prompt_inputs.get("token_type_ids"))
            parsed_prompts.append(engine_prompt)

        self._validate_and_add_requests(
            prompts=parsed_prompts,
            params=pooling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        items = self.engine_class.validate_outputs(outputs,
                                                   PoolingRequestOutput)

        return [ScoringRequestOutput.from_base(item) for item in items]

    def score(
        self,
        text_1: Union[SingletonPrompt, Sequence[SingletonPrompt]],
        text_2: Union[SingletonPrompt, Sequence[SingletonPrompt]],
        /,
        *,
        truncate_prompt_tokens: Optional[int] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> list[ScoringRequestOutput]:
        """Generate similarity scores for all pairs ``<text,text_pair>``.

        The inputs can be ``1 -> 1``, ``1 -> N`` or ``N -> N``.
        In the ``1 - N`` case the ``text_1`` sentence will be replicated ``N``
        times to pair with the ``text_2`` sentences.
        The input pairs are used to build a list of prompts for the
        cross encoder model. This class automatically batches the prompts,
        considering the memory constraint. For the best performance, put all
        of your texts into a single list and pass it to this method.

        Args:
            text_1: can be a single prompt or a list of prompts, in which
                case it has to have the same length as the ``text_2`` list
            text_2: The texts to pair with the query to form the input
                to the LLM. See :class:`~vllm.inputs.PromptType` for
                more details about the format of each prompts.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.

        Returns:
            A list of ``ScoringRequestOutput`` objects containing the
            generated scores in the same order as the input prompts.
        """
        runner_type = self.llm_engine.model_config.runner_type
        if runner_type != "pooling":
            messages = ["LLM.score() is only supported for pooling models."]

            supported_runner_types = self.llm_engine.model_config \
                .supported_runner_types
            if "pooling" in supported_runner_types:
                messages.append(
                    "Your model supports the 'pooling' runner, but is "
                    f"currently initialized for the '{runner_type}' runner. "
                    "Please initialize vLLM using `--task embed`, "
                    "`--task classify`, `--task score` etc.")

            raise ValueError(" ".join(messages))

        if self.llm_engine.model_config.task not in ("embed", "score"):
            raise ValueError(
                "Score API is only enabled for `--task embed or --task score`")

        # the tokenizer for models such as
        # "cross-encoder/ms-marco-MiniLM-L-6-v2" doesn't support passing
        # lists of tokens to the `text` and `text_pair` kwargs
        tokenizer = self.llm_engine.get_tokenizer()

        def ensure_str(prompt: SingletonPrompt):
            if isinstance(prompt, dict):
                if "multi_modal_data" in prompt:
                    raise ValueError("Multi-modal prompt is not "
                                     "supported for scoring")
                elif "prompt_token_ids" in prompt:
                    prompt = tokenizer.decode(
                        cast(TokensPrompt, prompt)["prompt_token_ids"])
                elif "prompt" in prompt:
                    prompt = cast(TextPrompt, prompt)["prompt"]
            assert type(prompt) is str
            return prompt

        if isinstance(text_1, (str, dict)):
            # Convert a single prompt to a list.
            text_1 = [text_1]
        input_text_1: list[str] = [ensure_str(t) for t in text_1]

        if isinstance(text_2, (str, dict)):
            # Convert a single prompt to a list.
            text_2 = [text_2]
        input_text_2: list[str] = [ensure_str(t) for t in text_2]

        _validate_score_input_lens(input_text_1, input_text_2)

        if self.llm_engine.model_config.is_cross_encoder:
            return self._cross_encoding_score(tokenizer, input_text_1,
                                              input_text_2,
                                              truncate_prompt_tokens, use_tqdm,
                                              lora_request,
                                              prompt_adapter_request)
        else:
            return self._embedding_score(
                tokenizer,
                input_text_1,  # type: ignore[arg-type]
                input_text_2,  # type: ignore[arg-type]
                truncate_prompt_tokens,
                use_tqdm,
                lora_request,
                prompt_adapter_request)

    def start_profile(self) -> None:
        self.llm_engine.start_profile()

    def stop_profile(self) -> None:
        self.llm_engine.stop_profile()

    def reset_prefix_cache(self, device: Optional[Device] = None) -> bool:
        return self.llm_engine.reset_prefix_cache(device)

    def sleep(self, level: int = 1):
        """
        Put the engine to sleep. The engine should not process any requests.
        The caller should guarantee that no requests are being processed
        during the sleep period, before `wake_up` is called.

        Args:
            level: The sleep level. Level 1 sleep will offload the model 
                weights and discard the kv cache. The content of kv cache 
                is forgotten. Level 1 sleep is good for sleeping and waking
                up the engine to run the same model again. The model weights 
                are backed up in CPU memory. Please make sure there's enough 
                CPU memory to store the model weights. Level 2 sleep will 
                discard both the model weights and the kv cache. The content 
                of both the model weights and kv cache is forgotten. Level 2 
                sleep is good for sleeping and waking up the engine to run a
                different model or update the model, where previous model 
                weights are not needed. It reduces CPU memory pressure.
        """
        self.reset_prefix_cache()
        self.llm_engine.sleep(level=level)

    def wake_up(self, tags: Optional[list[str]] = None):
        """
        Wake up the engine from sleep mode. See the :meth:`sleep` method
        for more details.
        
        Args:
            tags: An optional list of tags to reallocate the engine memory 
                for specific memory allocations. Values must be in 
                ("weights", "kv_cache",). If None, all memory is reallocated.
                wake_up should be called with all tags (or None) before the 
                engine is used again.
        """
        self.llm_engine.wake_up(tags)

    # LEGACY
    def _convert_v1_inputs(
        self,
        prompts: Optional[Union[str, list[str]]],
        prompt_token_ids: Optional[Union[list[int], list[list[int]]]],
    ):
        # skip_tokenizer_init is now checked in engine

        if prompts is not None:
            prompts = [p["content"] for p in parse_and_batch_prompt(prompts)]
        if prompt_token_ids is not None:
            prompt_token_ids = [
                p["content"] for p in parse_and_batch_prompt(prompt_token_ids)
            ]

        num_requests = None
        if prompts is not None:
            num_requests = len(prompts)
        if prompt_token_ids is not None:
            if (num_requests is not None
                    and num_requests != len(prompt_token_ids)):
                raise ValueError("The lengths of prompts and prompt_token_ids "
                                 "must be the same.")

            num_requests = len(prompt_token_ids)
        if num_requests is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")

        parsed_prompts: list[PromptType] = []
        for i in range(num_requests):
            item: PromptType

            if prompts is not None:
                item = TextPrompt(prompt=prompts[i])
            elif prompt_token_ids is not None:
                item = TokensPrompt(prompt_token_ids=prompt_token_ids[i])
            else:
                raise AssertionError

            parsed_prompts.append(item)

        return parsed_prompts

    def _validate_and_add_requests(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        guided_options: Optional[GuidedDecodingRequest] = None,
        priority: Optional[list[int]] = None,
    ) -> None:
        if guided_options is not None:
            warnings.warn(
                "guided_options_request is deprecated, use "
                "SamplingParams.guided_decoding instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(prompts, (str, dict)):
            # Convert a single prompt to a list.
            prompts = [prompts]

        num_requests = len(prompts)
        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")
        if isinstance(lora_request,
                      list) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request "
                             "must be the same.")

        for sp in params if isinstance(params, list) else (params, ):
            if isinstance(sp, SamplingParams):
                self._add_guided_params(sp, guided_options)

                # We only care about the final output
                sp.output_kind = RequestOutputKind.FINAL_ONLY

        # Add requests to the engine.
        for i, prompt in enumerate(prompts):
            self._add_request(
                prompt,
                params[i] if isinstance(params, Sequence) else params,
                lora_request=lora_request[i] if isinstance(
                    lora_request, Sequence) else lora_request,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority[i] if priority else 0,
            )

    def _add_request(
        self,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(
            request_id,
            prompt,
            params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
        )

    def _add_guided_params(
            self,
            params: SamplingParams,
            guided_options: Optional[GuidedDecodingRequest] = None):
        if guided_options is None:
            return params

        if params.guided_decoding is not None:
            raise ValueError("Cannot set both guided_options_request and "
                             "params.guided_decoding.")

        params.guided_decoding = GuidedDecodingParams(
            json=guided_options.guided_json,
            regex=guided_options.guided_regex,
            choice=guided_options.guided_choice,
            grammar=guided_options.guided_grammar,
            json_object=guided_options.guided_json_object,
            backend=guided_options.guided_decoding_backend,
            whitespace_pattern=guided_options.guided_whitespace_pattern,
            structural_tag=guided_options.structural_tag,
        )
        return params
    # 负责自回归地调用step，所有的sample并行地进行step
    def _run_engine(
            self, *, use_tqdm: bool
    ) -> list[Union[RequestOutput, PoolingRequestOutput]]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
            )

        # Run the engine.
        outputs: list[Union[RequestOutput, PoolingRequestOutput]] = []
        total_in_toks = 0
        total_out_toks = 0
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step() #  print("step once")
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # Calculate tokens only for RequestOutput
                            n = len(output.outputs)
                            assert output.prompt_token_ids is not None
                            total_in_toks += len(output.prompt_token_ids) * n
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            total_out_toks += sum(
                                len(stp.token_ids) for stp in output.outputs)
                            out_spd = (total_out_toks /
                                       pbar.format_dict["elapsed"])
                            pbar.postfix = (
                                f"est. speed input: {in_spd:.2f} toks/s, "
                                f"output: {out_spd:.2f} toks/s")
                            pbar.update(n)
                        else:
                            pbar.update(1)

        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))
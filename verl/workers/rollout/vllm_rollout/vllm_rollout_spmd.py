# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import asyncio
import getpass
import logging
import os
import pickle
import socket
from contextlib import contextmanager
from types import MethodType
from typing import Any

import numpy as np
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import DictConfig, ListConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.config import RolloutConfig
from verl.workers.rollout.base import BaseRollout

# import beam search
# overwrite the LLM
from verl.workers.rollout.vllm_rollout.llm import LLM
from verl.workers.rollout.vllm_rollout.pref_instance_cls import (SAGEParams, OutputObject)

from verl.workers.rollout.vllm_rollout.utils import (
    fetch_single_prompt_list, 
    fetch_group_first_elements,
    get_adaptive_merge_ratio, 
    create_attention_mask, 
    generate_uniform_indicator_tensor,
    convert_generate_to_sage )

from verl.utils.reward_score.__init__ import default_compute_score

from tqdm import tqdm 
import math

import time
import json

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config 

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}

        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        compilation_config = {}

        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        # enforce_eager must be False to use cudagraph
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_config["compilation_config"] = CompilationConfig(
                    level=CompilationLevel.PIECEWISE, cudagraph_capture_sizes=cudagraph_capture_sizes
                )
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len, # max_prompt_length + max_response_length
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **compilation_config,
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=VLLM_SLEEP_LEVEL)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

        self.sage_batch_size = self.config.beam_batch_size
        self.sage_params = SAGEParams(exploration_width=self.config.beam_width,  
                                            max_tokens=self.sampling_params.max_tokens-10, 
                                            max_think_tokens=int((self.sampling_params.max_tokens-10) * self.config.think_token_ratio), 
                                            token_penalty=self.config.token_penalty, 
                                            diverse_penalty= self.config.diverse_penalty,
                                            insert_sot=self.config.insert_sot,
                                            enable_sort=self.config.enable_sort,
                                            min_think_token=self.config.min_think_token,
                                            required_completions=self.config.required_completions,
                                            sort_freq=self.config.sort_freq,
                                            norm_beam_scores=self.config.norm_beam_scores,
                                            logprob_weight=self.config.logprob_weight,
                                            token_weight=self.config.token_weight,
                                            diverse_weight=self.config.diverse_weight,
                                            accept_ratio=self.config.accept_ratio,
                                            log_mode=False,
                                            max_step_num=self.config.max_step_num,
                                            per_step_token=self.config.per_step_token,
                                            token_constrain=self.config.token_constrain
                                            )
        
        self.required_completions = self.config.required_completions if self.config.required_completions else self.sage_params.exploration_width  
        # overwrite following two settings
        self.validate_first = self.config.validate_first
        if self.config.adaptive_merge:
            self.validate_first=True

        self.tokenizer = self.inference_engine.get_tokenizer()
        self.sot_id = self.tokenizer.convert_tokens_to_ids("<think>")
        self.eot_id = self.tokenizer.convert_tokens_to_ids("</think>")
        self.skip_think_ids = [self.sot_id, self.eot_id]

        



    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        print("in the generate sequences") 
        # TODO control the beam search generation
        generation_mode = self.config.get("generation_mode", "vllm")

 
        tokenizer = self.tokenizer

        # TODO used to pass metrics
        meta_info={}


        # batch size = n * batch size = 8 * 2560 =2048
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        # Attention ! the batch size = n * num_prompts
        batch_size = idx.size(0)
        

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]# inputs

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        if not is_validate:
            data_sources = prompts.non_tensor_batch["data_source"].tolist()
            ground_truths = prompts.non_tensor_batch["reward_model"].tolist()
            single_data_sources=fetch_single_prompt_list(data_sources, self.config.n)
            single_ground_truths=fetch_single_prompt_list(ground_truths, self.config.n)


        # if not is_validate:
        #     breakpoint()

        if not do_sample: # check the vanilla output args
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run

        with self.update_sampling_params(**kwargs): 

            if "sage" in generation_mode and not is_validate:
                # breakpoint()
                # TODO set the step tracing
                global_steps = prompts.meta_info["global_steps"]
                total_steps = prompts.meta_info["total_training_steps"]


                # TODO: inputs from AAABBBCCC to ABC:
                beam_inputs = fetch_single_prompt_list(vllm_inputs, self.config.n)

                # TODO: devide into batchs
                prompt_batches = [beam_inputs[i:i + self.sage_batch_size] for i in range(0, len(beam_inputs), self.sage_batch_size)]

                sage_outputs= []
                # apply beam search
                if not self.config.apply_ablation:
                    print("start beam search")
                    # TODO: generate  (construct group outputs)
                    for batch_idx, prompt_batch in enumerate(tqdm(prompt_batches, desc="Processing batches")):
                        if "step" in generation_mode:
                            batch_outputs=self.inference_engine.SAGE(prompt_batch, self.sage_params) 
                        else:
                            batch_outputs=self.inference_engine.TSearch_with_Phi(prompt_batch, self.sage_params)

                        sage_outputs=sage_outputs+batch_outputs
                # TODO: if apply ablation, replace beam search outputs with vllm
                else:
                    print("ablation vllm rollout")
                    ablation_require_num = self.required_completions = self.config.required_completions if self.config.required_completions > 0 else self.sage_params.exploration_width  
                    ablation_prompt_repeat_time = [ablation_require_num] * len(beam_inputs)
                    ablation_batchs  = fetch_group_first_elements(vllm_inputs, self.config.n, ablation_prompt_repeat_time)
                    ablation_sampling_params = self.sampling_params
                    ablation_sampling_params.prompt_logprobs = 0
                    ablation_vllm_outputs= self.inference_engine.generate(
                        prompts=ablation_batchs,  # because we have already convert it to prompt token id
                        sampling_params=ablation_sampling_params,
                        lora_request=lora_requests,
                        use_tqdm=True,
                    ) 

                    sage_outputs = convert_generate_to_sage(ablation_vllm_outputs, ablation_require_num, self.sampling_params.max_tokens)

                # TODO: output from ABC to AAABBBCCC 
                outputs=[]
                scores_across_groups=[]
                json_list = []
                if_skip_think = []
                think_length = []
                rollout_length = []
                # controll the repeat time of each prompt
                for id, output in enumerate(sage_outputs):
                    scores_within_group=[]
                    for seq in output.sequences:
                        res_start_idx=seq.prompt_length-1
                        # TODO record the rollout
                        json_text = {"prompt_id": id, 
                                     "query": tokenizer.decode(seq.tokens[:res_start_idx]),
                                     "response": tokenizer.decode(seq.tokens[res_start_idx:])} 
                        json_list.append(json_text)

                        # TODO add the beam search rollout metrics
                        rollout_length.append(len(seq.tokens[res_start_idx:]))
                        if self.eot_id in seq.tokens:
                            think_end_idx = seq.tokens.index(self.eot_id)
                        else:
                            think_end_idx = len(seq.tokens)
                        
                        think_length.append(len(seq.tokens[res_start_idx:think_end_idx]))
                        # use the query_response, so it's ok
                        if  seq.tokens[res_start_idx:res_start_idx+2] == self.skip_think_ids:
                            if_skip_think.append(1)
                        else:
                            if_skip_think.append(0)

                        seq.token_ids=seq.tokens[res_start_idx:]

                        seq.logprobs = seq.logprobs[res_start_idx:]
                        outputs.append(OutputObject(outputs=[seq]))

                        if self.validate_first:
                            data_source=single_data_sources[id]
                            groud_truth=single_ground_truths[id]["ground_truth"]
                            cur_solution_str = tokenizer.decode(seq.token_ids)
                            # the format is different when the data_source is different
                            cur_score=default_compute_score(data_source=data_source, solution_str=cur_solution_str, ground_truth=groud_truth) # get the reward
                            if "dapo" in data_source:
                                cur_score=cur_score["acc"]
                            scores_within_group.append(cur_score)

                    scores_across_groups.append(scores_within_group)

                # TODO record the beam search rollout and metrics
                meta_info["beam_rollout_len"] = int(np.array(rollout_length).mean())
                meta_info["beam_skip_think_num"] = int(np.array(if_skip_think).sum())
                meta_info["beam_think_len"] = int(np.array(think_length).mean())

  
                # TODO log the rollout
                if global_steps >= 0 and global_steps <= 900:
                    # write the rollout to text
                    file_dir = "/path/to/SAGE/verl/workers/rollout/vllm_rollout/rollout_outputs"
                    file_path = os.path.join(file_dir, f"step:{global_steps}.json")
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)  
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(json_list, f, indent=2, ensure_ascii=False)

                # the time that vllm should generate for each group
                prompt_repeat_time = [self.config.n - self.required_completions] * len(beam_inputs)

                # entirely use vllm generate for low accuracy question 
                if self.validate_first:
                    print("filter the low acc beams...")
                    acc_across_groups = [np.sum(scores)/self.required_completions for scores in scores_across_groups]
                    # if accuracy is low, then adapt the vanilla output
                    if self.config.adaptive_merge: 
                        print("compute adaptive merge ratio")
                        prompt_repeat_time = [max(self.config.n - self.required_completions, int(r * self.config.n)) for r in get_adaptive_merge_ratio(acc_across_groups, self.config.beam_min_acc)]
                    else:
                        prompt_repeat_time = [self.config.n - self.required_completions if acc >= self.config.beam_min_acc else self.config.n  for acc in acc_across_groups]

                    if self.config.time_truncation: # aggressively add the beam search ratio
                        minimal_vanilla_num = int(math.cos(global_steps / total_steps * math.pi / 2 ) * self.config.n) # a min threshold of vanilla sampling
                        prompt_repeat_time = [max(minimal_vanilla_num, p) for p in prompt_repeat_time]
                        meta_info["minial_vanilla_num"] = minimal_vanilla_num

                    avg_beam_acc = np.sum(acc_across_groups)/len(beam_inputs)
                    meta_info["avg_beam_acc"] = avg_beam_acc
                    meta_info["no_beam_group_num"] = prompt_repeat_time.count(self.config.n)
                    meta_info["avg_group_beam_num"] = self.config.n - sum(prompt_repeat_time)/len(prompt_repeat_time)
                    

                # TODO merge the generation output shape:
                mix_vllm_inputs = fetch_group_first_elements(vllm_inputs, self.config.n, prompt_repeat_time)

                vllm_outputs= self.inference_engine.generate(
                    prompts=mix_vllm_inputs,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=True,
                ) 

                # TODO  mix the vllm output and the beam search output, if not use the beam search, the n_b will be 0
                print("merge the outputs")
                mixed_outputs=[]
                v_start = 0
                b_start = 0
                # TODO mark the rollout source
                rollout_source = []

                for i in range(batch_size//self.config.n):
                    n_b = self.config.n - prompt_repeat_time[i]
                    n_v = prompt_repeat_time[i]
                    for j in range(b_start, b_start + n_b):
                        mixed_outputs.append(outputs[j])
                        rollout_source.append(True)
                    for j in range(v_start, v_start + n_v):
                        mixed_outputs.append(vllm_outputs[j])
                        rollout_source.append(False)
                    v_start += n_v
                    b_start += self.required_completions 

                # mixed_outputs = [outputs[i * n_b: (i+1) * n_b ] + vllm_outputs[i * n_v : (i+1) * n_v] for i in range(batch_size//self.config.n)]
                outputs= mixed_outputs
                rollout_source = torch.tensor(rollout_source, dtype = torch.bool).unsqueeze(1)

            else: 
                print("vanilla generation...")
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=is_validate,
                ) # TODO modified to our method

                rollout_source = torch.ones((batch_size, 1), dtype=torch.bool)

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = [] # token_id list
            rollout_log_probs = [] # log_prob list
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            think_len_list = []
            # TODO compute think idx of each seq
            for id, res in enumerate(response): 
                cur_think_len= res.index(self.eot_id)+1 if self.eot_id in res else len(res)
                think_len_list.append(cur_think_len)

            meta_info.update({"group_think_len": torch.tensor(think_len_list, dtype=torch.float32).mean().item()})


            # TODO  add length penalty of o1-pruner
            if not is_validate and self.config.length_penalty:
                
                start = time.time()
                # TODO compute the length reward
                # get each response's length
                last_token_pos = [len(r)-1 for r in response]
                # group-wise length penalty
                response_length_tensor = torch.tensor(last_token_pos, dtype=torch.float32).view(-1, self.config.n)+1
                mean_response_length = response_length_tensor.mean(dim=1, keepdim=True) # B,1
                raw_length_reward_tensor = (mean_response_length / response_length_tensor - 1).flatten() # [B,N] -> [B*N]

                # TODO compute the reward score
                acc_score_list = []
                
                for id, res in enumerate(response): 
                    data_source=single_data_sources[id // self.config.n]
                    groud_truth=single_ground_truths[id // self.config.n]["ground_truth"]
                    cur_solution_str = tokenizer.decode(res)
                    # the format is different when the data_source is different
                    cur_score=default_compute_score(data_source=data_source, solution_str=cur_solution_str, ground_truth=groud_truth) # get the reward
                    if "dapo" in data_source:
                        cur_score=cur_score["acc"]
                    acc_score_list.append(cur_score)

                # TODO norm the acc score
                score_tensor = torch.tensor(acc_score_list, dtype=torch.float32).view(-1, self.config.n)
                
                # each_group_acc = score_tensor.mean(dim=1, keepdim=True) # B,1
                #  score_reward_tensor = (score_tensor - each_group_acc).flatten() # [B,N] -> [B*N]
                # score_reward_tensor = (score_tensor / (each_group_acc + 1e-12) - 1).flatten() 
                score_reward_tensor = score_tensor.flatten()

                # TODO apply reward clip
                # TODO mask the length reward which is all right  or all wrong
                group_degrad_mask = generate_uniform_indicator_tensor(score_tensor)
                raw_length_reward_tensor = raw_length_reward_tensor.clip(min=-0.1, max=0.1) * group_degrad_mask
                meta_info.update({"length_penalty": raw_length_reward_tensor.mean().item() })
                
                # score_reward_tensor = score_reward_tensor.clip(min=-2, max=2)
                meta_info.update({"acc_score": score_reward_tensor.mean().item() })

               
                # TODO aggregate the acc score and the length score
                # lw = 0.1
                raw_length_reward_tensor = raw_length_reward_tensor + score_reward_tensor

                # TODO if group acc == 1 or 0, then set the reward as the acc reward, remove the norm, and clip the raw score.

                print(f"pre compute reward, last for {time.time()-start} s")

                # final reward
                meta_info.update({"length_reward": raw_length_reward_tensor.mean().item() })
                


            # padding to the same length
            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)


        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)



        batch = TensorDict(
            {
                "prompts": idx,  #[B,S]
                "responses": response, #[B,S]
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "rollout_source": rollout_source.to(idx.device), #[B,1] whether come from beam search
                "think_att_mask": create_attention_mask(think_len_list, response.shape[1]).to(idx.device)
            },
            batch_size=batch_size,
        ) 

        if not is_validate and self.config.length_penalty:
            tensor_length_reward = torch.zeros(response.shape[0], response.shape[1], dtype=torch.float32, device=response.device)
            tensor_length_reward[torch.arange(response.shape[0]), last_token_pos] = raw_length_reward_tensor.to(response.device)
            batch.update({"last_token_pos": torch.tensor(last_token_pos, dtype=torch.float32).unsqueeze(1)})
            batch.update({"length_reward": tensor_length_reward})

        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        # if not is_validate:
        #     breakpoint()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        loop = asyncio.get_running_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    def _get_free_port(self):
        ip = ray.util.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    async def _loop_forever(self):
        while True:
            message = await self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = await self._execute_method(method, *args, **kwargs)
            await self.socket.send(pickle.dumps(result))

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0 if not ray_noset_visible_devices() else int(os.environ.get("RAY_LOCAL_RANK", 0))
        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        elif method == "sleep":
            return await self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return await self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
 
    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address

    async def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    async def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    async def generate(self, *args, **kwargs):
        """Generate sequence with token-in-token-out."""
        raise NotImplementedError

    async def chat_completion(self, json_request):
        """OpenAI chat completion API."""
        raise NotImplementedError

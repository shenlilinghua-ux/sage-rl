# Does Your Reasoning Model Implicitly Know When to Stop Thinking?

This repository contains the unoffical reproduction code for our paper: Does Your Reasoning Model Implicitly Know When to Stop Thinking?

Original paper link: https://arxiv.org/abs/2602.08354

Original paper homepage: https://hzx122.github.io/sage-rl/

Original paper's Hugging Face page: https://huggingface.co/papers/2602.08354



The reproduction code consists of two components: SAGE and SAGE-RL.

For SAGE, we have rewritten vllm (https://github.com/vllm-project/vllm). The core code is located in `verl/workers/rollout/vllm_rollout/llm.py`, which contains the implementation of TSearch, SAGE, and its variants.

For SAGE-RL, we implemented it based on verl (https://github.com/verl-project/verl). The main deployment implementation is in `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`, which includes the implementation of mixed sampling using SAGE during the rollout phase.


## Tips for Running SAGE-RL

Given the various inquiries about SAGE-RL, we provide a list of tips to help you reproduce our paper results and achieve better outcomes for running SAGE-RL on your own tasks. 

### Environment

We provide  ``requirements.txt``  including the python package versions we used in our experiments. For optimal reproducibility, we recommend using the same package versions. However, please note that results may still vary due to differences in hardware configurations and CUDA versions, etc.



## Install Requirements
First, configure the environment required by verl, then install the necessary packages based on `requirements.txt` in our directory.

1. Set up the environment dependencies for the verl framework first.
2. Navigate to our project directory and install the required packages via the `requirements.txt` file.
3. Run the following scripts in the root directory:

        sudo pip3 uninstall ray bytedray -y
        pip3 install fastapi==0.115.12 tensordict==0.6.2 torchdata==0.11.0 peft==0.15.2
        pip3 install transformers==4.51.3

        pip3 install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6  

        pip3 install tensorboard

        cd /path/to/SAGE
        pip3 install -e .

        sudo apt-get install tmux

        pip install latex2sympy2
        pip install evaluate
        pip install word2number
        pip install sentence_transformers
        pip install matplotlib
        pip install tenacity

        # for higher evaluation accuracy
        pip install math-verify
        # fix the math-verify bug
        pip uninstall -y antlr4-python3-runtime
        pip install antlr4-python3-runtime==4.9.3


## Training Scripts

For specific  tutorial for running the code, please refer to the verl codebase https://github.com/verl-project/verl.
The complete experimental scripts are located in the `main_exp` directory.
One Case Study：
    
    PYTHONUNBUFFERED=1 python3  -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$train_files \
        data.val_files=$test_files \
        data.train_batch_size=$BATCH_SIZE \
        data.max_prompt_length=1024 \
        data.max_response_length=$max_len \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=$main_model \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
        actor_rollout_ref.actor.kl_loss_coef=$kl_penalty \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.clip_ratio_high=0.2 \
        actor_rollout_ref.actor.beam_clip_ratio_high=$beam_clip_ratio_high \
        actor_rollout_ref.actor.think_boost_ratio=$think_boost_ratio \
        actor_rollout_ref.actor.rollout_boost_ratio=$rollout_boost_ratio \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.nccl_timeout=$nccl_timeout \
        actor_rollout_ref.rollout.max_num_batched_tokens=20000 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
        actor_rollout_ref.rollout.n=$group_size \
        actor_rollout_ref.rollout.generation_mode=$generation_mode \
        actor_rollout_ref.rollout.token_penalty=$token_penalty \
        actor_rollout_ref.rollout.diverse_penalty=$diverse_penalty \
        actor_rollout_ref.rollout.think_token_ratio=0.9 \
        actor_rollout_ref.rollout.beam_batch_size=16 \
        actor_rollout_ref.rollout.insert_sot=$insert_sot \
        actor_rollout_ref.rollout.enable_sort=$enable_sort \
        actor_rollout_ref.rollout.merge_ratio=$merge_ratio \
        actor_rollout_ref.rollout.validate_first=$validate_first \
        actor_rollout_ref.rollout.adaptive_merge=$adaptive_merge \
        actor_rollout_ref.rollout.beam_min_acc=$beam_min_acc \
        actor_rollout_ref.rollout.time_truncation=$time_truncation \
        actor_rollout_ref.rollout.min_think_token=$min_think_token \
        actor_rollout_ref.rollout.required_completions=$required_completions \
        actor_rollout_ref.rollout.apply_ablation=$apply_ablation \
        actor_rollout_ref.rollout.sort_freq=$sort_freq \
        actor_rollout_ref.rollout.norm_beam_scores=$norm_beam_scores \
        actor_rollout_ref.rollout.logprob_weight=$logprob_weight \
        actor_rollout_ref.rollout.token_weight=$token_weight \
        actor_rollout_ref.rollout.diverse_weight=$diverse_weight \
        actor_rollout_ref.rollout.length_penalty=$length_penalty \
        actor_rollout_ref.rollout.beam_width=$beam_width \
        actor_rollout_ref.rollout.accept_ratio=$accept_ratio \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb','tensorboard'] \
        trainer.project_name=$project_name \
        trainer.experiment_name=$experiment_name \
        trainer.n_gpus_per_node=$NUM_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=200 \
        trainer.test_freq=10 \
        trainer.total_epochs=1 \
        trainer.resume_mode='disable' \
        trainer.validation_data_dir=/path/to/$project_name/$experiment_name/val_outputs \
        trainer.default_local_dir=/path/tp/$project_name/$experiment_name 


## Citation
If you find this repository helpful, please cite the original paper:

```bibtex
@article{huang2026does,
  title={Does Your Reasoning Model Implicitly Know When to Stop Thinking?},
  author={Huang, Zixuan and Xia, Xin and Ren, Yuxi and Zheng, Jianbin and Wang, Xuanda and Zhang, Zhixia and Xie, Hongyan and Liang, Songshi and Chen, Zehao and Xiao, Xuefeng and others},
  journal={arXiv preprint arXiv:2602.08354},
  year={2026}
}









export PYTHONPATH=:$PYTHONPATH 


NUM_GPUS=8


BATCH_SIZE=32
PPO_MINI_BATCH_SIZE=16
MICRO_BATCH_SIZE=1
ROLLOUT_GPU_MEMORY_UTILIZATION=0.8  
TP=4 


dapo_math_path=/path/to/your_data/verl_data/dapo_math/math_style_train.parquent
AIME_2024=/path/to/your_data/verl_data/AIME_2024/aime-2024.parquet
gsm8k_train_path=/path/to/your_data/verl_data/gsm8k/train.parquet
gsm8k_test_path=/path/to/your_data/verl_data/gsm8k/test.parquet
math_train_path=/path/to/your_data/verl_data/math/train.parquet
math_test_path=/path/to/your_data/verl_data/math/test.parquet
dapo_train_path=/path/to/your_data/verl_data/dapo-math-17k_dedup.parquet
# aime2024_test_path=/path/to/your_data/verl_data/aime-2024.parquet
math500_test_path=/path/to/your_data/verl_data/math500_test.parquet
math_train_level_3_to_5_path=/path/to/your_data/verl_data/math/train_level_3.parquent 

# train_files="['$math_train_path','$dapo_math_path']"
train_files="['$dapo_math_path','$math_train_level_3_to_5_path']"
test_files="['$math500_test_path']"



main_model="/path/to/your_data/models/DeepSeek-R1-Distill-Qwen-1.5B"

# response setting
max_len=8192


# this setting will replace the beam search outputs  with vllm
apply_ablation=false

# rollout settings
generation_mode="step_sage"
group_size=8
beam_width=2
required_completions=1
insert_sot=false
enable_sort=true
token_penalty=false
diverse_penalty=false
min_think_token=-1


sort_freq=-1
norm_beam_scores=false
logprob_weight=0.8
token_weight=0.1
diverse_weight=0.1

accept_ratio=1.0

# reward
length_penalty=false

# loss
beam_clip_ratio_high=0.2
think_boost_ratio=1.0
rollout_boost_ratio=1.0



# the group merge setting
merge_ratio=0.5 # the ratio of vanilla vllm outputs, decide the last vllm num
validate_first=true # whether valide the beam search results before use.
adaptive_merge=false # this setting enables adaptive merge ratio for each group, which will cover the validate_first and the merge_ratio.
beam_min_acc=0 # if the avg acc of beam search lower than this, we will dropout all beam search rollout. truncation based on acc
time_truncation=false # truncation with time, set as a filter to cut the beams in the early period


# hyper params
kl_penalty=0.001

# enable long time rollout
nccl_timeout=1200

# wandb settings
suffix=$(date +%Y%m%d%H%M)
project_name="beam_deepseek_qwen_1.5b_dapo_math_math500"
experiment_name="${generation_mode}_gs_${group_size}_width_${beam_width}_require_${required_completions}_lp_${length_penalty}_vf_${validate_first}_mc_${beam_min_acc}_am_${adaptive_merge}_mtt_${min_think_token}_ab_${apply_ablation}_sf_${sort_freq}_nbs${norm_beam_scores}_dp_${diverse_penalty}${suffix}"
experiment_name="no_boost_adv_${generation_mode}_sort_${enable_sort}_gs_${group_size}_width_${beam_width}_require_${required_completions}_lp_${length_penalty}_vf_${validate_first}_ab_${apply_ablation}_${suffix}"


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
    trainer.validation_data_dir=/path/to/your_data/sage_rl_results/$project_name/$experiment_name/val_outputs \
    trainer.default_local_dir=/path/to/your_data/sage_rl_results/$project_name/$experiment_name 





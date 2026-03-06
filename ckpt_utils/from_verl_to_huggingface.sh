#!/usr/bin/env bash
set -x

project_name="check"
experiment_name="checkpoint"

# 配置
# CHECKPOINT_ROOT="/path/to/your_data/multi_agent_rl_results/${project_name}/${experiment_name}"
# OUTPUT_ROOT="/path/to/your_data/multi_agent_rl_results/${project_name}/${experiment_name}"

CHECKPOINT_ROOT="$1"


OUTPUT_ROOT=$CHECKPOINT_ROOT

BACKEND="fsdp"

# 查找所有checkpoint并转换
checkpoints=($(find "${CHECKPOINT_ROOT}" -maxdepth 1 -type d -name "global_step_*" | sort -V))
mkdir -p "${OUTPUT_ROOT}"

last_ckpt=${checkpoints[-1]}
checkpoints=($last_ckpt)

for ckpt_path in "${checkpoints[@]}"; do
    ckpt_name=$(basename "${ckpt_path}")
    actor_dir="${ckpt_path}/actor"
    aux_dir="${ckpt_path}/aux_model"
    if [ -d "${actor_dir}" ]; then
        echo "转换 ${actor_dir}..."
        python -m verl.model_merger merge \
            --backend ${BACKEND} \
            --local_dir "${actor_dir}" \
            --target_dir "${OUTPUT_ROOT}/actor/${ckpt_name}"
    fi
    if [ -d "${aux_dir}" ]; then
        echo "转换 ${aux_dir}..."
        python -m verl.model_merger merge \
            --backend ${BACKEND} \
            --local_dir "${aux_dir}" \
            --target_dir "${OUTPUT_ROOT}/aux/${ckpt_name}"
    fi
done

echo "完成！输出目录: ${OUTPUT_ROOT}"
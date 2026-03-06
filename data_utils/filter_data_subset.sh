dapo_math_path=/path/to/verl_data/dapo_math/math_style_train.parquent
n=20
# AIME_2024=/path/to/verl_data/AIME_2024/aime-2024.parquet
data_path=/opt/tiger/aime-2024-math-style.parquet

# /path/to/verl_data/math500_test.parquet

python /path/to/SAGE/data_utils/filter_parquent.py $data_path $n
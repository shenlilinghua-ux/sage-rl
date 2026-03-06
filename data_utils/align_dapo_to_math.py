import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# 输入输出路径（根据实际情况修改）
INPUT_PATH = "/path/to/verl_data/AIME_2024/aime-2024.parquet"  # 替换为你的输入文件路径
OUTPUT_PATH = "/path/to/verl_data/dapo_math/train_fixed1.parquet"  # 替换为处理后的输出路径
OUTPUT_PATH = "/opt/tiger/aime-2024-math-style.parquet"  # 替换为处理后的输出路径

def process_parquet():
    # 读取文件并分批次处理
    parquet_file = pq.ParquetFile(INPUT_PATH)
    writer = None
    current_index = 0  # 从0开始的递增索引

    for batch in parquet_file.iter_batches():
        df = batch.to_pandas()
        # 1. ability字段改为小写'math'
        df['ability'] = 'math'
        df['data_source'] = 'DigitalLearningGmbH/MATH-lighteval'
        # 2. 重构extra_info：递增index + split='train'
        df['extra_info'] = [
            {'index': current_index + i, 'split': 'train'} 
            for i in range(len(df))
        ]
        # df['prompt']=df['source_prompt']
        df['reward_model'] = df['reward_model'].apply(
                lambda x: {'ground_truth': x['ground_truth'], 'style': 'rule'}
            )


        current_index += len(df)  # 更新索引
        
        # 写入输出文件
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_PATH, table.schema)
        writer.write_table(table)
    
    writer.close()
    print(f"处理完成，输出至: {OUTPUT_PATH}，总样本数: {current_index}")

    
if __name__ == "__main__":
    process_parquet()
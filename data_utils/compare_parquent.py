import pandas as pd
import pyarrow.parquet as pq
import argparse
from pathlib import Path
def preview_parquet(file_path, n=10):
    """
    预览 Parquet 文件的前 N 个元素，打印每个字段的值
    
    Args:
        file_path (str): Parquet 文件路径
        n (int): 要预览的元素数量（默认前5个）
    """
    try:
        # 读取前 N 行数据（高效，仅加载所需部分）
        parquet_file = pq.ParquetFile(file_path)
        # 计算需要读取的批次（确保覆盖前 N 行）
        batches = []
        total = 0
        for batch in parquet_file.iter_batches():
            batches.append(batch)
            total += batch.num_rows
            if total >= n:
                break
        # 转换为 DataFrame 并截取前 N 行
        df = pd.concat([batch.to_pandas() for batch in batches]).head(n)
        
        print(f"\n===== Parquet 文件: {file_path} =====")
        print(f"总样本数: {parquet_file.metadata.num_rows}")
        print(f"预览前 {n} 个样本:")
        print("-" * 100)
        
        # 遍历每行，打印字段和值
        for idx, row in df.iterrows():
            print(f"----- 样本 {idx + 1} -----")
            for col in df.columns:
                # 处理长文本（截断显示，避免输出过长）
                value = row[col]
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."  # 长文本截断
                print(f"{col}: {value}")
            print("-" * 100)
            
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")


def print_parquet_schema(file_path):

    # 使用 pyarrow 读取 Parquet 元数据（高效，无需加载全部数据）
    parquet_file = pq.ParquetFile(file_path)
    schema = parquet_file.schema

    # 遍历所有字段并打印名称和类型
    for field in schema:
        print(field.name)
        print(field.logical_type)



def main():
    # 解析命令行参数
    FILE1_PATH = "/path/to/verl_data/AIME_2024/aime-2024.parquet"
    FILE2_PATH = "/path/to/verl_data/math/train.parquet"
    FILE1_PATH = "/path/to/verl_data/math500_test.parquet"
    target_path=FILE1_PATH
    preview_parquet(target_path)
    # target_path=FILE2_PATH
    # preview_parquet(target_path)

    
if __name__ == "__main__":
    main()
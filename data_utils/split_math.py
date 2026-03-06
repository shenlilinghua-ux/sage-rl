import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
from pathlib import Path
import random

def sample_parquet_subset(file_path, n):
    """
    从 Parquet 文件中随机抽取 n 条数据，保存为子集文件
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 读取文件并获取总样本数
    parquet_file = pq.ParquetFile(file_path)
    total_rows = parquet_file.metadata.num_rows
    
    if n <= 0:
        raise ValueError(f"子集大小必须为正数，当前为: {n}")
    if n > total_rows:
        raise ValueError(f"子集大小 ({n}) 超过文件总样本数 ({total_rows})")
    
    print(f"从 {file_path} 中随机抽取 {n} 条数据（总样本数: {total_rows}）")
    
    # 生成随机索引（0到总样本数-1之间的不重复索引）
    random_indices = set(random.sample(range(total_rows), n))
    
    writer = None
    selected_count = 0
    current_global_idx = 0  # 累计全局索引，替代通过row_group获取
    
    try:
        for batch in parquet_file.iter_batches():
            batch_size = len(batch)
            # 当前批次的全局索引范围 [current_global_idx, current_global_idx + batch_size)
            batch_indices = set(range(current_global_idx, current_global_idx + batch_size))
            # 找到当前批次中被选中的索引
            overlap = random_indices & batch_indices
            
            if overlap:
                # 转换批次为DataFrame
                batch_df = batch.to_pandas()
                # 计算在批次内的相对索引（相对于当前批次的偏移量）
                relative_indices = [idx - current_global_idx for idx in overlap]
                # 筛选出选中的行
                selected = batch_df.iloc[relative_indices]
                selected_count += len(selected)
                
                # 写入输出文件
                table = pa.Table.from_pandas(selected)
                if writer is None:
                    output_path = file_path.parent / f"{file_path.stem}_{n}.parquet"
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)
            
            # 更新全局索引
            current_global_idx += batch_size
            
            # 已收集足够样本则退出
            if selected_count == n:
                break
        
        if writer:
            writer.close()
            print(f"子集已保存至: {output_path}")
        else:
            raise RuntimeError("未筛选到任何数据，请检查文件是否为空")
    
    except Exception as e:
        print(f"处理出错: {str(e)}")
        if writer:
            writer.close()
        if 'output_path' in locals() and output_path.exists():
            output_path.unlink()
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 Parquet 文件中随机抽取子集")
    parser.add_argument("file_path", help="输入 Parquet 文件的路径")
    parser.add_argument("n", type=int, help="要抽取的子集大小（正整数）")
    args = parser.parse_args()
    
    sample_parquet_subset(args.file_path, args.n)
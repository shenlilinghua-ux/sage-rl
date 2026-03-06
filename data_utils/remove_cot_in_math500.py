import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
import argparse

def process_prompt_field(prompt_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """处理prompt字段，删除content只保留role"""
    processed_prompts = []
    
    for item in prompt_data:
        # 创建一个只包含role的新字典
        new_item = {'role': item.get('role', 'user')}
        processed_prompts.append(new_item)
    
    return processed_prompts

def process_parquet_file(input_path: str, output_path: str) -> None:
    """
    处理parquet文件，删除prompt字段中的content内容
    
    Args:
        input_path: 输入parquet文件路径
        output_path: 输出parquet文件路径
    """
    print(f"读取文件: {input_path}")
    
    # 读取parquet文件
    df = pd.read_parquet(input_path)
    
    print(f"原始数据行数: {len(df)}")
    print(f"列名: {df.columns.tolist()}")
    
    # 检查是否有prompt列
    if 'prompt' not in df.columns:
        print("错误: 文件中没有'prompt'列")
        return
    
    # 处理每一行的prompt字段
    def process_row(row):

        instruction_following = " Let's think step by step and output the final answer within \\boxed{}."
        prompt_data = row['prompt']
        cur_prompt = row['prompt'][0]['content']
        prompt_data[0]['content'] = cur_prompt.replace(instruction_following, "")
        return prompt_data

    
    # 应用处理函数
    df['prompt'] = df.apply(process_row, axis=1)
    
    # 保存处理后的数据
    print(f"保存到: {output_path}")
    df.to_parquet(output_path, index=False)
    
    # 显示示例数据
    print("\n处理后的示例数据:")
    for i in range(min(3, len(df))):
        print(f"\n第 {i+1} 行:")
        print(f"prompt: {df.iloc[i]['prompt']}")
        if 'answer' in df.columns:
            print(f"answer: {df.iloc[i]['answer']}")

def process_directory(input_dir: str, output_dir: str) -> None:
    """
    处理目录中的所有parquet文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有parquet文件
    parquet_files = list(input_path.glob("*.parquet"))
    parquet_files += list(input_path.glob("*.parq"))
    
    print(f"找到 {len(parquet_files)} 个parquet文件")
    
    for input_file in parquet_files:
        output_file = output_path / input_file.name
        print(f"\n处理文件: {input_file.name}")
        process_parquet_file(str(input_file), str(output_file))

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='处理parquet文件中的prompt字段，删除content内容')
    parser.add_argument('input', help='输入文件或目录路径')
    parser.add_argument('output', help='输出文件或目录路径')
    parser.add_argument('--batch', action='store_true', 
                       help='批量处理目录中的所有parquet文件')
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量处理目录
        process_directory(args.input, args.output)
    else:
        # 处理单个文件
        process_parquet_file(args.input, args.output)
    
    print("\n处理完成！")

if __name__ == "__main__":
    # 直接运行示例
    # 使用方法1: 处理单个文件
    # python script.py input.parquet output.parquet
    
    # 使用方法2: 批量处理目录
    # python script.py input_dir output_dir --batch
    
    # 或者直接在这里指定路径运行
    input_file = "/path/to/verl_data/math500_test.parquet"
    output_file = "/opt/tiger/math500_test_wihout_cot.parquet"
    
    # 运行处理
    process_parquet_file(input_file, output_file)
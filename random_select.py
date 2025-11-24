import json
import random
import os

def sample_jsonl(input_path, sample_size=1000):
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}")
        return

    # 构造输出文件路径 (保存到原目录，文件名前加 sampled_1000_)
    dir_name = os.path.dirname(input_path)
    file_name = os.path.basename(input_path)
    output_path = os.path.join(dir_name, f"sampled_{sample_size}_{file_name}")

    print(f"正在读取文件: {input_path} ...")
    
    data_list = []
    
    try:
        # 读取所有数据
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        print("警告: 跳过一行无法解析的 JSON 数据")
                        continue

        total_count = len(data_list)
        print(f"读取完成，共计 {total_count} 条数据。")

        # 进行随机筛选
        if total_count <= sample_size:
            print(f"数据量小于等于 {sample_size}，将保存所有数据。")
            sampled_data = data_list
        else:
            print(f"正在随机筛选 {sample_size} 条数据...")
            sampled_data = random.sample(data_list, sample_size)

        # 保存到新文件
        print(f"正在保存到: {output_path} ...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in sampled_data:
                # ensure_ascii=False 保证中文能正常显示，而不是转义字符
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print("处理完成！")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 配置路径
    target_file = "/data/jcxy/haolu/workspace/store/train_data/ins_data/Counseling_Report_DPO_Tag.jsonl"
    
    # 执行筛选
    sample_jsonl(target_file, 2000)
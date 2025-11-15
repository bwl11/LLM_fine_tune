import re
import json
import os

def format_text_to_json(input_filename, output_filename):
    """
    读取文本文件，将其按句子分割，并将每句话格式化为
    一个独立的 JSON 对象，逐行写入到输出文件中（JSON Lines 格式）。
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_filename):
        print(f"错误：找不到文件 '{input_filename}'。请确保文件存在于正确的位置。")
        return

    try:
        # 1. 以 UTF-8 编码读取整个文件内容
        with open(input_filename, 'r', encoding='utf-8') as f_in:
            content = f_in.read()

        # 2. 将文本中的换行符替换掉，以防干扰句子分割
        content = content.replace('\n', ' ').replace('\r', '')

        # 3. 使用正则表达式按句尾标点分割文本
        sentences = re.split(r'(?<=[。！？])', content)

        # 4. 打开输出文件，准备逐行写入
        with open(output_filename, 'w', encoding='utf-8') as f_out:
            # 5. 遍历分割后的每一个句子
            for s in sentences:
                # 清理句子前后的多余空白字符
                cleaned_sentence = s.strip()
                
                # 如果清理后句子不为空
                if cleaned_sentence:
                    # 格式化成 "<s>...</s>" 的形式
                    formatted_text = f"<s>{cleaned_sentence}</s>"
                    
                    # 创建独立的字典对象
                    sentence_object = {"text": formatted_text}
                    
                    # 使用 json.dumps() 将单个字典转换为 JSON 字符串
                    # ensure_ascii=False 确保日文正常显示
                    json_line = json.dumps(sentence_object, ensure_ascii=False)
                    
                    # 将这个 JSON 字符串写入文件，并在末尾加上换行符
                    f_out.write(json_line + '\n')

        print(f"处理成功！结果已保存到文件 '{output_filename}' 中。")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# --- 主程序执行部分 ---
if __name__ == "__main__":
    # 定义输入和输出文件名
    input_file = 'data_togawasaki.txt'
    output_file = 'saki_train_data.jsonl'
    
    # 调用函数执行转换
    format_text_to_json(input_file, output_file)
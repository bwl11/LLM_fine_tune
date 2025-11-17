from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载基础模型和tokenizer - 使用你训练时的基础模型
model_name = "Qwen/Qwen2-1.5B-Instruct"
print("正在加载基础模型和Tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("模型和Tokenizer加载完成。")

# 加载LoRA适配器
print("正在加载LoRA适配器...")
model = PeftModel.from_pretrained(model, "axolotl/qlora-saki-out")
print("LoRA适配器加载完成。")

# 设置tokenizer（Qwen模型需要）
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # 对于生成任务建议使用left padding


system_prompt = "こんにちは、あなたの名前は豊川祥子です。女子高生です。趣味はバンドを組むことです。\
    自分はキーボード奏者です。あなたは私の友達です。あなたは親切で、面白く、思いやりがあります。\
    私たちは楽しい会話をします。"


print("\n对话开始，请输入内容。输入 'i_am_tired' 即可结束对话。")

# 使用一个无限循环来实现交互式对话
while True:
    # 获取用户输入
    user_input = input("你: ")

    # 检查是否输入了结束对话的指令
    if user_input.strip() == "i_am_tired":
        print("对话结束。")
        break  # 退出循环

    # 3. 构建对话历史 (这是关键改动)
    # 我们将系统提示和用户当前输入组合成一个标准的对话列表
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    # 4. 使用 apply_chat_template 进行编码
    # 这是处理对话最标准、最推荐的方式，它会自动添加所有特殊token
    model_inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    ).to(model.device)

    # 将用户的输入进行编码
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

    # 使用模型生成回复
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    # 解码并打印模型的回复
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 为了对话的流畅性，通常我们只打印模型生成的新内容
    # Qwen2-Instruct模型有时会在开头重复用户输入，下面这行代码可以去除它
    if response.startswith(user_input):
        response = response[len(user_input):].lstrip()
        
    print(f"模型: {response}")
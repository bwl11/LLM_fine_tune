from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载基础模型和tokenizer - 使用你训练时的基础模型
model_name = "Qwen/Qwen2-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载LoRA适配器
model = PeftModel.from_pretrained(model, "./qlora-saki-out")

# 设置tokenizer（Qwen模型需要）
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # 对于生成任务建议使用left padding

# 进行推理
text = ""
inputs = tokenizer(text, return_tensors="pt").to(model.device)  # 移动到GPU

outputs = model.generate(
    **inputs,
    max_new_tokens=3000,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
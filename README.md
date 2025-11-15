# LLM_fine_tune
local fine tuning test for llm


# 一个测试微调后模型输出的代码

run_example.py

# 典型的微调后输出模型的构成

./qlora-out/
├── checkpoint-xxx/          # 训练检查点
├── adapter_config.json      # LoRA适配器配置
├── adapter_model.bin        # LoRA权重文件（主要参数）
├── README.md               # 模型说明
└── special_tokens_map.json # 特殊token映射

# 微调后合并成完整模型的命令
python -m axolotl.cli.merge_lora ../configs/qlora.yml --lora_model_dir="./qlora-out" --output_dir="./merged-model"
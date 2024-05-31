import torch
from transformers import AutoTokenizer, BertForQuestionAnswering

# 初始化模型和 tokenizer
model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")

model_path = input("model folder path: ")

# 加載權重文件
checkpoint_path = f"{model_path}/model.ckpt"  # 替換為你的 model.ckpt 文件的路徑
model.load_state_dict(torch.load(checkpoint_path))

# 保存完整的模型（包括配置和權重）
model_save_path = f"{model_path}/"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
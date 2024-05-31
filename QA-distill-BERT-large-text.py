import os
import torch
from transformers import AutoTokenizer, DistilBertForQuestionAnswering

class COLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# 加载保存的模型和 tokenizer
model_save_path = "./model-QA-Distill-BERT/"
model = DistilBertForQuestionAnswering.from_pretrained(model_save_path)
tokenizer = AutoTokenizer.from_pretrained(model_save_path)

def answer_question(context, question):
    paragraphs = context.split("\n\n\n")  # 假设段落之间有两个换行符

    best_answer = None
    best_score = float('-inf')

    for paragraph in paragraphs:
        inputs = tokenizer.encode_plus(question, paragraph, add_special_tokens=True, return_tensors="pt")

        # 移除 token_type_ids
        inputs.pop("token_type_ids", None)

        input_ids = inputs["input_ids"].tolist()[0]

        with torch.no_grad():
            outputs = model(**inputs)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores) + 1

        # 确保答案范围合理
        answer_start = max(answer_start, 0)
        answer_end = min(answer_end, len(input_ids))

        answer_ids = input_ids[answer_start:answer_end]
        answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
        answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))

        # 计算答案的总得分
        total_score = start_scores[0, answer_start] + end_scores[0, answer_end - 1]

        if total_score > best_score:
            best_score = total_score
            best_answer = answer

    return best_answer

##############################################################################################

print("========= ALL FILES ========")
for file in os.listdir("./doc"):
    print(f"{COLORS.OKGREEN}./doc/{file}{COLORS.ENDC}")
print("============================\n")

doc_path = input(COLORS.OKBLUE+"Reading file path: "+COLORS.ENDC)
doc_content = open(doc_path, "r").read()
print(f"{doc_content[:100]}......")
print(f"......{doc_content[-100:]}")


while True:
    question = input(COLORS.OKGREEN+"Input some question (type exit to exit): "+COLORS.ENDC)
    if(question==''): break
    
    answer = answer_question(doc_content, question)
    
    print(COLORS.FAIL+"Answer: " +answer+COLORS.ENDC)

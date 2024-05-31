import torch
from transformers import AutoTokenizer, BertForQuestionAnswering

##########################################
# ! THIS IS FOR BERT, NOT DISTILL BERT ! #
##########################################


class bcolors:
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
model_save_path = "./model-BERT-full-paragraph/"
model = BertForQuestionAnswering.from_pretrained(model_save_path)
tokenizer = AutoTokenizer.from_pretrained(model_save_path)

# 定义回答问题的函数
def answer_question(context, question):
    inputs = tokenizer.encode_plus(context, question, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    with torch.no_grad():
        outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1

    answer_ids = input_ids[answer_start:answer_end]
    answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
    answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
    
    return answer

# 示例使用
# question = "What is the capital of France?"
# context = "The capital of France is Paris."

while True:
    context = input(bcolors.OKBLUE+"Input some context: "+bcolors.ENDC)
    if(context==''): break
    # print("======")

    while True:
        question = input(bcolors.OKGREEN+"Input some question (type exit to exit): "+bcolors.ENDC)
        if(question=='exit'):
            # print("======")
            break
        else:
            answer = answer_question(context, question)
        print(bcolors.FAIL+"Answer: " +answer+bcolors.ENDC)

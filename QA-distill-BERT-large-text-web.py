import re
import torch
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, DistilBertForQuestionAnswering

MODEL_SAVE_PATH = "./model-QA-Distill-BERT/"
model = DistilBertForQuestionAnswering.from_pretrained(MODEL_SAVE_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)

##############################################################################################
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

def separate_context_sliding_window(context, max_length=400, stride=200):
    """
    Separate the context into chunks using a sliding window approach to avoid exceeding the max token length.
    """
    paragraphs = context.split("\n\n")
    clean_paragraphs = []

    for paragraph in paragraphs:
        clean_paragraph = paragraph.replace('\n', ' ')
        sentences = re.split(r'(?<=[.!?])\s+', clean_paragraph)
        
        current_paragraph = ""
        current_length = 0
        chunks = []
        
        for sentence in sentences:
            sentence_length = len(tokenizer.tokenize(sentence))
            
            if current_length + sentence_length <= max_length:
                current_paragraph += " " + sentence
                current_length += sentence_length
            else:
                if current_paragraph:
                    chunks.append(current_paragraph.strip())
                current_paragraph = sentence
                current_length = sentence_length
        
        if current_paragraph:
            chunks.append(current_paragraph.strip())

        for chunk in chunks:
            tokens = tokenizer.tokenize(chunk)
            for i in range(0, len(tokens), stride):
                sub_chunk = tokens[i:i+max_length]
                clean_paragraphs.append(tokenizer.convert_tokens_to_string(sub_chunk))

    return clean_paragraphs


def answer_question(paragraphs, question):

    best_answer = None
    best_score = float('-inf')

    for paragraph in paragraphs:
        inputs = tokenizer.encode_plus(paragraph, question, add_special_tokens=True, return_tensors="pt")

        # 移除 token_type_ids
        inputs.pop("token_type_ids", None)

        input_ids = inputs["input_ids"].tolist()[0]

        with torch.no_grad():
            outputs = model(**inputs)

        # 獲取分數（張量），表示每個 token 的得分數
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # 獲取最高分數的數值
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)# + 1


        # 確保起始位置是在 max(start, 0), 
        # 確保結束位置是在 max(end, total length), 
        answer_start = max(answer_start, 0)
        answer_end = min(answer_end, len(input_ids))

        answer_ids = input_ids[answer_start:answer_end]
        answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
        answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))

        # 計算答案的總分
        total_score = start_scores[0, answer_start] + end_scores[0, answer_end - 1]

        if total_score > best_score:
            best_score = total_score
            best_answer = answer

    return best_answer

def fetch_document_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract all text within the body tag
    return soup.body.get_text(separator=' ', strip=True)

def main():
    while True:
        doc_source = input(COLORS.OKBLUE+"Enter a file path or URL: "+COLORS.ENDC)
        if(doc_source==''): break

        doc_content = ""
        if doc_source.startswith("http://") or doc_source.startswith("https://"):
            try:
                doc_content = fetch_document_from_url(doc_source)
                break
            except requests.RequestException as e:
                print(COLORS.FAIL+"Failed to fetch document from URL:"+COLORS.ENDC, e)
                continue

    paragraphs = separate_context_sliding_window(doc_content)

    while True:
        question = input(COLORS.OKGREEN+"Input some question: "+COLORS.ENDC)
        if(question==''): break
        
        answer = answer_question(paragraphs, question)
        
        print(COLORS.FAIL+"Answer: " +answer+COLORS.ENDC)

if __name__ == "__main__":
    main()

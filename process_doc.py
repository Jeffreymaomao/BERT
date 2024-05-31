import re

def separate_context(context, max_length=512):
    paragraphs = context.split("\n\n")
    clean_paragraphs = []

    for paragraph in paragraphs:
        clean_paragraph = paragraph.replace('\n', ' ')
        words = clean_paragraph.split(" ")
        
        if len(words) > max_length:
            # 如果单词数量超过 max_length，则拆分段落
            for i in range(0, len(words), max_length):
                sub_paragraph = " ".join(words[i:i + max_length])
                clean_paragraphs.append(sub_paragraph)
        else:
            clean_paragraphs.append(clean_paragraph)

    return clean_paragraphs

def approx_length_of_paragraph(paragraph):
    return len(paragraph.split())

def separate_context(context, max_length=512):
    paragraphs = context.split("\n\n")
    clean_paragraphs = []

    for paragraph in paragraphs:
        clean_paragraph = paragraph.replace('\n', ' ')
        sentences = re.split(r'(?<=[.!?])\s+', clean_paragraph)
        current_paragraph = ""

        for sentence in sentences:
            if approx_length_of_paragraph(current_paragraph + " " + sentence) <= max_length:
                current_paragraph += " " + sentence
            else:
                if current_paragraph:
                    clean_paragraphs.append(current_paragraph.strip())
                current_paragraph = sentence

        if current_paragraph:
            clean_paragraphs.append(current_paragraph.strip())

    return clean_paragraphs

doc_path = "./doc/Einstein's-equation.wiki.txt"
doc_content = open(doc_path, "r").read()

process_paragraphs = separate_context(doc_content)

f = open("./doc/Einstein's-equation.wiki.process.txt", "w")

for process_paragraph in process_paragraphs:
    words = process_paragraph.split(" ")
    # print(f"---{len(words)}", file=f)
    print("", file=f)
    print(process_paragraph, file=f)
    

f.close()
from data_clean import text_list as text
from data_process import doc_process

docs = doc_process(text)

for i in docs :
    print(i)
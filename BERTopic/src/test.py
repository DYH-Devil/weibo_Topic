#encoding=GBK
import numpy as np
import pandas as pd
from BERTopic.src.data_process import data_process
from BERTopic.src.cluster import ump_sentence_res
from BERTopic.src.c_tfidf_calc import topic_size

docs = data_process()

#构建docs_df DataFrame
docs_df = pd.DataFrame(docs ,columns = ['Doc'])
docs_df['Topic'] = ump_sentence_res['label']
docs_df['Doc_ID'] = range(len(docs_df))
print(docs_df)

#构建topic_size Dataframe
t_size = topic_size(docs_df)
print(t_size)

topic_merge = t_size.loc[t_size['Size'] < 100].Topic
print(topic_merge)
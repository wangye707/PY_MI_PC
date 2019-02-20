import os
data_dir='./title'
docid_doc='docid_doc.txt'
embed_wiki_pdc_d50='embed_wiki_pdc_d50'
embed_wiki_pdc_d50_norm='embed_wiki_pdc_d50_norm'
qid_query='qid_query.txt'
#batch_size=store_batch_parm.batch_size
word_dict='word_dict.txt'

def read_embedding_vec(data_dir,embed_wiki_pdc_d50_norm):
    file_name=os.path.join(data_dir,embed_wiki_pdc_d50_norm)
    with open(file_name,'r') as f:
        lines=f.readlines()
        words_vec=dict()
        for line in lines:
            vec_split=line.split()
            temp_word_vec=[]
#            for word_split,i in word_split:
            for i,vec in enumerate(vec_split):
                if i>0:
                    temp_word_vec.append(vec)
#            change the type of  temp_word_id to float64
            temp_word_id_int=temp_word_vec
#            print('word_vec_name:',vec_split[0])
            words_vec[float(vec_split[0])]=temp_word_id_int
#        print('this is words_vec length:',len(words_vec))
        #this is words_vec length: 96379
    return words_vec

def read_word_by_qid(data_dir,qid_query):
    file_name=os.path.join(data_dir,qid_query)
    with open(file_name,'r') as f:
        lines=f.readlines()
        qid_words=dict()
        for line in lines:
            word_split=line.split()
            temp_word_id=[]
#            for word_split,i in word_split:
            for i,word_id in enumerate(word_split):
                if i>1:
                    temp_word_id.append(int(word_id))
#            change the type of  temp_word_id to float64
            temp_word_id_int=temp_word_id
            qid_words[float(word_split[0])]=temp_word_id_int
    return qid_words

embed=read_embedding_vec(data_dir,embed_wiki_pdc_d50)
if embed.get(3) is None:
	print('I am None!')

#word_qid=read_word_by_qid(data_dir,qid_query)
#print('word_qid:',word_qid[float(668)])
#for i in word_qid[668]:
#	print(i)

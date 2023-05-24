import json
from my_lib.util.eval.translate_metric import get_nltk33_sent_bleu1 as get_sent_bleu1, \
                                              get_nltk33_sent_bleu2 as get_sent_bleu2,  \
                                            get_nltk33_sent_bleu3 as get_sent_bleu3,  \
                                            get_nltk33_sent_bleu4 as get_sent_bleu4,  \
                                            get_nltk33_sent_bleu as get_sent_bleu
from my_lib.util.eval.translate_metric import get_corp_bleu1,get_corp_bleu2,get_corp_bleu3,get_corp_bleu4,get_corp_bleu
from my_lib.util.eval.translate_metric import get_meteor,get_rouge,get_cider
from config import keep_test_data_id_path,res_path

keep_test_data_ids=[]
with open(keep_test_data_id_path,'r') as f:
    for line in f:
        keep_test_data_ids.append(int(line.strip()))

# res_path='../../../data/python1/result/result_26.json'

with open(res_path,'r') as f:
    res_data=json.load(f)
gold_texts=[]
pred_texts=[]
# sblues=[]
for i,item in enumerate(res_data):
    if i in keep_test_data_ids:
        assert item['id']==i+1
        gold_text=item['gold_text'].split()
        pred_text=item['pred_text'].split()
        gold_texts.append([gold_text])
        pred_texts.append(pred_text)

print(len(pred_texts),len(gold_texts))
print('The performance on the cleand Java testing set is:')
print(get_meteor.__name__,':',get_meteor(pred_texts,gold_texts))
print(get_rouge.__name__,':',get_rouge(pred_texts,gold_texts))
print(get_sent_bleu1.__name__,':',get_sent_bleu1(pred_texts,gold_texts))
print(get_sent_bleu2.__name__,':',get_sent_bleu2(pred_texts,gold_texts))
print(get_sent_bleu3.__name__,':',get_sent_bleu3(pred_texts,gold_texts))
print(get_sent_bleu4.__name__,':',get_sent_bleu4(pred_texts,gold_texts))
print(get_sent_bleu.__name__,':',get_sent_bleu(pred_texts,gold_texts))
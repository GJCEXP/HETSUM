#coding=utf-8
import logging
import os
import sys
from my_lib.util.eval.translate_metric import get_nltk33_sent_bleu1 as get_sent_bleu1, \
                                              get_nltk33_sent_bleu2 as get_sent_bleu2,  \
                                            get_nltk33_sent_bleu3 as get_sent_bleu3,  \
                                            get_nltk33_sent_bleu4 as get_sent_bleu4,  \
                                            get_nltk33_sent_bleu as get_sent_bleu
from my_lib.util.eval.translate_metric import get_corp_bleu1,get_corp_bleu2,get_corp_bleu3,get_corp_bleu4,get_corp_bleu
from my_lib.util.eval.translate_metric import get_meteor,get_rouge,get_cider
import math
# from config_py27 import *

train_data_name='train_data'
valid_data_name='valid_data'
test_data_name='test_data'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#顶级数据目录
top_data_dir= '../../../data/python_GypSum'

raw_data_dir=os.path.join(top_data_dir,'raw_data/')
train_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(train_data_name))
valid_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(valid_data_name))
test_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(test_data_name))
tech_term_path=os.path.join(raw_data_dir,'tech_term.txt')
keep_test_data_id_path=os.path.join(raw_data_dir,'keep_test_data_ids.txt')

max_code_len=248     #285
# max_code_str_len=10 #22
max_ast_size=412    #355
# max_ast_func_size=317   #209
# max_ast_attr_size=165      #147
max_text_len=19       #22

token_data_dir=os.path.join(top_data_dir,'token_data/')
train_token_data_path=os.path.join(token_data_dir,'{}.json'.format(train_data_name))
valid_token_data_path=os.path.join(token_data_dir,'{}.json'.format(valid_data_name))
test_token_data_path=os.path.join(token_data_dir,'{}.json'.format(test_data_name))

USER_WORDS=[('\\','n'),('e','.','g','.'),('i','.','e','.'),('-','>')]

basic_info_dir=os.path.join(top_data_dir,'basic_info/')
size_info_path=os.path.join(basic_info_dir,'size_info.pkl')
rev_dic_path=os.path.join(basic_info_dir,'rev_dic.json')
noise_token_path=os.path.join(basic_info_dir,'noise_token.json')
size_info_pdf_path=os.path.join(basic_info_dir,'dist_of_code_ast_and_text_size.pdf')
size_info_png_path=os.path.join(basic_info_dir,'dist_of_code_ast_and_text_size.png')

w2i2w_dir=os.path.join(top_data_dir,'w2i2w/')
io_token_w2i_path=os.path.join(w2i2w_dir,'io_token_w2i.pkl')
io_token_i2w_path=os.path.join(w2i2w_dir,'io_token_i2w.pkl')
code_pos_w2i_path=os.path.join(w2i2w_dir,'code_pos_w2i.pkl')
code_pos_i2w_path=os.path.join(w2i2w_dir,'code_pos_i2w.pkl')

io_min_token_count=3
unk_aliased=True  #是否将未知的rare tokens进行标号处理

avail_data_dir=os.path.join(top_data_dir,'avail_data/')
train_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(train_data_name))
valid_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(valid_data_name))
test_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(test_data_name))
# io_token_sim_id_path=os.path.join(avail_data_dir,'io_token_sim_ids.npy')

OUT_BEGIN_TOKEN='</s>'
OUT_END_TOKEN='</e>'
PAD_TOKEN='<pad>'
UNK_TOKEN='<unk>'

model_dir=os.path.join(top_data_dir,'model/')
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3" #"5,6,7,8,9","0,1,2,3,4"
import os
from torch_geometric.nn import SAGEConv, GCNConv, GATConv,TransformerConv
#
emb_dims = 512  #################### 512
ast_gnn_layers=6    #################### 6
code_att_layers=2   #################### 2
text_att_layers=6   #################### 6
train_batch_size=192+32    #################### 192
version='39b11'  ####################
model_name='codescriber_v{}_{}_{}_{}_{}'.format(version,ast_gnn_layers,code_att_layers,text_att_layers,emb_dims)
params = dict(model_dir=model_dir,
              model_name=model_name,
              model_id=None,
              emb_dims=emb_dims,
              ast_gnn_layers=ast_gnn_layers, #############2080*10
              ast_GNN=SAGEConv,
              ast_gnn_aggr='mean',
              code_att_layers=code_att_layers,
              code_att_heads=8,
              code_att_head_dims=None,
              code_ff_hid_dims=4 * emb_dims,
              text_att_layers=text_att_layers,
              text_att_heads=8,
              text_att_head_dims=None,
              text_ff_hid_dims=4 * emb_dims,
              drop_rate=0.2,
              copy=True,
              pad_idx=0,
              train_batch_size=train_batch_size,
              pred_batch_size=math.ceil(train_batch_size * 1.5),  #################### 2
              max_train_size=-1,  #################### -1 means all
              max_valid_size=-1,  #################### 20 math.ceil(train_batch_size * 1.5) * 20
              max_big_epochs=100,  #################### 100
              early_stop=10,
              regular_rate=1e-5,
              lr_base=5e-4,
              lr_decay=0.95,
              min_lr_rate=0.01,
              warm_big_epochs=3,
              beam_width=5,
              start_valid_epoch=60, #################### 50
              gpu_ids=os.environ["CUDA_VISIBLE_DEVICES"],
              train_mode=True)
# from tmp_google_bleu import get_sent_bleu
train_metrics = [get_sent_bleu]
valid_metric = get_sent_bleu
test_metrics = [get_rouge, get_cider,get_meteor,
                get_sent_bleu1,get_sent_bleu2,get_sent_bleu3,get_sent_bleu4,get_sent_bleu,
                get_corp_bleu1,get_corp_bleu2,get_corp_bleu3,get_corp_bleu4,get_corp_bleu] #[get_corp_bleu]

#the path of result in practical prediction
res_dir=os.path.join(top_data_dir,'result/')
res_path=os.path.join(res_dir,model_name+'.json')

import random
import torch
import numpy as np
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)    # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)    # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True
seeds=[0,42,7,23,124,1084,87]
seed_torch(seeds[1])
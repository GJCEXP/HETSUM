# coding=utf-8
import os
import re
import sys
from my_lib.neural_module.learn_strategy import LrWarmUp
from my_lib.neural_module.transformer import TranEnc, TranDec, DualTranDec,ResFF,ResMHA
from my_lib.neural_module.embedding import PosEnc
from my_lib.neural_module.loss import LabelSmoothSoftmaxCEV2, CriterionNet
from my_lib.neural_module.balanced_data_parallel import BalancedDataParallel
from my_lib.neural_module.copy_attention import DualMultiCopyGenerator,MultiCopyGenerator,DualCopyGenerator
from my_lib.neural_module.beam_search import trans_beam_search
from my_lib.neural_model.seq_to_seq_model import TransSeq2Seq
from my_lib.neural_model.base_model import BaseNet
from my_lib.neural_module.transformer import ResFF
from typing import Any,Optional,Union

from config import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader.data_list_loader import DataListLoader
from torch_geometric.utils import to_dense_batch
from torch_geometric.data.storage import (BaseStorage, NodeStorage,EdgeStorage)
from torch_geometric.nn.data_parallel import DataParallel
from torch_geometric.nn import HeteroConv,GraphNorm
import random
import numpy as np
import os
import logging
import pickle
import json
import codecs
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import math
from copy import deepcopy

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]

class Datax(HeteroData):
    # def __init__(self,
    #              ast_func_node=dict(x=None),
    #              ast_attr_node=dict(x=None),
    #              ast_glob_node=dict(x=None),
    #              ):
    #     super().__init__()
    #     self.ast_func_node=ast_func_node
    def __cat_dim__(self, key: str, value: Any,
                    store: Optional[NodeOrEdgeStorage] = None, *args,
                    **kwargs) -> Any:
        if bool(re.search('(token)', key)): #|map
            return None  # generate a new 0 dimension
        if bool(re.search('(pos)', key)):
            return -1
        return super().__cat_dim__(key, value,store)    #return不能漏了！！！

    # def __inc__(self, key: str, value: Any,
    #             store: Optional[NodeOrEdgeStorage] = None, *args,
    #             **kwargs) -> Any:
    #     if 'index' in key:
    #         print(store.size())
    #         return torch.tensor(store.size()).view(2, 1)
    #     else:
    #         return 0

    # @property
    # def num_nodes(self) -> Optional[int]:
    #     r"""Returns the number of nodes in the graph."""
    #     return sum([value.size(0) for value in self.x_dict.values()])
        # return super().num_nodes

class Datasetx(Dataset):
    '''
    文本对数据集对象（根据具体数据再修改）
    '''
    def __init__(self,
                 code_asts,
                 texts=None,
                 ids=None,
                 text_max_len=None,
                 text_begin_idx=1,
                 text_end_idx=2,
                 pad_idx=0):
        self.len = len(code_asts)  # 样本个数
        self.text_max_len = text_max_len
        self.text_begin_idx = text_begin_idx
        self.text_end_idx = text_end_idx

        # if code_max_len is None:
        #     self.code_max_len = max([len(item['code']['tokens']) for item in code_asts])
        if text_max_len is None and texts is not None:
            self.text_max_len = max([len(text) for text in texts])  # 每个输出只是一个序列
        self.code_asts = code_asts
        self.texts = texts
        self.ids = ids
        self.pad_idx = pad_idx

    def __getitem__(self, index):
        if self.texts is None:
            pad_text_in = np.zeros((self.text_max_len + 1,), dtype=np.int64)  # decoder端的输入
            pad_text_in[0] = self.text_begin_idx
            pad_text_out = None
        else:
            tru_text = self.texts[index][:self.text_max_len]  # 先做截断
            pad_text_in = np.lib.pad(tru_text,
                                    (1, self.text_max_len - len(tru_text)),
                                    'constant',
                                    constant_values=(self.text_begin_idx, self.pad_idx))
            tru_text_out = np.lib.pad(tru_text,
                                     (0, 1),
                                     'constant',
                                     constant_values=(0, self.text_end_idx))  # padding
            pad_text_out = np.lib.pad(tru_text_out,
                                     (0, self.text_max_len + 1 - len(tru_text_out)),
                                     'constant',
                                     constant_values=(self.pad_idx, self.pad_idx))  # padding
            # pad_out_input=np.lib.pad(pad_out[:-1],(1,0),'constant',constant_values=(self.text_begin_idx, 0))
        data=Datax()
        data['node'].x=torch.tensor(self.code_asts[index]['nodes'])
        data['node'].src_map=torch.tensor(self.code_asts[index]['node2text_map_ids']).long()
        data['node'].code_pos = torch.tensor(self.code_asts[index]['node_in_code_poses'])
        data['node'].code_mask=torch.tensor(self.code_asts[index]['leaf_node_mask']).bool()
        data['node','child','node'].edge_index=torch.tensor(self.code_asts[index]['node_child_node_edges']).long()
        data['node','parent','node'].edge_index=torch.tensor(self.code_asts[index]['node_parent_node_edges']).long()
        data['node','sibling_next','node'].edge_index=torch.tensor(self.code_asts[index]['sibling_next_sibling_edges']).long()
        data['node','sibling_prev','node'].edge_index=torch.tensor(self.code_asts[index]['sibling_prev_sibling_edges']).long()
        data['node','dfg_next','node'].edge_index=torch.tensor(self.code_asts[index]['dfg_next_dfg_edges']).long()
        data['node','dfg_prev','node'].edge_index=torch.tensor(self.code_asts[index]['dfg_prev_dfg_edges']).long()
        data['text'].text_token_input=torch.tensor(pad_text_in).long()
        if self.texts is not None:
            data['text'].text_token_output = torch.tensor(pad_text_out).long()
        data['text'].num_nodes = pad_text_in.shape[0]
        if self.ids is not None:
            data['idx'].idx=torch.tensor(self.ids[index])
            data['idx'].num_nodes=1
        # print(data.num_nodes)
        return data

    def __len__(self):
        return self.len

class CodeASTEnc(nn.Module):
    def __init__(self,
                 emb_dims,
                 ast_max_size,
                code_max_len,
                ast_node_emb_op,
                 code_mpos_voc_size,
                 code_npos_voc_size,
                 code_att_layers=2,
                 code_att_heads=8,
                 code_att_head_dims=None,
                 code_ff_hid_dims=2048,
                 ast_gnn_layers=6,
                 ast_GNN=SAGEConv,
                 ast_gnn_aggr='mean',
                 drop_rate=0.,
                 **kwargs,
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.pad_idx = kwargs['pad_idx']
        self.ast_max_size = ast_max_size
        self.code_max_len=code_max_len
        self.emb_dims=emb_dims

        # assert len(ast_sim_node_ids.shape)==1
        # ast_sim_node_voc_size=np.unique(ast_sim_node_ids).shape[0]
        # self.ast_node_to_sim_token_map_op=nn.Embedding.from_pretrained(torch.tensor(ast_sim_node_ids).view([-1,1]).float(),freeze=True,padding_idx=kwargs['pad_idx'])
        self.ast_node_emb_op = ast_node_emb_op
        # self.ast_node_to_sim_token_map_op=ast_node_to_sim_token_map_op
        # self.ast_node_emb_op = nn.Embedding(ast_node_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
        # self.sim_node_emb_op = nn.Embedding(ast_sim_node_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
        self.code_mpos_emb_op = nn.Embedding(code_mpos_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
        self.code_npos_emb_op = nn.Embedding(code_npos_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
        # nn.init.xavier_uniform_(self.ast_node_emb_op.weight[1:, ])
        # nn.init.xavier_uniform_(self.ast_sim_node_emb_op.weight[1:, ])
        nn.init.xavier_uniform_(self.code_mpos_emb_op.weight[1:, ])
        nn.init.xavier_uniform_(self.code_npos_emb_op.weight[1:, ])

        # self.ast_emb_norm_op = nn.LayerNorm(emb_dims)
        self.emb_drop_op = nn.Dropout(p=drop_rate)
        self.code_emb_norm_op = nn.LayerNorm(emb_dims)
        # self.ast_emb_norm_op = nn.LayerNorm(emb_dims)

        self.code_enc_op = TranEnc(query_dims=emb_dims,
                                    head_num=code_att_heads,
                                    ff_hid_dims=code_ff_hid_dims,
                                    head_dims=code_att_head_dims,
                                    layer_num=code_att_layers,
                                    drop_rate=drop_rate,
                                    pad_idx=kwargs['pad_idx'])

        self.gnn_layers = ast_gnn_layers
        self.gnn_ops=nn.ModuleList()
        self.gnorm_ops=nn.ModuleList()
        self.grelu_ops=nn.ModuleList()
        for _ in range(ast_gnn_layers):
            if ast_GNN==TransformerConv:
                gnn=HeteroConv({
                    ('node', 'child', 'node'):TransformerConv((emb_dims,emb_dims), out_channels=emb_dims//code_att_heads,heads=code_att_heads, aggr=ast_gnn_aggr,dropout=drop_rate,root_weight=True),
                    ('node', 'parent', 'node'):TransformerConv((emb_dims,emb_dims), out_channels=emb_dims//code_att_heads,heads=code_att_heads, aggr=ast_gnn_aggr,dropout=drop_rate,root_weight=False),
                    ('node', 'sibling_next', 'node'):TransformerConv((emb_dims,emb_dims), out_channels=emb_dims//code_att_heads,heads=code_att_heads, aggr=ast_gnn_aggr,dropout=drop_rate,root_weight=False),
                    ('node', 'sibling_prev', 'node'):TransformerConv((emb_dims,emb_dims), out_channels=emb_dims//code_att_heads,heads=code_att_heads, aggr=ast_gnn_aggr,dropout=drop_rate,root_weight=False),
                    ('node', 'dfg_next', 'node'):TransformerConv((emb_dims,emb_dims), out_channels=emb_dims//code_att_heads,heads=code_att_heads, aggr=ast_gnn_aggr,dropout=drop_rate,root_weight=False),
                    ('node', 'dfg_prev', 'node'):TransformerConv((emb_dims,emb_dims), out_channels=emb_dims//code_att_heads,heads=code_att_heads, aggr=ast_gnn_aggr,dropout=drop_rate,root_weight=False),
                },aggr='sum')
            elif ast_GNN==SAGEConv:
                gnn=HeteroConv({
                    ('node', 'child', 'node'): ast_GNN((emb_dims,emb_dims), emb_dims, aggr=ast_gnn_aggr,root_weight=True),
                    ('node', 'parent', 'node'): ast_GNN((emb_dims,emb_dims), emb_dims, aggr=ast_gnn_aggr,root_weight=False),
                    ('node', 'sibling_next', 'node'): ast_GNN((emb_dims,emb_dims), emb_dims, aggr=ast_gnn_aggr,root_weight=False),
                    ('node', 'sibling_prev', 'node'): ast_GNN((emb_dims,emb_dims), emb_dims, aggr=ast_gnn_aggr,root_weight=False),
                    ('node', 'dfg_next', 'node'): ast_GNN((emb_dims,emb_dims), emb_dims, aggr=ast_gnn_aggr,root_weight=False),
                    ('node', 'dfg_prev', 'node'): ast_GNN((emb_dims,emb_dims), emb_dims, aggr=ast_gnn_aggr,root_weight=False),
                },aggr='sum')
            else:
                gnn=HeteroConv({
                    ('node', 'child', 'node'): ast_GNN(emb_dims, emb_dims, aggr=ast_gnn_aggr),
                    ('node', 'parent', 'node'): ast_GNN(emb_dims, emb_dims, aggr=ast_gnn_aggr),
                    ('node', 'sibling_next', 'node'): ast_GNN(emb_dims, emb_dims, aggr=ast_gnn_aggr),
                    ('node', 'sibling_prev', 'node'): ast_GNN(emb_dims, emb_dims, aggr=ast_gnn_aggr),
                    ('node', 'dfg_next', 'node'): ast_GNN(emb_dims, emb_dims, aggr=ast_gnn_aggr),
                    ('node', 'dfg_prev', 'node'): ast_GNN(emb_dims, emb_dims, aggr=ast_gnn_aggr),
                },aggr='sum')
            self.gnn_ops.append(gnn)
            self.grelu_ops.append(nn.Sequential(nn.ReLU(), nn.Dropout(p=drop_rate)))
            self.gnorm_ops.append(nn.LayerNorm(emb_dims))

    def forward(self, data):
        assert len(data['node'].x.size()) == 1  #[batch_ast_node_num,]
        assert len(data['node'].src_map.size())==1 #[batch_ast_node_num,]
        assert len(data['node'].code_pos.size())==2 # [2(m,n),batch_ast_node_num]
        assert len(data['node'].code_mask.size())==1 #[batch_ast_node_num,]
        assert len(data.edge_index_dict[('node','child','node')].size()) == 2  # 点是一堆节点序号[2,batch_xx_edge_num]
        assert len(data.edge_index_dict[('node','parent','node')].size()) == 2  # 点是一堆节点序号[2,batch_xx_edge_num]
        assert len(data.edge_index_dict[('node','sibling_prev','node')].size()) == 2  # 点是一堆节点序号[2,batch_xx_edge_num]
        assert len(data.edge_index_dict[('node','sibling_next','node')].size()) == 2  # 点是一堆节点序号[2,batch_xx_edge_num]
        assert len(data.edge_index_dict[('node','dfg_prev','node')].size()) == 2  # 点是一堆节点序号[2,batch_xx_edge_num]
        assert len(data.edge_index_dict[('node','dfg_next','node')].size()) == 2  # 点是一堆节点序号[2,batch_xx_edge_num]

        #encode the source code
        # sim_node_x=self.ast_node_to_sim_token_map_op(data.x_dict['node']).squeeze(-1).long()  #[batch_ast_node_num,]
        # sim_node_mask=sim_node_x.abs().sign()   #[batch_ast_node_num,]
        # sim_node_emb=self.ast_sim_node_emb_op(sim_node_x[sim_node_mask==True])   ##[batch_part_ast_node_num,emb_dims]

        ast_node_emb=self.ast_node_emb_op(data.x_dict['node'])  ##[batch_ast_node_num,emb_dims]
        # ast_node_emb[sim_node_mask==True,:]=ast_node_emb[sim_node_mask==True,:].add(sim_node_emb)*0.5
        data['node'].x=self.emb_drop_op(ast_node_emb) ##[batch_ast_node_num,emb_dims]

        code_emb=data['node'].x[data['node'].code_mask==True,:]* np.sqrt(self.emb_dims) ##[batch_leaf_node_num,emb_dims]
        # data['node'].x=self.ast_emb_norm_op(data['node'].x) ##[batch_ast_node_num,emb_dims]
        code_mpos_emb=self.code_mpos_emb_op(data['node'].code_pos[0,:][data['node'].code_mask==True])     #[batch_leaf_node_num,emb_dims]
        code_npos_emb=self.code_npos_emb_op(data['node'].code_pos[1,:][data['node'].code_mask==True])     #[batch_leaf_node_num,emb_dims]
        code_pos_emb=self.emb_drop_op(code_mpos_emb.add(code_npos_emb)) #[batch_leaf_node_num,emb_dims]

        code_x_batch=data.x_batch_dict['node'][data['node'].code_mask==True]    #[batch_leaf_node_num,]
        
        code_emb,code_mask=to_dense_batch(code_emb,
                                        batch=code_x_batch,
                                        fill_value=self.pad_idx,
                                        max_num_nodes=self.code_max_len)    #[batch_size,code_max_len,emb_dims],[batch_size,code_max_len]
        code_pos_emb,_=to_dense_batch(code_pos_emb,
                                        batch=code_x_batch,
                                        fill_value=self.pad_idx,
                                        max_num_nodes=self.code_max_len)    #[batch_size,code_max_len,emb_dims],[batch_size,code_max_len]
        code_emb=self.code_emb_norm_op(code_emb.add(code_pos_emb))   #[batch_size,code_max_len,emb_dims]
        code_enc=self.code_enc_op(query=code_emb,query_mask=code_mask)  # [batch_data_num,code_max_len,emb_dims]
        sparse_code_enc=code_enc.contiguous().view(-1,code_enc.size(-1))[code_mask.view(-1)==True,:] ###[batch_leaf_node_num,emb_dims] convert dense batch into sparse batch
        data['node'].x[data['node'].code_mask==True,:]=data['node'].x[data['node'].code_mask==True,:].add(sparse_code_enc)  #[batch_leaf_node_num,emb_dims]
        
        
        # =code_emb
        # ast_node_emb=data['node'].x.clone()
        for gnn,relu,norm in zip(self.gnn_ops,self.grelu_ops,self.gnorm_ops):
            x_dict=gnn(x_dict=data.x_dict,edge_index_dict=data.edge_index_dict)   # dict(xx_node:[batch_xx_node_num,hid_dims])
            data['node'].x=norm(data['node'].x.add(relu(x_dict['node']))) #data[key].x residual connection
            # data['node'].x=norm(ast_node_emb.add(relu(x_dict['node']))) #data[key].x residual connection
            
        ast_enc,_=to_dense_batch(data.x_dict['node'],
                                  batch=data.x_batch_dict['node'], #data['leaf'].x_batch也可以
                                  fill_value=self.pad_idx,
                                  max_num_nodes=self.ast_max_size)  #[batch_size,ast_max_size,emb_dims],[batch_size,ast_max_size]

        code_src_map,_=to_dense_batch(data.src_map_dict['node'][data['node'].code_mask==True],
                                    batch=code_x_batch,  # data['leaf'].x_batch也可以
                                    fill_value=self.pad_idx,
                                    max_num_nodes=self.code_max_len)    # [batch_data_num,code_max_len]
        ast_code_enc,_=to_dense_batch(data.x_dict['node'][data['node'].code_mask==True],
                            batch=code_x_batch,  # data['leaf'].x_batch也可以
                            fill_value=self.pad_idx,
                            max_num_nodes=self.code_max_len)    # [batch_data_num,code_max_len]

        return ast_enc,ast_code_enc,code_enc,code_src_map

class Dec(nn.Module):
    def __init__(self,
                 emb_dims,
                 text_voc_size,
                 text_emb_op,
                 text_max_len,
                 enc_out_dims,
                 att_layers,
                 att_heads,
                 att_head_dims=None,
                 ff_hid_dims=2048,
                 drop_rate=0.,
                 **kwargs
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        kwargs.setdefault('copy', True)
        self._copy = kwargs['copy']
        self.emb_dims = emb_dims
        self.text_voc_size = text_voc_size
        # embedding dims为text_voc_size+2*code_max_len

        # assert len(text_sim_token_ids.shape)==1
        # text_sim_token_voc_size=np.unique(text_sim_token_ids).shape[0]
        # self.text_token_to_sim_token_map_op=nn.Embedding.from_pretrained(torch.tensor(text_sim_token_ids).view([-1,1]).float(),freeze=True,padding_idx=kwargs['pad_idx'])
        # self.text_token_to_sim_token_map_op=text_token_to_sim_token_map_op
        self.text_emb_op = text_emb_op
        # self.text_emb_op = nn.Embedding(text_voc_size + code_max_len, emb_dims, padding_idx=kwargs['pad_idx'])
        # self.sim_token_emb_op = nn.Embedding(text_sim_token_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
        # nn.init.xavier_uniform_(self.text_emb_op.weight[1:, ])
        # nn.init.xavier_uniform_(self.sim_token_emb_op.weight[1:, ])
        self.pos_encoding = PosEnc(max_len=text_max_len+1, emb_dims=emb_dims, train=True, pad=True,pad_idx=kwargs['pad_idx'])  #不要忘了+1,因为输入前加了begin_id
        # nn.init.xavier_uniform_(self.pos_encoding.weight[1:, ])
        self.emb_layer_norm = nn.LayerNorm(emb_dims)
        # self.text_dec_op = TranDec(query_dims=emb_dims,
        #                            key_dims=enc_out_dims,
        #                            head_nums=att_heads,
        #                            head_dims=att_head_dims,
        #                            layer_num=att_layers,
        #                            ff_hid_dims=ff_hid_dims,
        #                            drop_rate=drop_rate,
        #                            pad_idx=kwargs['pad_idx'],
        #                            self_causality=True)
        self.text_dec_op = DualTranDec(query_dims=emb_dims,
                                    key_dims=enc_out_dims,
                                    head_num=att_heads,
                                    ff_hid_dims=ff_hid_dims,
                                    head_dims=att_head_dims,
                                    layer_num=att_layers,
                                    drop_rate=drop_rate,
                                    pad_idx=kwargs['pad_idx'],
                                    mode='sequential',
                                    self_causality=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.out_fc = nn.Linear(emb_dims, text_voc_size)
        self.copy_generator = DualMultiCopyGenerator(tgt_dims=emb_dims,
                                                     tgt_voc_size=text_voc_size,
                                                     src_dims=enc_out_dims,
                                                     att_heads=att_heads,
                                                     att_head_dims=att_head_dims,
                                                     drop_rate=drop_rate,
                                                     pad_idx=kwargs['pad_idx'])

    def forward(self,ast_enc,ast_code_enc,code_enc,code_src_map,text_input):
        # sim_text_token=self.text_token_to_sim_token_map_op(text_input).squeeze(-1).long()  #[batch_text,L_text]
        # sim_token_mask=sim_text_token.abs().sign()   #[batch_text,L_text]
        # sim_token_emb=self.text_sim_token_emb_op(sim_text_token[sim_token_mask==True])   # (B*L_text,D_text_emb)  .view(sim_text_token.size())
        
        text_emb = self.text_emb_op(text_input)   # (B,L_text,D_text_emb)
        # text_emb[sim_token_mask==True,:]=text_emb[sim_token_mask==True,:].add(sim_token_emb)*0.5
        text_emb=text_emb* np.sqrt(self.emb_dims)
        pos_emb = self.pos_encoding(text_input)  # # (B,L_text,D_emb)
        text_dec = self.dropout(text_emb.add(pos_emb))  # (B,L_text,D_emb)
        text_dec = self.emb_layer_norm(text_dec)  # (B,L_text,D_emb)

        ast_mask = ast_enc.abs().sum(-1).sign()  # (batch_size,ast_max_size)
        code_mask=code_enc.abs().sum(-1).sign() # (batch_size,code_max_len)
        text_mask = text_input.abs().sign()  # (B,L_text)
        text_dec = self.text_dec_op(query=text_dec,
                                    key1=ast_enc,
                                    key2=code_enc,
                                    query_mask=text_mask,
                                    key_mask1=ast_mask,
                                    key_mask2=code_mask
                                    )  # (B,L_text,D_text_emb)

        if not self._copy:
            text_output = self.out_fc(text_dec)  # (B,L_text,text_voc_size)包含begin_idx和end_idx
            # text_output = F.softmax(text_output, dim=-1)
            # text_output[:,:,-1]=0.    #不生成begin_idx，默认该位在text_voc_size最后一个，置0
        else:
            # text_output=F.pad(text_output,(0,2*self.text_max_len)) #pad last dim
            text_output = self.copy_generator(text_dec,
                                             ast_code_enc,code_src_map,
                                             code_enc,code_src_map)
        # text_output[:, :, self.text_voc_size - 1] = 0.  # 不生成begin_idx，默认该位在text_voc_size最后一个，置0
        # text_output[:, :, 0] = 0.  # pad位不生成
        return text_output.transpose(1, 2)

class TNet(BaseNet):
    def __init__(self,
                 emb_dims,
                 ast_max_size,
                 code_max_len,
                 text_max_len,
                #  sim_token_ids,
                 io_voc_size,
                 code_mpos_voc_size,
                 code_npos_voc_size,
                 text_voc_size,
                 code_att_layers=2,
                 code_att_heads=8,
                 code_att_head_dims=None,
                 code_ff_hid_dims=2048,
                 ast_gnn_layers=6,
                 ast_GNN=SAGEConv,
                 ast_gnn_aggr='add',
                 text_att_layers=3,
                 text_att_heads=8,
                 text_att_head_dims=None,
                 text_ff_hid_dims=2048,
                 drop_rate=0.,
                 **kwargs,
                 ):
        super().__init__()
        kwargs.setdefault('copy', True)
        kwargs.setdefault('pad_idx', 0)  # GraphData.batch to_dense_data用的
        self.init_params = locals()
        io_token_emb_op=nn.Embedding(io_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
        nn.init.xavier_uniform_(io_token_emb_op.weight[1:, ])
        # assert len(sim_token_ids.shape)==1
        # print(np.unique(sim_token_ids).shape[0],np.unique(sim_token_ids).max()+1)
        # sim_token_voc_size=np.unique(sim_token_ids).shape[0]
        # assert np.unique(sim_token_ids).shape[0]==np.unique(sim_token_ids).max()+1
        # io_token_to_sim_token_map_op=nn.Embedding.from_pretrained(torch.tensor(sim_token_ids).view([-1,1]).float(),freeze=True,padding_idx=kwargs['pad_idx'])
        # sim_token_emb_op = nn.Embedding(np.unique(sim_token_ids).shape[0], emb_dims, padding_idx=kwargs['pad_idx'])
        # nn.init.xavier_uniform_(sim_token_emb_op.weight[1:, ])
        self.enc_op = CodeASTEnc(emb_dims=emb_dims,
                                ast_max_size=ast_max_size,
                                code_max_len=code_max_len,
                                # ast_node_voc_size=ast_node_voc_size,
                                ast_node_emb_op=io_token_emb_op,
                                # ast_node_to_sim_token_map_op=io_token_to_sim_token_map_op,
                                code_mpos_voc_size=code_mpos_voc_size,
                                code_npos_voc_size=code_npos_voc_size,
                                code_att_layers=code_att_layers,
                                code_att_heads=code_att_heads,
                                code_att_head_dims=code_att_head_dims,
                                code_ff_hid_dims=code_ff_hid_dims,
                                ast_gnn_layers=ast_gnn_layers,
                                ast_GNN=ast_GNN,
                                ast_gnn_aggr=ast_gnn_aggr,
                                drop_rate=drop_rate,
                                pad_idx=kwargs['pad_idx'])
        self.dec_op = Dec(emb_dims=emb_dims,
                            text_voc_size=text_voc_size,
                            text_max_len=text_max_len,
                            # code_max_len=code_max_len,
                            text_emb_op=io_token_emb_op,
                            # text_token_to_sim_token_map_op=io_token_to_sim_token_map_op,
                            enc_out_dims=emb_dims,
                            att_layers=text_att_layers,
                            att_heads=text_att_heads,
                            att_head_dims=text_att_head_dims,
                            ff_hid_dims=text_ff_hid_dims,
                            drop_rate=drop_rate,
                            copy=kwargs['copy'],
                            pad_idx=kwargs['pad_idx'])

    def forward(self, code_ast):
        text_input=code_ast['text'].text_token_input.clone()
        del code_ast['text']
        ast_enc,ast_code_enc,code_enc,code_src_map = self.enc_op(data=code_ast)
        text_output = self.dec_op(ast_enc=ast_enc,ast_code_enc=ast_code_enc,code_enc=code_enc,
                                    code_src_map=code_src_map,
                                    text_input=text_input)
        return text_output

class TModel(TransSeq2Seq):
    def __init__(self,
                #  sim_token_ids,
                 model_dir,
                 model_name='Transformer_based_model',
                 model_id=None,
                 emb_dims=512,
                 code_att_layers=3,
                 code_att_heads=8,
                 code_att_head_dims=None,
                 code_ff_hid_dims=2048,
                 ast_gnn_layers=3,
                 ast_GNN=SAGEConv,
                 ast_gnn_aggr='add',
                 text_att_layers=3,
                 text_att_heads=8,
                 text_att_head_dims=None,
                 text_ff_hid_dims=2048,
                 drop_rate=0.,
                 copy=True,
                 pad_idx=0,
                 train_batch_size=32,
                 pred_batch_size=32,
                 max_train_size=-1,
                 max_valid_size=32 * 10,
                 max_big_epochs=20,
                 regular_rate=1e-5,
                 lr_base=0.001,
                 lr_decay=0.9,
                 min_lr_rate=0.01,
                 warm_big_epochs=2,
                 start_valid_epoch=20,
                 early_stop=20,
                 Net=TNet,
                 Dataset=Datasetx,
                 beam_width=1,
                 train_metrics=[get_sent_bleu],
                 valid_metric=get_sent_bleu,
                 test_metrics=[get_sent_bleu],
                 train_mode=True,
                 **kwargs
                 ):
        logging.info('Construct %s' % model_name)
        super().__init__(model_name=model_name,
                         model_dir=model_dir,
                         model_id=model_id)
        self.init_params = locals()
        # self.sim_token_ids=sim_token_ids
        self.emb_dims = emb_dims
        self.code_att_layers = code_att_layers
        self.code_att_heads = code_att_heads
        self.code_att_head_dims = code_att_head_dims
        self.code_ff_hid_dims = code_ff_hid_dims
        self.ast_gnn_layers = ast_gnn_layers
        self.ast_GNN = ast_GNN
        self.ast_gnn_aggr = ast_gnn_aggr
        self.text_att_layers = text_att_layers
        self.text_att_heads = text_att_heads
        self.text_att_head_dims = text_att_head_dims
        self.text_ff_hid_dims = text_ff_hid_dims
        self.drop_rate = drop_rate
        self.pad_idx = pad_idx
        self.copy = copy
        self.train_batch_size = train_batch_size
        self.pred_batch_size = pred_batch_size
        self.max_train_size = max_train_size
        self.max_valid_size = max_valid_size
        self.max_big_epochs = max_big_epochs
        self.regular_rate = regular_rate
        self.lr_base = lr_base
        self.lr_decay = lr_decay
        self.min_lr_rate = min_lr_rate
        self.warm_big_epochs = warm_big_epochs
        self.start_valid_epoch=start_valid_epoch
        self.early_stop=early_stop
        self.Net = Net
        self.Dataset = Dataset
        self.beam_width = beam_width
        self.train_metrics = train_metrics
        self.valid_metric = valid_metric
        self.test_metrics = test_metrics
        self.train_mode = train_mode

    def _logging_paramerter_num(self):
        logging.info("{} have {} paramerters in total".format(self.model_name, sum(
            x.numel() for x in self.net.parameters() if x.requires_grad)))
        # 计算enc+dec的parameter总数
        code_ast_enc_param_num = sum(x.numel() for x in self.net.module.enc_op.code_enc_op.parameters() if x.requires_grad) + \
                            sum(x.numel() for x in self.net.module.enc_op.gnn_ops.parameters() if x.requires_grad) + \
                            sum(x.numel() for x in self.net.module.enc_op.gnorm_ops.parameters() if x.requires_grad) + \
                            sum(x.numel() for x in self.net.module.enc_op.grelu_ops.parameters() if x.requires_grad)

        text_dec_param_num = sum(x.numel() for x in self.net.module.dec_op.text_dec_op.parameters() if x.requires_grad)
                            # sum(x.numel() for x in self.net.module.dec_op.copy_generator.parameters() if x.requires_grad)
        enc_dec_param_num = code_ast_enc_param_num + text_dec_param_num
        logging.info("{} have {} paramerters in encoder and decoder".format(self.model_name, enc_dec_param_num))

    def fit(self,
            train_data,
            valid_data,
            **kwargs
            ):
        self.ast_max_size=0
        self.code_max_len = 0
        self.io_voc_size = 0
        self.code_mpos_voc_size = 0
        self.code_npos_voc_size = 0
        self.text_max_len=0
        for code_ast,text in zip(train_data['code_asts'],train_data['texts']):
            self.ast_max_size = max(self.ast_max_size,len(code_ast['nodes']))
            self.code_max_len = max(self.code_max_len,code_ast['leaf_node_mask'].sum())
            self.io_voc_size = max(self.io_voc_size,max(code_ast['nodes']))
            self.code_mpos_voc_size = max(self.code_mpos_voc_size,np.max(code_ast['node_in_code_poses'][0,:]))
            self.code_npos_voc_size = max(self.code_npos_voc_size,np.max(code_ast['node_in_code_poses'][1,:]))
            self.text_max_len=max(self.text_max_len,len(text))
        self.io_voc_size+=1
        self.code_mpos_voc_size+=1
        self.code_npos_voc_size+=1

        self.text_voc_size = len(train_data['text_dic']['text_i2w'])  # 包含了begin_idx和end_idx
        self.io_voc_size=max(self.io_voc_size,self.text_voc_size+2*self.code_max_len)
        # print(self.ast_max_size, self.code_max_len,self.text_max_len,
        #       self.io_voc_size, self.text_voc_size,
        #       self.code_mpos_voc_size,self.code_npos_voc_size)

        net = self.Net(
                        # sim_token_ids=self.sim_token_ids,
                        emb_dims=self.emb_dims,
                       ast_max_size=self.ast_max_size,
                       code_max_len=self.code_max_len,
                       text_max_len=self.text_max_len,
                       io_voc_size=self.io_voc_size,
                       code_mpos_voc_size=self.code_mpos_voc_size,
                       code_npos_voc_size=self.code_npos_voc_size,
                       text_voc_size=self.text_voc_size,
                       code_att_layers=self.code_att_layers,
                       code_att_heads=self.code_att_heads,
                       code_att_head_dims=self.code_att_head_dims,
                       code_ff_hid_dims=self.code_ff_hid_dims,
                       ast_gnn_layers=self.ast_gnn_layers,
                       ast_GNN=self.ast_GNN,
                       ast_gnn_aggr=self.ast_gnn_aggr,
                       text_att_layers=self.text_att_layers,
                       text_att_heads=self.text_att_heads,
                       text_att_head_dims=self.text_att_head_dims,
                       text_ff_hid_dims=self.text_ff_hid_dims,
                       drop_rate=self.drop_rate,
                       pad_idx=self.pad_idx,
                       copy=self.copy
                       )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先

        self.net =DataParallel(net.to(device),follow_batch=['x'])  # 并行使用多GPU
        # self.net = BalancedDataParallel(0, net.to(device), dim=0)  # 并行使用多GPU
        # self.net = net.to(device)  # 数据转移到设备
        self._logging_paramerter_num()  # 需要有并行的self.net和self.model_name
        self.net.train()  # 设置网络为训练模式

        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.lr_base,
                                    weight_decay=self.regular_rate)

        # ast_enc_params=self.net.module.enc.ast_enc.parameters()
        # ast_enc_param_ids=list(map(id,ast_enc_params))
        # ex_params=filter(lambda p: id(p) not in ast_enc_param_ids,self.net.parameters())
        # optim_cfg = [{'params': ast_enc_params, 'lr': 0.001,'weight_decay': self.regular_rate* 10.},
        #              {'params': ex_params, 'lr': self.lr_base, 'weight_decay': self.regular_rate}]
        # self.optimizer=optim.Adam(optim_cfg)

        self.criterion = LabelSmoothSoftmaxCEV2(reduction='mean', ignore_index=self.pad_idx, label_smooth=0.0)
        # self.criterion = nn.NLLLoss(ignore_index=self.pad_idx)

        self.text_begin_idx = self.text_voc_size - 1
        self.text_end_idx = self.text_voc_size - 2
        self.tgt_begin_idx,self.tgt_end_idx=self.text_begin_idx,self.text_end_idx
        assert train_data['text_dic']['text_i2w'][self.text_end_idx] == OUT_END_TOKEN
        assert train_data['text_dic']['text_i2w'][self.text_begin_idx] == OUT_BEGIN_TOKEN  # 最后两个是end_idx 和begin_idx

        self.max_train_size = len(train_data['code_asts']) if self.max_train_size == -1 else self.max_train_size
        train_code_asts, train_texts,train_ids = zip(*random.sample(list(zip(train_data['code_asts'], train_data['texts'],train_data['ids'])),
                                                     min(self.max_train_size,
                                                         len(train_data['code_asts']))
                                                     )
                                      )

        train_set = self.Dataset(code_asts=train_code_asts,
                                 texts=train_texts,
                                 ids=train_ids,
                                 text_max_len=self.text_max_len,
                                 text_begin_idx=self.text_begin_idx,
                                 text_end_idx=self.text_end_idx,
                                 pad_idx=self.pad_idx)
        # train_loader = DataLoader(dataset=train_set,
        #                           train_batch_size=self.train_batch_size,
        #                           shuffle=True,
        #                           follow_batch=['ast_node', 'ast_node_after'])
        train_loader=DataListLoader(dataset=train_set,
                                    batch_size=self.train_batch_size,
                                    shuffle=True,
                                    drop_last=True)

        if self.warm_big_epochs is None:
            self.warm_big_epochs = max(self.max_big_epochs // 10, 2)
        self.scheduler = LrWarmUp(self.optimizer,
                                  min_rate=self.min_lr_rate,
                                  lr_decay=self.lr_decay,
                                  warm_steps=self.warm_big_epochs * len(train_loader),
                                  # max(self.max_big_epochs//10,2)*train_loader.__len__()
                                  reduce_steps=len(train_loader))  # 预热次数 train_loader.__len__()
        if self.train_mode:  # 如果进行训练
            # best_net_path = os.path.join(self.model_dir, '{}_best_net.net'.format(self.model_name))
            # self.net.load_state_dict(torch.load(best_net_path))
            # self.net.train()
            # torch.cuda.empty_cache()
            for i in range(0,self.max_big_epochs):
                # logging.info('---------Train big epoch %d/%d' % (i + 1, self.max_big_epochs))
                pbar = tqdm(train_loader)
                for j, batch_data in enumerate(pbar):
                    batch_text_output = []
                    ids=[]
                    for data in batch_data:
                        batch_text_output.append(data['text'].text_token_output.unsqueeze(0))
                        del data['text'].text_token_output
                        ids.append(data['idx'].idx.item())
                        del data['idx']

                    batch_text_output = torch.cat(batch_text_output, dim=0).to(device)
                    # print(batch_text_output[:2,:])
                    pred_text_output = self.net(batch_data)

                    loss = self.criterion(pred_text_output, batch_text_output)  # 计算loss
                    self.optimizer.zero_grad()  # 梯度置0
                    loss.backward()  # 反向传播
                    # clip_grad_norm_(self.net.parameters(),1e-2)  #减弱梯度爆炸
                    self.optimizer.step()  # 优化
                    self.scheduler.step()  # 衰减

                    # log_info = '[Big epoch:{}/{}]'.format(i + 1, self.max_big_epochs)
                    # if i+1>=self.start_valid_epoch:
                    text_dic = {'text_i2w': train_data['text_dic']['text_i2w'],
                               'ex_text_i2ws': [train_data['text_dic']['ex_text_i2ws'][k] for k in ids]}
                    log_info=self._get_log_fit_eval(loss=loss,
                                                    pred_tgt=pred_text_output,
                                                    gold_tgt=batch_text_output,
                                                    tgt_i2w=text_dic
                                                    )
                    log_info = '[Big epoch:{}/{},{}]'.format(i + 1, self.max_big_epochs, log_info)
                    pbar.set_description(log_info)
                    del pred_text_output,batch_text_output,batch_data

                del pbar
                if i+1 >= self.start_valid_epoch:
                    self.max_valid_size = len(valid_data['code_asts']) if self.max_valid_size == -1 else self.max_valid_size
                    valid_srcs, valid_tgts, ex_text_i2ws = zip(*random.sample(list(zip(valid_data['code_asts'],
                                                                                       valid_data['texts'],
                                                                                       valid_data['text_dic']['ex_text_i2ws'])),
                                                                              min(self.max_valid_size,
                                                                                  len(valid_data['code_asts']))
                                                                              )
                                                               )
                    text_dic = {'text_i2w': train_data['text_dic']['text_i2w'],
                                'ex_text_i2ws': ex_text_i2ws}
                    # torch.cuda.empty_cache()
                    worse_epochs = self._do_validation(valid_srcs=valid_srcs,  # valid_data['code_asts']
                                                       valid_tgts=valid_tgts,  # valid_data['texts']
                                                       tgt_i2w=text_dic,  # valid_data['text_dic']
                                                       increase_better=True,
                                                       last=False)  # 根据验证集loss选择best_net
                    # worse_epochs = self._do_validation(valid_srcs=valid_data['code_asts'],  #
                    #                                    valid_tgts=valid_data['texts'],  #
                    #                                    tgt_i2w=valid_data['text_dic'],  #
                    #                                    increase_better=True,
                    #                                    last=False)  # 根据验证集loss选择best_net
                    if worse_epochs>=self.early_stop:
                        break
        # torch.cuda.empty_cache()
        self._do_validation(valid_srcs=valid_data['code_asts'],
                            valid_tgts=valid_data['texts'],
                            tgt_i2w=valid_data['text_dic'],
                            increase_better=True,
                            last=True)  # 根据验证集loss选择best_net
        self._logging_paramerter_num()  # 需要有并行的self.net和self.model_name

    def predict(self,
                code_asts,
                text_dic):
        logging.info('Predict outputs of %s' % self.model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先
        # self.net = self.net.to(device)  # 数据转移到设备,不重新赋值不行
        self.net.eval()  # 切换测试模式
        enc_op=DataParallel(self.net.module.enc_op,follow_batch=['x'])
        dec_op=torch.nn.DataParallel(self.net.module.dec_op)
        # enc.eval()
        # dec.eval()
        data_set = self.Dataset(code_asts=code_asts,
                                texts=None,
                                ids=None,
                                text_max_len=self.text_max_len,
                                text_begin_idx=self.text_begin_idx,
                                text_end_idx=self.text_end_idx,
                                pad_idx=self.pad_idx)  # 数据集，没有out，不需要id

        data_loader = DataListLoader(dataset=data_set,
                                     batch_size=self.pred_batch_size,   #1.5,2.5
                                     shuffle=False)
                                 # follow_batch=['ast_node', 'ast_node_after'])  # data loader
        pred_text_id_np_batches = []  # 所有batch的预测出的id np
        with torch.no_grad():  # 取消梯度
            pbar = tqdm(data_loader)
            for batch_data in pbar:
                # 从batch_data图里把解码器输入输出端数据调出来
                batch_text_input = []
                for data in batch_data:
                    batch_text_input.append(data['text'].text_token_input.unsqueeze(0))
                    del data['text']
                batch_text_input = torch.cat(batch_text_input, dim=0).to(device)

                # 先跑encoder，生成编码
                batch_ast_enc,batch_ast_code_enc,batch_code_enc,batch_code_src_map=enc_op(batch_data)
                batch_text_output: list = []  # 每步的output tensor
                if self.beam_width == 1:
                    for i in range(self.text_max_len + 1):  # 每步开启
                        pred_out = dec_op(ast_enc=batch_ast_enc,ast_code_enc=batch_ast_code_enc,code_enc=batch_code_enc,code_src_map=batch_code_src_map,text_input=batch_text_input)  # 预测该步输出 (B,text_voc_size,L_text)
                        batch_text_output.append(pred_out[:, :, i].unsqueeze(-1).to('cpu').data.numpy())  # 将该步输出加入msg output
                        if i < self.text_max_len:  # 如果没到最后，将id加入input
                            batch_text_input[:, i + 1] = torch.argmax(pred_out[:, :, i], dim=1)
                    batch_pred_text = np.concatenate(batch_text_output, axis=-1)[:, :, :-1]  # (B,D_tgt,L_tgt)
                    batch_pred_text[:, self.tgt_begin_idx, :] = -np.inf  # (B,D_tgt,L_tgt)
                    batch_pred_text[:, self.pad_idx, :] = -np.inf  # (B,D_tgt,L_tgt)
                    batch_pred_text_np = np.argmax(batch_pred_text, axis=1)  # (B,L_tgt) 要除去pad id和begin id
                    pred_text_id_np_batches.append(batch_pred_text_np)  # [(B,L_tgt)]
                else:
                    batch_pred_text=trans_beam_search(net=dec_op,
                                                      beam_width=self.beam_width,
                                                      dec_input_arg_name='text_input',
                                                      length_penalty=1,
                                                      begin_idx=self.tgt_begin_idx,
                                                      pad_idx=self.pad_idx,
                                                      end_idx=self.tgt_end_idx,
                                                      ast_enc=batch_ast_enc,
                                                      ast_code_enc=batch_ast_code_enc,
                                                      code_enc=batch_code_enc,
                                                      code_src_map=batch_code_src_map,
                                                      text_input=batch_text_input
                                                      )     # (B,L_tgt)

                    pred_text_id_np_batches.append(batch_pred_text.to('cpu').data.numpy()[:,:-1])  # [(B,L_tgt)]

        pred_text_id_np = np.concatenate(pred_text_id_np_batches,axis=0)  # (AB,tgt_voc_size,L_tgy)
        self.net.train()  # 切换回训练模式
        # pred_texts=[[{**text_dic['text_i2w'],**text_dic['ex_text_i2ws'][j]}[i] for ]]
        # 利用字典将msg转为token
        pred_texts = self._tgt_ids2tokens(pred_text_id_np, text_dic, self.text_end_idx)

        return pred_texts  # 序列概率输出形状为（A,D)
    
    def generate_texts(self,code_asts,text_dic,res_path,gold_texts,raw_data,token_data,**kwargs):
        '''
        生成src对应的tgt并保存
        :param code_asts:
        :param text_dic:
        :param res_path:
        :param kwargs:
        :return:
        '''
        logging.info('>>>>>>>Generate the targets according to sources and save the result to {}'.format(res_path))
        kwargs.setdefault('beam_width',1)
        res_dir=os.path.dirname(res_path)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        pred_texts=self.predict(code_asts=code_asts,
                                text_dic=text_dic
                                )
        # codes=map(lambda x:x['code']['tokens'],code_asts)
        # codes=self._code_ids2tokens(codes,code_i2w,self.pad_idx)
        gold_texts=self._tgt_ids2tokens(gold_texts,text_dic,self.pad_idx)
        res_data = []
        for i,(pred_text,gold_text,raw_item,token_item) in \
                enumerate(zip(pred_texts,gold_texts,raw_data,token_data)):
            sent_bleu=self.valid_metric([pred_text],[gold_text])
            res_data.append(dict(pred_text=' '.join(pred_text),
                                 gold_text=' '.join(gold_text),
                                 sent_bleu=sent_bleu,
                                 raw_code=raw_item['code'],
                                 raw_text=raw_item['text'],
                                 id=raw_item['id'],
                                 token_text=token_item['text'],
                                 ))
        # res_df=pd.DataFrame(res_dic).T
        # # print(res_df)
        # excel_writer = pd.ExcelWriter(res_path)  # 根据路径savePath打开一个excel写文件
        # res_df.to_excel(excel_writer,header=True,index=True)
        # excel_writer.save()
        with codecs.open(res_path,'w',encoding='utf-8') as f:
            json.dump(res_data,f,indent=4, ensure_ascii=False)
        self._logging_paramerter_num()  # 需要有并行的self.net和self.model_name
        logging.info('>>>>>>>The result has been saved to {}'.format(res_path))

    def _code_ids2tokens(self,code_idss, code_i2w, end_idx):
        return [[code_i2w[idx] for idx in (code_ids[:code_ids.tolist().index(end_idx)]
                                                    if end_idx in code_ids else code_ids)]
                          for code_ids in code_idss]
    
    def _tgt_ids2tokens(self, text_id_np, text_dic, end_idx=0, **kwargs):
        if self.copy:
            text_tokens: list = []
            for j, text_ids in enumerate(text_id_np):
                text_i2w = {**text_dic['text_i2w'], **text_dic['ex_text_i2ws'][j]}
                end_i = text_ids.tolist().index(end_idx) if end_idx in text_ids else len(text_ids)
                text_tokens.append([text_i2w[text_idx] for text_idx in text_ids[:end_i]])
                # if end_i == 0:
                #     print()
        else:
            text_i2w=text_dic['text_i2w']
            text_tokens = [[text_i2w[idx] for idx in (text_ids[:text_ids.tolist().index(end_idx)]
                                                      if end_idx in text_ids else text_ids)]
                          for text_ids in text_id_np]

        return text_tokens

if __name__ == '__main__':

    logging.info('Parameters are listed below: \n'+'\n'.join(['{}: {}'.format(key,value) for key,value in params.items()]))

    model = TModel(
                    # sim_token_ids=np.load(io_token_sim_id_path),
                    model_dir=params['model_dir'],
                   model_name=params['model_name'],
                   model_id=params['model_id'],
                   emb_dims=params['emb_dims'],
                   code_att_layers=params['code_att_layers'],
                   code_att_heads=params['code_att_heads'],
                   code_att_head_dims=params['code_att_head_dims'],
                   code_ff_hid_dims=params['code_ff_hid_dims'],
                   ast_gnn_layers=params['ast_gnn_layers'],
                   ast_GNN=params['ast_GNN'],
                   ast_gnn_aggr=params['ast_gnn_aggr'],
                   text_att_layers=params['text_att_layers'],
                   text_att_heads=params['text_att_heads'],
                   text_att_head_dims=params['text_att_head_dims'],
                   text_ff_hid_dims=params['text_ff_hid_dims'],
                   drop_rate=params['drop_rate'],
                   copy=params['copy'],
                   pad_idx=params['pad_idx'],
                   train_batch_size=params['train_batch_size'],
                   pred_batch_size=params['pred_batch_size'],
                   max_train_size=params['max_train_size'],  # -1 means all
                   max_valid_size=params['max_valid_size'],  ####################10
                   max_big_epochs=params['max_big_epochs'],
                   regular_rate=params['regular_rate'],
                   lr_base=params['lr_base'],
                   lr_decay=params['lr_decay'],
                   min_lr_rate=params['min_lr_rate'],
                   warm_big_epochs=params['warm_big_epochs'],
                   early_stop=params['early_stop'],
                   start_valid_epoch=params['start_valid_epoch'],
                   Net=TNet,
                   Dataset=Datasetx,
                   beam_width=params['beam_width'],
                   train_metrics=train_metrics,
                   valid_metric=valid_metric,
                   test_metrics=test_metrics,
                   train_mode=params['train_mode'])

    logging.info('Load data ...')
    # print(train_avail_data_path)
    with codecs.open(train_avail_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with codecs.open(valid_avail_data_path, 'rb') as f:
        valid_data = pickle.load(f)
    with codecs.open(test_avail_data_path, 'rb') as f:
        test_data = pickle.load(f)
    # io_token_sim_ids=np.load(io_token_sim_id_path)

    # with codecs.open(code_node_i2w_path, 'rb') as f:
    #     code_i2w = pickle.load(f)

    with codecs.open(test_token_data_path,'r') as f:
        test_token_data=json.load(f)

    with codecs.open(test_raw_data_path,'r') as f:
        test_raw_data=json.load(f)

    # train_data['code_asts']=train_data['code_asts'][:1000]
    # train_data['texts']=train_data['texts'][:1000]
    # train_data['ids']=train_data['ids'][:1000]

    # print(len(train_data['texts']), len(valid_data['texts']), len(test_data['texts']))
    model.fit(train_data=train_data,
              valid_data=valid_data)

    for key, value in params.items():
        logging.info('{}: {}'.format(key, value))
    logging.info('Parameters are listed below: \n'+'\n'.join(['{}: {}'.format(key,value) for key,value in params.items()]))

    # test_data['code_asts']=test_data['code_asts'][14246:]
    # test_data['texts']=test_data['texts'][14246:]
    # test_data['ids']=test_data['ids'][14246:]

    # valid_data['code_asts']=valid_data['code_asts'][12762:]
    # valid_data['texts']=valid_data['texts'][12762:]
    # valid_data['ids']=valid_data['ids'][12762:]

    test_eval_df=model.eval(test_srcs=test_data['code_asts'],
                            test_tgts=test_data['texts'],
                            tgt_i2w=test_data['text_dic'])
    logging.info('Model performance on test dataset:\n')
    for i in range(0,len(test_eval_df.columns),4):
        print(test_eval_df.iloc[:, i:i+4])

    model.generate_texts(code_asts=test_data['code_asts'],
                         text_dic=test_data['text_dic'],
                         res_path=res_path,
                         # code_i2w=code_i2w, d
                         gold_texts=test_data['texts'],
                         raw_data=test_raw_data,
                         token_data=test_token_data)

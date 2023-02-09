import logging

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F


import copy

class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        # init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])

        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

        # self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        # torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic.float()  # self.dense(harmonic)
        
class TGAT_Encoder(torch.nn.Module):
    def __init__(self, args, ngh_finder, node_num, edge_num, n_feat = None, e_feat = None):
        super(TGAT_Encoder, self).__init__()

        self.args = args
        self.device = args.device
        self.num_layers = args.num_layers
        self.ngh_finder = ngh_finder # sample neighbors for entities;
        self.attn_mode = args.attn_mode # choose attntion functions;
        self.use_time = args.use_time # whether utilize time information;
        self.agg_method = args.agg_method # choose aggregation functions in the encoder;
        self.n_head = args.n_head # number of attention head;
        self.drop_out = args.drop_out # drop out rate;
        self.emb_dim = args.emb_dim # dimension of entity/relation embedding;
        self.time_dim = args.time_dim # dimention of time embedding;
        self.node_num = node_num # number of entities;
        self.edge_num = edge_num # number of edges;
        self.pretrained = args.use_pretrain
        self.test_batch = args.test_batch
        self.logger = logging.getLogger(__name__)

        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=False)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=False)
        self.feat_dim = self.n_feat_th.shape[1]

        self.n_feat_dim = self.emb_dim
        self.e_feat_dim = self.emb_dim
        self.model_dim = (self.n_feat_dim + self.e_feat_dim + self.time_dim)

        # node and edge feature map:
        self.vars = torch.nn.ParameterDict()
        self.node_w = torch.nn.Parameter(torch.ones(self.feat_dim, self.n_feat_dim))
        torch.nn.init.xavier_normal_(self.node_w)
        self.vars['node_w'] = self.node_w
        self.edge_w = torch.nn.Parameter(torch.ones(self.feat_dim, self.e_feat_dim))
        torch.nn.init.xavier_normal_(self.edge_w)
        self.vars['edge_w'] = self.edge_w

        if self.agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')

            if self.attn_mode == 'multi':
                
                d_k = d_v = self.model_dim // self.n_head

                w_input = torch.nn.Parameter(torch.ones(self.n_feat_dim, self.n_feat_dim))
                nn.init.normal_(w_input, mean=0, std=np.sqrt(2.0 / (2 * self.n_feat_dim)))
                self.vars['w_input'] = w_input

                w_vs = torch.nn.Parameter(torch.ones(self.model_dim, self.model_dim))
                nn.init.normal_(w_vs, mean=0, std=np.sqrt(2.0 / (self.model_dim  + d_v)))
                self.vars['w_vs'] = w_vs

                w_ks = torch.nn.Parameter(torch.ones(self.model_dim, self.model_dim))
                nn.init.normal_(w_ks, mean=0, std=np.sqrt(2.0 / (self.model_dim  + d_v)))
                self.vars['w_ks'] = w_ks

                w_qs = torch.nn.Parameter(torch.ones(self.model_dim, self.model_dim))
                nn.init.normal_(w_qs, mean=0, std=np.sqrt(2.0 / (self.model_dim  + d_v)))
                self.vars['w_qs'] = w_qs

                w_fc = torch.nn.Parameter(torch.ones(self.model_dim, self.model_dim))
                nn.init.normal_(w_fc, mean=0, std=np.sqrt(2.0 / (self.model_dim  + d_v)))
                self.vars['w_fc'] = w_fc

                w1_agg_fc = torch.nn.Parameter(torch.ones(2 * self.n_feat_dim, self.model_dim + self.n_feat_dim))
                nn.init.normal_(w1_agg_fc, mean=0, std=np.sqrt(2.0 / (self.model_dim  + d_v)))
                self.vars['w1_agg_fc'] = w1_agg_fc

                w2_agg_fc = torch.nn.Parameter(torch.ones(self.n_feat_dim, 2 * self.n_feat_dim))
                nn.init.normal_(w2_agg_fc, mean=0, std=np.sqrt(2.0 / (self.model_dim  + d_v)))
                self.vars['w2_agg_fc'] = w2_agg_fc

                # multi_head att weight d_k = d_v = model_dim // n_head

            elif self.attn_mode == 'simple':
                fc_w = torch.nn.Parameter(torch.ones(self.model_dim, self.model_dim))
                nn.init.xavier_normal_(fc_w)
                self.vars['fc_w'] = fc_w

                shared_attn = torch.nn.Parameter(torch.ones(1, 2 * self.model_dim))
                nn.init.xavier_normal_(shared_attn)
                self.vars['shared_attn'] = shared_attn

            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.n_feat_dim, self.e_feat_dim, self.time_dim) for _ in range(self.num_layers)])
        else:

            raise ValueError('invalid agg_method value, use attn or lstm')

        if self.use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.time_dim)
        else:
            raise ValueError('invalid time option!')

    def forward(self, head_idx_l, tail_idx_l, rel_idx_l, cut_time_l, num_neighbors, vars_dict=None):

        """
        Params
        ------
        nodrel_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that nodrel_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """

        if vars_dict is None:
            # print('vars_dict is None')
            vars_dict = self.vars
        # else:
            # print('vars_dict is not None')
        rel_ids_batch_th = torch.from_numpy(rel_idx_l).long().to(self.device)

        head_embed = self.tem_conv(head_idx_l, cut_time_l, self.num_layers, vars_dict, num_neighbors)
        tail_embed = self.tem_conv(tail_idx_l, cut_time_l, self.num_layers, vars_dict, num_neighbors)
        rel_embed = torch.mm(self.edge_raw_embed(rel_ids_batch_th), self.edge_w)
        return head_embed, tail_embed, rel_embed

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, vars_dict, num_neighbors):
        assert (curr_layers >= 0)
        # print('curr_layers ', curr_layers)

        device = self.n_feat_th.device
        # print('device ', device)

        batch_size = len(src_idx_l)

        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        # print('src_node_batch_th ', src_node_batch_th)

        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)
        # print('cut_time_l_th.shape ', cut_time_l_th.shape)

        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # print('cut_time_l_th.unsqueeze ', cut_time_l_th.shape)

        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        # print('src_node_t_embed ', src_node_t_embed)

        src_node_feat = torch.mm(self.node_raw_embed(src_node_batch_th), self.node_w)
        # print('src_node_feat ', src_node_feat)

        if curr_layers == 0:
            # print('curr_layers ', curr_layers)
            # print('return src_node_feat')
            return src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(src_idx_l, cut_time_l, curr_layers=curr_layers - 1,
                                               vars_dict=vars_dict, num_neighbors=num_neighbors)
            # print('curr_layers ', curr_layers)
            # print(src_node_conv_feat.shape)
            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch \
                = self.ngh_finder.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors=num_neighbors)

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device)
            # print('src_idx_l', src_idx_l)
            # print('src_ngh_node_batch_th ', src_ngh_node_batch_th)
            src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(device)
            # print('src_ngh_eidx_batch', src_ngh_eidx_batch.size())
            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            # print('src_ngh_t_batch_delta', src_ngh_t_batch_delta)
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)
            # print('src_ngh_t_batch_th', src_ngh_t_batch_th)

            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten()  # reshape(batch_size, -1)
            # print('src_ngh_node_batch_flat ', src_ngh_node_batch_flat)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten()  # reshape(batch_size, -1)
            # print('src_ngh_t_batch_flat ', src_ngh_t_batch_flat)

            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat,
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1,
                                                   num_neighbors=num_neighbors, vars_dict=vars_dict)

            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, self.n_feat_dim)
            # print("src_ngh_feat.shape", src_ngh_feat.shape)

            # get edge time features and node features
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            # print('true src_ngh_t_embed ', src_ngh_t_embed)
            src_ngn_edge_feat = torch.mm(
                self.edge_raw_embed(src_ngh_eidx_batch).view(-1, self.feat_dim),
                self.edge_w).view(batch_size, -1, self.n_feat_dim)

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            # print('mask', mask)
            attn_m = self.attn_model_list[curr_layers - 1]

            local, weight = attn_m(src_node_conv_feat,
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed,
                                   src_ngn_edge_feat,
                                   mask,
                                   vars_dict)
            return local

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars

    def update_embeddings(self, num_neighbors, mode = "ext"):
        rel_embedding = self.update_edge_embedding()
        node_embedding = self.update_node_embedding(num_neighbors, mode = mode)
        return node_embedding, rel_embedding

    def update_edge_embedding(self):
        return torch.mm(self.edge_raw_embed.weight, self.edge_w)

    def update_node_embedding(self, num_neighbors, vars_dict=None, mode = "ext"):
        if vars_dict is None:
            vars_dict = self.vars
        if mode == "ext":
            node_emb = torch.Tensor(self.args.test_time, self.node_num, self.emb_dim).to(self.device)
            node_idx = np.array(range(self.node_num))
            num_batch = int(len(node_idx) / self.test_batch)
            for t in range(self.args.train_time, self.args.train_time + self.args.test_time):
                for i in range(num_batch):
                    node_idx_l = node_idx[self.test_batch * i:self.test_batch * (i + 1)]
                    cut_time_l = np.array([t]*len(node_idx_l ))
                    node_emb_l = self.tem_conv(node_idx_l, cut_time_l, self.num_layers, vars_dict, num_neighbors)
                    node_emb[t - self.args.train_time, torch.from_numpy(node_idx_l).long().to(self.device)] = node_emb_l.data
        
        elif mode == "int":
            node_emb = torch.Tensor(self.args.train_time, self.node_num, self.emb_dim).to(self.device)
            node_idx = np.array(range(self.node_num))
            num_batch = int(len(node_idx) / self.test_batch)
            for t in range(0, self.args.train_time):
                for i in range(num_batch):
                    node_idx_l = node_idx[self.test_batch * i:self.test_batch * (i + 1)]
                    cut_time_l = np.array([t]*len(node_idx_l ))
                    node_emb_l = self.tem_conv(node_idx_l, cut_time_l, self.num_layers, vars_dict, num_neighbors)
                    node_emb[t, torch.from_numpy(node_idx_l).long().to(self.device)] = node_emb_l.data

        return node_emb

class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """

    def __init__(self, feat_dim, edge_dim, time_dim,
                 attn_mode='multi', n_head=2, drop_out=0.1):
        
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """

        super(AttnModel, self).__init__()

        self.feat_dim = feat_dim
        self.time_dim = time_dim

        self.edge_in_dim = (feat_dim + edge_dim + time_dim)
        self.model_dim = self.edge_in_dim

        assert (self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode

        if attn_mode == 'multi':
            self.multi_head_target = MultiHeadAttention(n_head,
                                                        d_model=self.model_dim,
                                                        d_k=self.model_dim // n_head,
                                                        d_v=self.model_dim // n_head,
                                                        dropout=drop_out)
            self.logger.info('Using scaled multi attention')

        elif attn_mode == 'simple':
            self.multi_head_target = Attention(d_model=self.model_dim,
                                               d_k=self.model_dim,
                                               d_v=self.model_dim,
                                               dropout=drop_out)
            self.logger.info('Using scaled simple attention')
        else:
            raise ValueError('attn_mode can only be multi or simple')

    def forward(self, src, src_t, seq, seq_t, seq_e, mask, vars_dict):
        
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        src_ext = torch.unsqueeze(src, dim=1)  # src [B, 1, D]
        src_e_ph = torch.zeros_like(src_ext)
        q = torch.cat([src_ext, src_e_ph, src_t], dim=2)  # [B, 1, D + De + Dt] -> [B, 1, D]
        k = torch.cat([seq, seq_e, seq_t], dim=2)  # [B, 1, D + De + Dt] -> [B, 1, D]

        mask = torch.unsqueeze(mask, dim=2)  # mask [B, N, 1]
        mask = mask.permute([0, 2, 1])  # mask [B, 1, N]

        # # target-attention
        output, attn = self.multi_head_target(q=q, k=k, v=k, vars_dict=vars_dict, mask=mask)  # output: [B, 1, D + Dt], attn: [B, 1, N]
        # print('src.shape', src.shape)
        output = output.squeeze(1)
        # print('output.shape', output.shape)
        # print('output', output)
        # print('output.squeeze().shape', output.shape)
        attn = attn.squeeze()

        # output = self.merger(output, src)
        x = torch.cat([output, src], dim=1)
        # x = self.layer_norm(x)
        x = F.relu(F.linear(x, vars_dict['w1_agg_fc']))
        output = F.linear(x, vars_dict['w2_agg_fc'])

        return output, attn

class MultiHeadAttention(torch.nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # replaced by vars_dict
        # self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        # self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        # self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        # nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # self.fc = nn.Linear(n_head * d_v, d_model)

        # nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, vars_dict, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        # print('q.shape',  q.shape)
        # q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        # print('q.shape',  q.shape)
        # k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        # v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = F.linear(q, vars_dict['w_qs']).view(sz_b, len_q, n_head, d_k)
        k = F.linear(k, vars_dict['w_ks']).view(sz_b, len_k, n_head, d_k)
        v = F.linear(v, vars_dict['w_vs']).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)
        # print('output.shape', outpshape)

        output = self.dropout(F.linear(output, vars_dict['w_fc']))
        output = self.layer_norm(output + residual)
        # output = self.layer_norm(output)

        return output, attn

class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # print('q.shape', q.shape)
        # print('k.shape', k.shape)
        attn = torch.bmm(q, k.transpose(1, 2))  # calculate attention
        # print('attn', attn.shape)
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        # print('attn', attn)
        attn = self.dropout(attn)  # [n * b, l_v, d]

        output = torch.bmm(attn, v)

        return output, attn

class Attention(torch.nn.Module):
    ''' Simple Attention module '''

    def __init__(self, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        # self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, vars_dict, mask=None):

        d_k, d_v = self.d_k, self.d_v

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        # print('q.shape',  q.shape)
        # q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        # print('q.shape',  q.shape)
        # k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        # v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # q = F.linear(q, vars_dict['w_qs']).view(sz_b, len_q, d_k)
        # k = F.linear(k, vars_dict['w_ks']).view(sz_b, len_k, d_k)
        # v = F.linear(v, vars_dict['w_vs']).view(sz_b, len_v, d_v)

        q = q.contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2)
        q = q.expand(sz_b, len_q, len_k, d_k)    # [(n*b), lq, lk, dk]

        k = k.contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1)
        k = k.expand(sz_b, len_q, len_k, d_k)    # [(n*b), lq, lk, dk]

        v = v.contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        # print('q', q.shape)
        # print('k', k.shape)
        mask = mask.repeat(1, 1, 1)  # (n*b) x .. x ..
        # print('mask', mask.shape)

        q_k = torch.cat([q, k], dim=3)
        attn = F.linear(q_k, vars_dict['shared_attn']).squeeze(dim=3)

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)
        # attn = self.dropout(attn)
        # print('attn.shape', attn.shape)
        # print('v.shape', v.shape)

        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)
        # print('output.shape', output.shape)

        output = output.contiguous().view(sz_b, len_q, d_k)  # b x lq x (n*dv)
        # print('output.shape', outpshape)

        # output = self.dropout(F.linear(output, vars_dict['fc_w']))
        output = F.linear(output, vars_dict['fc_w'])
        output = self.layer_norm(output + residual)
        # output = self.layer_norm(output)

        return output, attn

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import math
import os
import warnings
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime

import h5py
import nni
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings('ignore')  # if to ignore warnings
RANDOM_SEED = 23
#from modeling.SA_models import SAITS, TransformerEncoder
#from modeling.brits import BRITS
#from modeling.mrnn import MRNN
#from modeling.unified_dataloader import UnifiedDataLoader
from utils import Controller, setup_logger, save_model, load_model, check_saving_dir_for_model, \
    masked_mae_cal, masked_rmse_cal, masked_mre_cal
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class ScaledDotProductAttention(nn.Module):
    """scaled dot-product attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """original Transformer multi-head attention"""

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_k ** 0.5, attn_dropout)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            # this mask is imputation mask, which is not generated from each batch, so needs broadcasting on batch dim
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)  # For batch and head axis broadcasting.

        v, attn_weights = self.attention(q, k, v, attn_mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, attn_dropout=0.1, **kwargs):
        super(EncoderLayer, self).__init__()

        self.diagonal_attention_mask = kwargs['diagonal_attention_mask']
        self.device = kwargs['device']
        self.d_time = d_time
        self.d_feature = d_feature

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input):
        if self.diagonal_attention_mask:
            mask_time = torch.eye(self.d_time).to(self.device)
        else:
            mask_time = None

        residual = enc_input
        # here we apply LN before attention cal, namely Pre-LN, refer paper https://arxiv.org/abs/2002.04745
        enc_input = self.layer_norm(enc_input)
        enc_output, attn_weights = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=mask_time)
        enc_output = self.dropout(enc_output)
        enc_output += residual

        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights

def masked_mae_cal(inputs, target, mask):
    """ calculate Mean Absolute Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """ Sinusoid position encoding table """

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class TransformerEncoder(nn.Module):
    def __init__(self, n_groups, n_group_inner_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
                 **kwargs):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = kwargs['input_with_mask']
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature
        self.param_sharing_strategy = kwargs['param_sharing_strategy']
        self.MIT = kwargs['MIT']
        self.device = kwargs['device']

        if kwargs['param_sharing_strategy'] == 'between_group':
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, dropout, **kwargs)
                for _ in range(n_group_inner_layers)
            ])
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers
            # and repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, dropout, **kwargs)
                for _ in range(n_groups)
            ])

        self.embedding = nn.Linear(actual_d_feature, d_model)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        self.dropout = nn.Dropout(p=dropout)
        self.reduce_dim = nn.Linear(d_model, d_feature)

    def impute(self, inputs):
        X, masks = inputs['X'], inputs['missing_mask']
        input_X = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        input_X = self.embedding(input_X)
        enc_output = self.dropout(self.position_enc(input_X))

        if self.param_sharing_strategy == 'between_group':
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        learned_presentation = self.reduce_dim(enc_output)
        imputed_data = masks * X + (1 - masks) * learned_presentation  # replace non-missing part with original data
        return imputed_data, learned_presentation

    def forward(self, inputs, stage):
        X, masks = inputs['X'], inputs['missing_mask']
        imputed_data, learned_presentation = self.impute(inputs)
        reconstruction_MAE = masked_mae_cal(learned_presentation, X, masks)
        if (self.MIT or stage == 'val') and stage != 'test':
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(learned_presentation, inputs['X_holdout'], inputs['indicating_mask'])
        else:
            imputation_MAE = torch.tensor(0.0)

        return {'imputed_data': imputed_data,
                'reconstruction_loss': reconstruction_MAE, 'imputation_loss': imputation_MAE,
                'reconstruction_MAE': reconstruction_MAE, 'imputation_MAE': imputation_MAE}


class SAITS(nn.Module):
    def __init__(self, n_groups, n_group_inner_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
                 **kwargs):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = kwargs['input_with_mask']
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature
        self.param_sharing_strategy = kwargs['param_sharing_strategy']
        self.MIT = kwargs['MIT']
        self.device = kwargs['device']

        if kwargs['param_sharing_strategy'] == 'between_group':
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack_for_first_block = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0, **kwargs)
                for _ in range(n_group_inner_layers)
            ])
            self.layer_stack_for_second_block = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0, **kwargs)
                for _ in range(n_group_inner_layers)
            ])
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers
            # and repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack_for_first_block = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0, **kwargs)
                for _ in range(n_groups)
            ])
            self.layer_stack_for_second_block = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0, **kwargs)
                for _ in range(n_groups)
            ])

        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        # for operation on time dim
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_z = nn.Linear(d_model, d_feature)
        # for operation on measurement dim
        self.embedding_2 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_beta = nn.Linear(d_model, d_feature)
        self.reduce_dim_gamma = nn.Linear(d_feature, d_feature)
        # for delta decay factor
        self.weight_combine = nn.Linear(d_feature + d_time, d_feature)

    def impute(self, inputs):
        X, masks = inputs['X'], inputs['missing_mask']
        # first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(self.position_enc(input_X_for_first))  # namely term e in math algo
        if self.param_sharing_strategy == 'between_group':
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_first_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # second DMSA block
        input_X_for_second = torch.cat([X_prime, masks], dim=2) if self.input_with_mask else X_prime
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(input_X_for_second)  # namely term alpha in math algo
        if self.param_sharing_strategy == 'between_group':
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_second_block:
                    enc_output, attn_weights = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_second_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, attn_weights = encoder_layer(enc_output)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # attention-weighted combine
        attn_weights = attn_weights.squeeze()  # namely term A_hat in math algo
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = F.sigmoid(self.weight_combine(torch.cat([masks, attn_weights], dim=2)))  # namely term eta
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        X_c = masks * X + (1 - masks) * X_tilde_3  # replace non-missing part with original data
        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def forward(self, inputs, stage):
        X, masks = inputs['X'], inputs['missing_mask']
        reconstruction_loss = 0
        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(inputs)

        reconstruction_loss += masked_mae_cal(X_tilde_1, X, masks)
        reconstruction_loss += masked_mae_cal(X_tilde_2, X, masks)
        final_reconstruction_MAE = masked_mae_cal(X_tilde_3, X, masks)
        reconstruction_loss += final_reconstruction_MAE
        reconstruction_loss /= 3

        if (self.MIT or stage == 'val') and stage != 'test':
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(X_tilde_3, inputs['X_holdout'], inputs['indicating_mask'])
        else:
            imputation_MAE = torch.tensor(0.0)

        return {'imputed_data': imputed_data,
                'reconstruction_loss': reconstruction_loss, 'imputation_loss': imputation_MAE,
                'reconstruction_MAE': final_reconstruction_MAE, 'imputation_MAE': imputation_MAE}

def read_arguments(arg_parser, cfg_parser):
    # file path
    arg_parser.dataset_base_dir = cfg_parser.get('file_path', 'dataset_base_dir')
    arg_parser.result_saving_base_dir = cfg_parser.get('file_path', 'result_saving_base_dir')
    # dataset info
    arg_parser.seq_len = cfg_parser.getint('dataset', 'seq_len')
    arg_parser.batch_size = cfg_parser.getint('dataset', 'batch_size')
    arg_parser.num_workers = cfg_parser.getint('dataset', 'num_workers')
    arg_parser.feature_num = cfg_parser.getint('dataset', 'feature_num')
    arg_parser.dataset_name = cfg_parser.get('dataset', 'dataset_name')
    arg_parser.dataset_path = os.path.join(arg_parser.dataset_base_dir, arg_parser.dataset_name)
    arg_parser.eval_every_n_steps = cfg_parser.getint('dataset', 'eval_every_n_steps')
    # training settings
    arg_parser.MIT = cfg_parser.getboolean('training', 'MIT')
    arg_parser.ORT = cfg_parser.getboolean('training', 'ORT')   
    arg_parser.lr = cfg_parser.getfloat('training', 'lr')
    arg_parser.optimizer_type = cfg_parser.get('training', 'optimizer_type')
    arg_parser.weight_decay = cfg_parser.getfloat('training', 'weight_decay')
    arg_parser.device = cfg_parser.get('training', 'device')
    arg_parser.epochs = cfg_parser.getint('training', 'epochs')
    arg_parser.early_stop_patience = cfg_parser.getint('training', 'early_stop_patience')
    arg_parser.model_saving_strategy = cfg_parser.get('training', 'model_saving_strategy')
    arg_parser.max_norm = cfg_parser.getfloat('training', 'max_norm')
    arg_parser.imputation_loss_weight = cfg_parser.getfloat('training', 'imputation_loss_weight')
    arg_parser.reconstruction_loss_weight = cfg_parser.getfloat('training', 'reconstruction_loss_weight')
    # model settings
    arg_parser.model_name = cfg_parser.get('model', 'model_name')
    arg_parser.model_type = cfg_parser.get('model', 'model_type')
    
    return arg_parser

def summary_write_into_tb(summary_writer, info_dict, step, stage):
    """write summary into tensorboard file"""
    summary_writer.add_scalar(f'total_loss/{stage}', info_dict['total_loss'], step)
    summary_writer.add_scalar(f'imputation_loss/{stage}', info_dict['imputation_loss'], step)
    summary_writer.add_scalar(f'imputation_MAE/{stage}', info_dict['imputation_MAE'], step)
    summary_writer.add_scalar(f'reconstruction_loss/{stage}', info_dict['reconstruction_loss'], step)
    summary_writer.add_scalar(f'reconstruction_MAE/{stage}', info_dict['reconstruction_MAE'], step)


def result_processing(results, args):
    """process results and losses for each training step"""
    results['total_loss'] = torch.tensor(0.0, device=args.device)
    if args.model_type == 'BRITS':
        results['total_loss'] = results['consistency_loss'] * args.consistency_loss_weight
    results['reconstruction_loss'] = results['reconstruction_loss'] * args.reconstruction_loss_weight
    results['imputation_loss'] = results['imputation_loss'] * args.imputation_loss_weight
    if args.MIT:
        results['total_loss'] += results['imputation_loss']
    if args.ORT:
        results['total_loss'] += results['reconstruction_loss']
    return results

def process_each_training_step(results, optimizer, val_dataloader, training_controller, summary_writer, logger, args, model):
    """process each training step and return whether to early stop"""
    state_dict = training_controller(stage='train')
    # apply gradient clipping if args.max_norm != 0
    if args.max_norm != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
        
    results['total_loss'].backward()
    optimizer.step()

    summary_write_into_tb(summary_writer, results, state_dict['train_step'], 'train')
    if state_dict['train_step'] % args.eval_every_n_steps == 0:
        state_dict_from_val = validate(model, val_dataloader, summary_writer, training_controller, logger, args, optimizer)
        if state_dict_from_val['should_stop']:
            logger.info(f'Early stopping worked, stop now...')
            return True
    return False

def model_processing(data, model, stage, 
                     # following arguments are only required in the training stage
                     optimizer=None, val_dataloader=None, summary_writer=None, training_controller=None, logger=None, args=None):
    if stage == 'train':
        optimizer.zero_grad()
        if not args.MIT:
            if args.model_type in ['BRITS', 'MRNN']:
                indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas = \
                    map(lambda x: x.to(args.device), data)
                inputs = {'indices': indices, 'forward': {'X': X, 'missing_mask': missing_mask, 'deltas': deltas},
                          'backward': {'X': back_X, 'missing_mask': back_missing_mask, 'deltas': back_deltas}}
            else:  # then for self-attention based models, i.e. Transformer/SAITS
                indices, X, missing_mask = map(lambda x: x.to(args.device), data)
                inputs = {'indices': indices, 'X': X, 'missing_mask': missing_mask}
            results = result_processing(model(inputs, stage), args)
            early_stopping = process_each_training_step(results, optimizer, val_dataloader, training_controller,
                                                        summary_writer, logger, args, model)
        else:
            if args.model_type in ['BRITS', 'MRNN']:
                indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas, X_holdout, \
                indicating_mask = map(lambda x: x.to(args.device), data)
                inputs = {'indices': indices, 'X_holdout': X_holdout, 'indicating_mask': indicating_mask,
                          'forward': {'X': X, 'missing_mask': missing_mask, 'deltas': deltas},
                          'backward': {'X': back_X, 'missing_mask': back_missing_mask, 'deltas': back_deltas}}
            else:
                indices, X, missing_mask, X_holdout, indicating_mask = map(lambda x: x.to(args.device), data)
                inputs = {'indices': indices, 'X': X, 'missing_mask': missing_mask,
                          'X_holdout': X_holdout, 'indicating_mask': indicating_mask}
            results = result_processing(model(inputs, stage), args)
            early_stopping = process_each_training_step(results, optimizer, val_dataloader,
                                                        training_controller, summary_writer, logger, args, model)
        return early_stopping

    else:  # in val/test stage
        if args.model_type in ['BRITS', 'MRNN']:
            indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas, X_holdout, \
            indicating_mask = map(lambda x: x.to(args.device), data)
            inputs = {'indices': indices, 'X_holdout': X_holdout, 'indicating_mask': indicating_mask,
                      'forward': {'X': X, 'missing_mask': missing_mask, 'deltas': deltas},
                      'backward': {'X': back_X, 'missing_mask': back_missing_mask, 'deltas': back_deltas}}
            inputs['missing_mask'] = inputs['forward']['missing_mask']  # for error calculation in validation stage
        else:
            indices, X, missing_mask, X_holdout, indicating_mask = map(lambda x: x.to(args.device), data)
            inputs = {'indices': indices, 'X': X, 'missing_mask': missing_mask,
                      'X_holdout': X_holdout, 'indicating_mask': indicating_mask}
        results = model(inputs, stage)
        results = result_processing(results, args)
        return inputs, results


def train(model, optimizer, train_dataloader, test_dataloader, summary_writer, training_controller, logger, args):
    for epoch in range(args.epochs):
        early_stopping = False
        args.final_epoch = True if epoch == args.epochs - 1 else False
        for idx, data in enumerate(train_dataloader):
            model.train()
            #try:
            early_stopping = model_processing(data, model, 'train', optimizer, test_dataloader, summary_writer,
                                              training_controller, logger, args)
            #except Exception:
            #    continue
            if early_stopping:
                break
        if early_stopping:
            break
        training_controller.epoch_num_plus_1()
    logger.info('Finished all epochs. Stop training now.')

def validate(model, val_iter, summary_writer, training_controller, logger, args, optimizer):
    model.eval()
    evalX_collector, evalMask_collector, imputations_collector = [], [], []
    total_loss_collector, imputation_loss_collector, reconstruction_loss_collector, reconstruction_MAE_collector = [], [], [], []

    with torch.no_grad():
        for idx, data in enumerate(val_iter):
            inputs, results = model_processing(data, model, 'val', args = args)
            evalX_collector.append(inputs['X_holdout'])
            evalMask_collector.append(inputs['indicating_mask'])
            imputations_collector.append(results['imputed_data'])

            total_loss_collector.append(results['total_loss'].data.cpu().numpy())
            reconstruction_MAE_collector.append(results['reconstruction_MAE'].data.cpu().numpy())
            reconstruction_loss_collector.append(results['reconstruction_loss'].data.cpu().numpy())
            imputation_loss_collector.append(results['imputation_loss'].data.cpu().numpy())

        evalX_collector = torch.cat(evalX_collector)
        evalMask_collector = torch.cat(evalMask_collector)
        imputations_collector = torch.cat(imputations_collector)
        imputation_MAE = masked_mae_cal(imputations_collector, evalX_collector, evalMask_collector)
    info_dict = {'total_loss': np.asarray(total_loss_collector).mean(),
                 'reconstruction_loss': np.asarray(reconstruction_loss_collector).mean(),
                 'imputation_loss': np.asarray(imputation_loss_collector).mean(),
                 'reconstruction_MAE': np.asarray(reconstruction_MAE_collector).mean(),
                 'imputation_MAE': imputation_MAE.cpu().numpy().mean()}
    state_dict = training_controller('val', info_dict, logger)
    summary_write_into_tb(summary_writer, info_dict, state_dict['val_step'], 'val')
    if args.param_searching_mode:
        nni.report_intermediate_result(info_dict['imputation_MAE'])
        if args.final_epoch or state_dict['should_stop']:
            nni.report_final_result(state_dict['best_imputation_MAE'])

    if (state_dict['save_model'] and args.model_saving_strategy) or args.model_saving_strategy == 'all':
        saving_path = os.path.join(
            args.model_saving, 'model_trainStep_{}_valStep_{}_imputationMAE_{:.4f}'.
                format(state_dict['train_step'], state_dict['val_step'], info_dict['imputation_MAE']))
        save_model(model, optimizer, state_dict, args, saving_path)
        logger.info(f'Saved model -> {saving_path}')
    return state_dict


def test_trained_model(model, test_dataloader, args):
    logger.info(f'Start evaluating on whole test set...')
    model.eval()
    evalX_collector, evalMask_collector, imputations_collector = [], [], []
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            inputs, results = model_processing(data, model, 'test', args = args)
            # collect X_holdout, indicating_mask and imputed data
            evalX_collector.append(inputs['X_holdout'])
            evalMask_collector.append(inputs['indicating_mask'])
            imputations_collector.append(results['imputed_data'])

        evalX_collector = torch.cat(evalX_collector)
        evalMask_collector = torch.cat(evalMask_collector)
        imputations_collector = torch.cat(imputations_collector)
        imputation_MAE = masked_mae_cal(imputations_collector, evalX_collector, evalMask_collector)
        imputation_RMSE = masked_rmse_cal(imputations_collector, evalX_collector, evalMask_collector)
        imputation_MRE = masked_mre_cal(imputations_collector, evalX_collector, evalMask_collector)

    assessment_metrics = {'imputation_MAE on the test set': imputation_MAE,
                          'imputation_RMSE on the test set': imputation_RMSE,
                          'imputation_MRE on the test set': imputation_MRE,
                          'trainable parameter num': args.total_params}
    with open(os.path.join(args.result_saving_path, 'overall_performance_metrics.out'), 'w') as f:
        logger.info('Overall performance metrics are listed as follows:')
        for k, v in assessment_metrics.items():
            logger.info(f'{k}: {v}')
            f.write(k + ':' + str(v))
            f.write('\n')

def impute_all_missing_data(model, train_data, val_data, test_data):
    logger.info(f'Start imputing all missing data in all train/val/test sets...')
    model.eval()
    imputed_data_dict = {}
    with torch.no_grad():
        for dataloader, set_name in zip([train_data, val_data, test_data], ['train', 'val', 'test']):
            indices_collector, imputations_collector = [], []
            for idx, data in enumerate(dataloader):
                if args.model_type in ['BRITS', 'MRNN']:
                    indices, X, missing_mask, deltas, back_X, back_missing_mask, back_deltas = \
                        map(lambda x: x.to(args.device), data)
                    inputs = {'indices': indices, 'forward': {'X': X, 'missing_mask': missing_mask, 'deltas': deltas},
                              'backward': {'X': back_X, 'missing_mask': back_missing_mask, 'deltas': back_deltas}}
                else:  # then for self-attention based models, i.e. Transformer/SAITS
                    indices, X, missing_mask = map(lambda x: x.to(args.device), data)
                    inputs = {'indices': indices, 'X': X, 'missing_mask': missing_mask}
                imputed_data, _ = model.impute(inputs)
                indices_collector.append(indices)
                imputations_collector.append(imputed_data)

            indices_collector = torch.cat(indices_collector)
            indices = indices_collector.cpu().numpy().reshape(-1)
            imputations_collector = torch.cat(imputations_collector)
            imputations = imputations_collector.data.cpu().numpy()
            ordered = imputations[np.argsort(indices)]  # to ensure the order of samples
            imputed_data_dict[set_name] = ordered

    imputation_saving_path = os.path.join(args.result_saving_path, 'imputations.h5')
    with h5py.File(imputation_saving_path, 'w') as hf:
        hf.create_dataset('imputed_train_set', data=imputed_data_dict['train'])
        hf.create_dataset('imputed_val_set', data=imputed_data_dict['val'])
        hf.create_dataset('imputed_test_set', data=imputed_data_dict['test'])
    logger.info(f'Done saving all imputed data into {imputation_saving_path}.')
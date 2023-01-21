import torch
if torch.cuda.is_available():
    torch.cuda.init()
    assert torch.cuda.is_initialized()
    print('Cuda is ready')
    
import pandas as pd
import argparse
import os 
import pandas as pd
import argparse
import os
import sys
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def read_data():
    #raw_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQwC6jRtVUk-2dkk2W3BDJZTOdsS427LN8Ixo-rQF4Afs6ice0rof7qh_EbnAy5lYEGqX-TCSvjpPyr/pub?gid=1483941225&single=true&output=csv',
    #                 index_col=['codigo','Tiempo']).drop(['Fecha','Exposicion'], axis=1)
    
    raw_df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQwC6jRtVUk-2dkk2W3BDJZTOdsS427LN8Ixo-rQF4Afs6ice0rof7qh_EbnAy5lYEGqX-TCSvjpPyr/pub?gid=1713335339&single=true&output=csv',
                     index_col=['codigo','Tiempo']).drop(['Fecha','Exposicion'], axis=1)

    raw_df.sort_index(inplace=True)
    return raw_df

def generate_train_val_test(raw_df, train_size=.8):
    unique_IDs = raw_df.index.get_level_values(0).unique().values
    train_set_ids, val_set_ids = train_test_split(unique_IDs, train_size=train_size)
    _,            test_set_ids = train_test_split(val_set_ids, train_size=train_size)
    return train_set_ids, val_set_ids, test_set_ids

def process_each_set(set_df,smallest_window):
    sample_ids = set_df.index.get_level_values(0).to_numpy().reshape(-1, smallest_window)[:, 0]
    feature_names = set_df.columns.tolist()
    X = set_df.to_numpy()
    X = X.reshape(len(sample_ids), smallest_window, len(feature_names))
    return X, feature_names


    
def random_mask(vector, artificial_missing_rate):
    """generate indices for random mask"""
    assert len(vector.shape) == 1
    indices = np.where(~np.isnan(vector))[0].tolist()
    indices = np.random.choice(indices, int(len(indices) * artificial_missing_rate))
    return indices

def add_artificial_mask(X, artificial_missing_rate, set_name):
    """ Add artificial missing values.
    Parameters
    ----------
    X: feature vectors
    artificial_missing_rate: rate of artificial missing values that are going to be create
    set_name: dataset name, train/val/test
    """
    sample_num, seq_len, feature_num = X.shape
    if set_name == 'train':
        # if this is train set, we don't need add artificial missing values right now.
        # If we want to apply MIT during training, dataloader will randomly mask out some values to generate X_hat

        # calculate empirical mean for model GRU-D, refer to paper
        mask = (~np.isnan(X)).astype(np.float32)
        X_filledWith0 = np.nan_to_num(X)
        empirical_mean_for_GRUD = np.sum(mask * X_filledWith0, axis=(0, 1)) / np.sum(mask, axis=(0, 1))
        data_dict = {
            'X': X,
            'empirical_mean_for_GRUD': empirical_mean_for_GRUD,
        }
    else:
        # if this is val/test set, then we need to add artificial missing values right now,
        # because we need they are fixed
        X = X.reshape(-1)
        indices_for_holdout = random_mask(X, artificial_missing_rate)
        X_hat = np.copy(X)
        X_hat[indices_for_holdout] = np.nan  # X_hat contains artificial missing values
        missing_mask = (~np.isnan(X_hat)).astype(np.float32)
        # indicating_mask contains masks indicating artificial missing values
        indicating_mask = ((~np.isnan(X_hat)) ^ (~np.isnan(X))).astype(np.float32)

        data_dict = {
            'X': X.reshape([sample_num, seq_len, feature_num]),
            'X_hat': X_hat.reshape([sample_num, seq_len, feature_num]),
            'missing_mask': missing_mask.reshape([sample_num, seq_len, feature_num]),
            'indicating_mask': indicating_mask.reshape([sample_num, seq_len, feature_num])
        }

    return data_dict

def make_datasets_dict(raw_df,train_set_ids,val_set_ids,test_set_ids, artificial_missing_rate = 0.1):
    timepoints_per_patient = pd.Series(raw_df.index.get_level_values(0).values).value_counts()
    smallest_window        = timepoints_per_patient.min()
    df                     = raw_df.loc[(slice(None), slice(1, smallest_window)), :]
    train_set = df.loc[train_set_ids]
    val_set   = df.loc[val_set_ids]
    test_set  = df.loc[test_set_ids]
    assert all(train_set.index.get_level_values(0).unique().to_list() == train_set_ids)
    assert all(val_set.index.get_level_values(0).unique().to_list() == val_set_ids)
    assert all(test_set.index.get_level_values(0).unique().to_list() == test_set_ids)
    
    train_set_X, feature_names = process_each_set(train_set, smallest_window)
    val_set_X,  _              = process_each_set(val_set, smallest_window)
    test_set_X,  _             = process_each_set(test_set, smallest_window)
    
    train_set_dict = add_artificial_mask(train_set_X, artificial_missing_rate, 'train')
    val_set_dict   = add_artificial_mask(val_set_X, artificial_missing_rate, 'val')
    test_set_dict  = add_artificial_mask(test_set_X, artificial_missing_rate, 'test')
    
    return {'train': train_set_dict,'val': val_set_dict,'test': test_set_dict}

def saving_into_h5(saving_dir, data_dict, classification_dataset):
    """ Save data into h5 file.
    Parameters
    ----------
    saving_dir: path of saving dir
    data_dict: data dictionary containing train/val/test sets
    classification_dataset: boolean, if this is a classification dataset
    """

    def save_each_set(handle, name, data):
        single_set = handle.create_group(name)
        if classification_dataset:
            single_set.create_dataset('labels', data=data['labels'].astype(int))
        single_set.create_dataset('X', data=data['X'].astype(np.float32))
        if name in ['val', 'test']:
            single_set.create_dataset('X_hat', data=data['X_hat'].astype(np.float32))
            single_set.create_dataset('missing_mask', data=data['missing_mask'].astype(np.float32))
            single_set.create_dataset('indicating_mask', data=data['indicating_mask'].astype(np.float32))
            
        if name in ['train_set_df']:
            single_set.create_dataset('df', data=data['train_set_df'])


    saving_path = os.path.join(saving_dir, 'datasets.h5')
    with h5py.File(saving_path, 'w') as hf:
        hf.create_dataset('empirical_mean_for_GRUD', data=data_dict['train']['empirical_mean_for_GRUD'])
        save_each_set(hf, 'train', data_dict['train'])
        save_each_set(hf, 'val', data_dict['val'])
        save_each_set(hf, 'test', data_dict['test'])
        
        
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LoadDataset(Dataset):
    def __init__(self, file_path, seq_len, feature_num, model_type):
        super(LoadDataset, self).__init__()
        self.file_path = file_path
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.model_type = model_type

class LoadValTestDataset(LoadDataset):
    """Loading process of val or test set"""

    def __init__(self, file_path, set_name, seq_len, feature_num, model_type):
        super(LoadValTestDataset, self).__init__(file_path, seq_len, feature_num, model_type)
        with h5py.File(self.file_path, 'r') as hf:  # read data from h5 file
            self.X = hf[set_name]['X'][:]
            self.X_hat = hf[set_name]['X_hat'][:]
            self.missing_mask = hf[set_name]['missing_mask'][:]
            self.indicating_mask = hf[set_name]['indicating_mask'][:]

        # fill missing values with 0
        self.X = np.nan_to_num(self.X)
        self.X_hat = np.nan_to_num(self.X_hat)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.model_type in ['Transformer', 'SAITS']:
            sample = (
                torch.tensor(idx),
                torch.from_numpy(self.X_hat[idx].astype('float32')),
                torch.from_numpy(self.missing_mask[idx].astype('float32')),
                torch.from_numpy(self.X[idx].astype('float32')),
                torch.from_numpy(self.indicating_mask[idx].astype('float32')),
            )
        else:
            assert ValueError, f'Error model type: {self.model_type}'
        return sample


class LoadTrainDataset(LoadDataset):
    """Loading process of train set"""

    def __init__(self, file_path, seq_len, feature_num, model_type, masked_imputation_task):
        super(LoadTrainDataset, self).__init__(file_path, seq_len, feature_num, model_type)
        self.masked_imputation_task = masked_imputation_task
        if masked_imputation_task:
            self.artificial_missing_rate = 0.1
            assert 0 < self.artificial_missing_rate < 1, 'artificial_missing_rate should be greater than 0 and less than 1'

        with h5py.File(self.file_path, 'r') as hf:  # read data from h5 file
            self.X = hf['train']['X'][:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        if self.masked_imputation_task:
            X = X.reshape(-1)
            indices = np.where(~np.isnan(X))[0].tolist()
            indices = np.random.choice(indices, round(len(indices) * self.artificial_missing_rate))
            X_hat = np.copy(X)
            X_hat[indices] = np.nan  # mask values selected by indices
            missing_mask = (~np.isnan(X_hat)).astype(np.float32)
            indicating_mask = ((~np.isnan(X)) ^ (~np.isnan(X_hat))).astype(np.float32)
            X = np.nan_to_num(X)
            X_hat = np.nan_to_num(X_hat)
            # reshape into time series
            X = X.reshape(self.seq_len, self.feature_num)
            X_hat = X_hat.reshape(self.seq_len, self.feature_num)
            missing_mask = missing_mask.reshape(self.seq_len, self.feature_num)
            indicating_mask = indicating_mask.reshape(self.seq_len, self.feature_num)

            if self.model_type in ['Transformer', 'SAITS']:
                sample = (
                    torch.tensor(idx),
                    torch.from_numpy(X_hat.astype('float32')),
                    torch.from_numpy(missing_mask.astype('float32')),
                    torch.from_numpy(X.astype('float32')),
                    torch.from_numpy(indicating_mask.astype('float32')),
                )
            else:
                assert ValueError, f'Error model type: {self.model_type}'
        else:
            # if training without masked imputation task, then there is no need to artificially mask out observed values
            missing_mask = (~np.isnan(X)).astype(np.float32)
            X = np.nan_to_num(X)
            if self.model_type in ['Transformer', 'SAITS']:
                sample = (
                    torch.tensor(idx),
                    torch.from_numpy(X.astype('float32')),
                    torch.from_numpy(missing_mask.astype('float32')),
                )
            else:
                assert ValueError, f'Error model type: {self.model_type}'
        return sample


class UnifiedDataLoader:
    def __init__(self, dataset_path, seq_len, feature_num, model_type, batch_size, num_workers=4,
                 masked_imputation_task=False):
        """
        dataset_path: path of directory storing h5 dataset;
        seq_len: sequence length, i.e. time steps;
        feature_num: num of features, i.e. feature dimensionality;
        batch_size: size of mini batch;
        num_workers: num of subprocesses for data loading;
        model_type: model type, determine returned values;
        masked_imputation_task: whether to return data for masked imputation task, only for training/validation sets;
        """
        self.dataset_path = os.path.join(dataset_path, 'datasets.h5')
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        self.masked_imputation_task = masked_imputation_task

        self.train_dataset, self.train_loader, self.train_set_size = None, None, None
        self.val_dataset, self.val_loader, self.val_set_size = None, None, None
        self.test_dataset, self.test_loader, self.test_set_size = None, None, None



        
    def get_train_val_dataloader(self):
        self.train_dataset = LoadTrainDataset(self.dataset_path, self.seq_len, self.feature_num, self.model_type,
                                              self.masked_imputation_task)
        self.val_dataset = LoadValTestDataset(self.dataset_path, 'val', self.seq_len, self.feature_num, self.model_type)
        self.train_set_size = self.train_dataset.__len__()
        self.val_set_size = self.val_dataset.__len__()
        
        
        train_sampler = torch.utils.data.RandomSampler(self.train_dataset , replacement=True, num_samples=200)
        val_sampler   = torch.utils.data.RandomSampler(self.val_dataset , replacement=True, num_samples=50)
        
        self.train_loader = DataLoader(self.train_dataset, self.batch_size,  num_workers=self.num_workers, sampler=train_sampler, drop_last=True)
        self.val_loader   = DataLoader(self.val_dataset, self.batch_size, num_workers=self.num_workers, sampler=val_sampler, drop_last=True)
        return self.train_loader, self.val_loader
    
from SAITS_functions import * 
import argparse
import os 
    
def get_args(seq_len = 6, feature_num  = 18, batch_size   = 4):

    #MODEL_DICT = { # Self-Attention (SA) based
        #'Transformer': TransformerEncoder, 
        #'SAITS': SAITS,
        # RNN based
        #'BRITS': BRITS, 'MRNN': MRNN,}

    #OPTIMIZER = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='path of config file')
    parser.add_argument('--test_mode', dest='test_mode', action='store_true', help='test mode to test saved model')
    parser.add_argument('--param_searching_mode', dest='param_searching_mode', action='store_true',
                        help='use NNI to help search hyper parameters')
    args = parser.parse_args('')


    args.config_path = f'{os.getcwd()}/configs/PhysioNet2012_SAITS_best.ini'



    args.test_mode = False
    args.param_searching_mode = False
    assert os.path.exists(args.config_path), f'Given config file "{args.config_path}" does not exists'
    # load settings from config file
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config_path)
    args = read_arguments(args, cfg)
    if args.model_type in ['Transformer', 'SAITS']:  # if SA-based model
            args.input_with_mask = cfg.getboolean('model', 'input_with_mask')
            args.n_groups = cfg.getint('model', 'n_groups')
            args.n_group_inner_layers = cfg.getint('model', 'n_group_inner_layers')
            args.param_sharing_strategy = cfg.get('model', 'param_sharing_strategy')
            assert args.param_sharing_strategy in ['inner_group', 'between_group'], \
                'only "inner_group"/"between_group" sharing'
            args.d_model = cfg.getint('model', 'd_model')
            args.d_inner = cfg.getint('model', 'd_inner')
            args.n_head = cfg.getint('model', 'n_head')
            args.d_k = cfg.getint('model', 'd_k')
            args.d_v = cfg.getint('model', 'd_v')
            args.dropout = cfg.getfloat('model', 'dropout')
            args.diagonal_attention_mask = cfg.getboolean('model', 'diagonal_attention_mask')

            dict_args = vars(args)

    # parameter insurance
    assert args.model_saving_strategy.lower() in ['all', 'best', 'none'], 'model saving strategy must be all/best/none'
    if args.model_saving_strategy.lower() == 'none':
        args.model_saving_strategy = False
    #assert args.optimizer_type in OPTIMIZER.keys(), \
    #    f'optimizer type should be in {OPTIMIZER.keys()}, but get{args.optimizer_type}'
    #assert args.device in ['cpu', 'cuda'], 'device should be cpu or cuda'



    time_now = datetime.now().__format__('%Y-%m-%d_T%H:%M:%S')
    args.model_saving, args.log_saving = check_saving_dir_for_model(args, time_now)
    #logger = setup_logger(args.log_saving + '_' + time_now, 'w')
    #logger.info(f'args: {args}');
    #logger.info(f'Config file path: {args.config_path}');
    #logger.info(f'Model name: {args.model_name}');


    args.dataset_path = f"{os.getcwd()}/data"
    args.num_workers  = os.cpu_count()
    args.seq_len      = seq_len#6
    args.feature_num  = feature_num#18
    args.batch_size   = batch_size#4
    args.model_type = 'SAITS' 
    args.num_workers =  os.cpu_count()
    args.device      =  'cuda'
    
    model_args = {
                'device': args.device, 'MIT': args.MIT,
                # imputer args
                'n_groups': dict_args['n_groups'], 'n_group_inner_layers': args.n_group_inner_layers,
                'd_time': args.seq_len, 'd_feature': args.feature_num, 'dropout': dict_args['dropout'],
                'd_model': dict_args['d_model'], 'd_inner': dict_args['d_inner'], 'n_head': dict_args['n_head'],
                'd_k': dict_args['d_k'], 'd_v': dict_args['d_v'],
                'input_with_mask': args.input_with_mask,
                'diagonal_attention_mask': args.diagonal_attention_mask,
                'param_sharing_strategy': args.param_sharing_strategy,
            }
    return args, model_args

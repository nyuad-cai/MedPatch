import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class EHRdataset(Dataset):
    def __init__(self, args, discretizer, normalizer, listfile, dataset_dir, return_names=True, period_length=48.0):
        self.return_names = return_names
        self.discretizer = discretizer
        self.args=args
        self.normalizer = normalizer
        self._period_length = period_length
        self._dataset_dir = dataset_dir
        listfile_path = listfile
        print(self._dataset_dir)
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self.CLASSES = self._listfile_header.strip().split(',')[3:]
        self._data = self._data[1:]
        #print(self._data[12])
        self._data = [line.split(',') for line in self._data]
        if self.args.task=='length-of-stay' or self.args.task=='decompensation':
            self.data_map = {
            (mas[0],float(mas[1])): {
                'labels': list(map(float, mas[3:])),
                'stay_id': float(mas[2]),
                'time': float(mas[1]),
                }
                for mas in self._data
                
            }
        else:
            self.data_map = {
            mas[0]: {
                'labels': list(map(float, mas[3:])),
                'stay_id': float(mas[2]),
                'time': float(mas[1]),
                }
                for mas in self._data
        }
        self.names = list(self.data_map.keys())
        self.times= None
        #print(self.data_map[12])
        # import pdb; pdb.set_trace()
        # self._data = [(line_[0], float(line_[1]), line_[2], float(line_[3])  ) for line_ in self._data]
        # self.names = [x[0] for x in self.data_map]
        # print(self.names[0:10])
    def _read_timeseries(self, ts_filename, time_bound=None):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                if time_bound is not None:
                    t = float(mas[0])
                    if t > time_bound + 1e-6:
                        break
                ret.append(np.array(mas))
        return (np.stack(ret), header)
    # def _read_timeseries(self, ts_filename, lower_bound, upper_bound):
    #     ret = []
    #     with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
    #         header = tsfile.readline().strip().split(',')
    #         assert header[0] == "Hours"
    #         for line in tsfile:
    #             mas = line.strip().split(',')
    #             t = float(mas[0])
    #             #print("upper_bound:",upper_bound)
    #             if t < lower_bound:
    #                 continue
    #             elif (t> lower_bound) & (t <upper_bound) :
    #                 ret.append(np.array(mas))
    #             elif t > upper_bound:
    #                 break
        try:
            #print(np.stack(ret))
            print(f"Length of time series in _read_timeseries: {len(ret)}")  # Debug statement
            return (np.stack(ret), header)
        except ValueError:
            #print("exception in read_timeseries")
            ret = ([['0.11666666666666667', '', '', '', '', '', '', '', '', '109', '',
                     '', '', '30', '', '', '', ''],
                    ['0.16666666666666666', '', '61.0', '', '', '', '', '', '', '109',
                    '', '64', '97.0', '29', '74.0', '', '', '']])
            #print(ts_filename, lower_bound, upper_bound)
            return (np.stack(ret), header)
        
    def read_by_file_name(self, index, time, time_bound=None):
        #print("index", index)
        if self.args.task=='length-of-stay' or self.args.task=='decompensation':
            t = self.data_map[(index,time)]['time'] if time_bound is None else time_bound
            y = self.data_map[(index,time)]['labels']
            stay_id = self.data_map[(index,time)]['stay_id']
            (X, header) = self._read_timeseries(index, time_bound=time_bound)
        else:
            #print("elsing")
            t = self.data_map[index]['time'] if time_bound is None else time_bound
            y = self.data_map[index]['labels']
            stay_id = self.data_map[index]['stay_id']
            (X, header) = self._read_timeseries(index, time_bound=time_bound)
        return {"X": X,
                "t": t,
                "y": y,
                'stay_id': stay_id,
                "header": header,
                "name": index}
    
    
    # def read_by_file_name(self, index, time, lower_bound, upper_bound):
    #     if self.args.task=='length-of-stay' or self.args.task=='decompensation':
    #         t = self.data_map[(index,time)]['time']
    #         y = self.data_map[(index,time)]['labels']
    #         stay_id = self.data_map[(index,time)]['stay_id']
    #         (X, header) = self._read_timeseries(index, lower_bound=lower_bound, upper_bound=time)
    #     else:
    #         #print("this upper:",upper_bound)
    #         t = self.data_map[index]['time']
    #         y = self.data_map[index]['labels']
    #         stay_id = self.data_map[index]['stay_id']
    #         (X, header) = self._read_timeseries(index, lower_bound=lower_bound, upper_bound=upper_bound)
    #     return {"X": X,
    #             "t": t,
    #             "y": y,
    #             'stay_id': stay_id,
    #             "header": header,
    #             "name": index}
    
    def get_decomp_los(self, index, time_bound=None):
        # name = self._data[index][0]
        # time_bound = self._data[index][1]
        # ys = self._data[index][3]
        # (data, header) = self._read_timeseries(index, time_bound=time_bound)
        # data = self.discretizer.transform(data, end=time_bound)[0]
        # if (self.normalizer is not None):
        #     data = self.normalizer.transform(data)
        # ys = np.array(ys, dtype=np.int32) if len(ys) > 1 else np.array(ys, dtype=np.int32)[0]
        # return data, ys
        # data, ys =
        return self.__getitem__(index, time_bound)
    def __getitem__(self, tuplee,  time_bound=None):
        if self.args.task=='length-of-stay' or self.args.task=='decompensation':
            time = tuplee[1]
            index = tuplee[0]
        else:
            #print("normal")
            index = tuplee
            if isinstance(index, int):
                index = self.names[index]
            time = None
        ret = self.read_by_file_name(index, time, time_bound)
        data = ret["X"]
        ts = ret["t"] if ret['t'] > 0.0 else self._period_length
        ys = ret["y"]
        # print("this is ys" , ys)
        names = ret["name"]
        data = self.discretizer.transform(data, end=ts)[0]
        if (self.normalizer is not None):
            data = self.normalizer.transform(data)
        if 'length-of-stay' in self._dataset_dir:
            ys = np.array(ys, dtype=np.float32) if len(ys) > 1 else np.array(ys, dtype=np.float32)[0]
        else:
            ys = np.array(ys, dtype=np.int32) if len(ys) > 1 else np.array(ys, dtype=np.int32)[0]
        return data, ys
    
    # def __getitem__(self, item_args, lower, upper):
    #     if self.args.task=='length-of-stay' or self.args.task=='decompensation':
    #         time = item_args[1]
    #         index = item_args[0]
    #     else:
    #         index = item_args
    #         if isinstance(index, int):
    #             index = self.names[index]
    #         time = None
    #     #print("just upper:", upper)
    #     ret = self.read_by_file_name(index, time, lower, upper)
    #     data = ret["X"]
    #     ts = data.shape[0]
    #     # print("Times included", ts) #ret["t"] if ret['t'] > 0.0 else self._period_length
    #     print(f"Shape of data in __getitem__: {data.shape}")
    #     ys = ret["y"]
    #     names = ret["name"]
    #     data = self.discretizer.transform(data, end=ts)[0]
    #     if (self.normalizer is not None):
    #         data = self.normalizer.transform(data)
    #     if 'length-of-stay' in self._dataset_dir:
    #         ys = np.array(ys, dtype=np.float32) if len(ys) > 1 else np.array(ys, dtype=np.float32)[0]
    #     else:
    #         ys = np.array(ys, dtype=np.int32) if len(ys) > 1 else np.array(ys, dtype=np.int32)[0]
    #     return data, ys
        
    def __len__(self):
        return len(self.names)
    def get_datasets(discretizer, normalizer, args):
        train_ds = EHRdataset(discretizer, normalizer, f'{args.ehr_data_dir}/{args.task}/train_listfile.csv', os.path.join(args.ehr_data_dir, f'{args.task}/train'))
        val_ds = EHRdataset(discretizer, normalizer, f'{args.ehr_data_dir}/{args.task}/val_listfile.csv', os.path.join(args.ehr_data_dir, f'{args.task}/train'))
        test_ds = EHRdataset(discretizer, normalizer, f'{args.ehr_data_dir}/{args.task}/test_listfile.csv', os.path.join(args.ehr_data_dir, f'{args.task}/test'))
        return train_ds, val_ds, test_ds
        
    def get_data_loader(discretizer, normalizer, dataset_dir, batch_size):
        train_ds, val_ds, test_ds = get_datasets(discretizer, normalizer, dataset_dir)
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16)
        val_dl = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16)
        return train_dl, val_dl
    def my_collate(batch):
        x = [item[0] for item in batch]
        x, seq_length = pad_zeros(x)
        targets = np.array([item[1] for item in batch])
        return [x, targets, seq_length]
    def pad_zeros(arr, min_length=None):
        dtype = arr[0].dtype
        seq_length = [x.shape[0] for x in arr]
        max_len = max(seq_length)
        ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in arr]
        if (min_length is not None) and ret[0].shape[0] < min_length:
            ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
                   for x in ret]
        return np.array(ret), seq_length


def get_datasets(discretizer, normalizer, args):
    print(f'{args.ehr_data_dir}/{args.task}/train_listfile.csv')
    train_ds = EHRdataset(args,discretizer, normalizer, f'{args.ehr_data_dir}/{args.task}/train_listfile.csv', os.path.join(args.ehr_data_dir, f'{args.task}/train'))
    val_ds = EHRdataset(args,discretizer, normalizer, f'{args.ehr_data_dir}/{args.task}/val_listfile.csv', os.path.join(args.ehr_data_dir, f'{args.task}/train'))
    test_ds = EHRdataset(args,discretizer, normalizer, f'{args.ehr_data_dir}/{args.task}/test_listfile.csv', os.path.join(args.ehr_data_dir, f'{args.task}/test'))
    return train_ds, val_ds, test_ds

def get_data_loader(discretizer, normalizer, dataset_dir, batch_size):
    train_ds, val_ds, test_ds = get_datasets(discretizer, normalizer, dataset_dir)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16)

    return train_dl, val_dl
        
def my_collate(batch):
    x = [item[0] for item in batch]
    x, seq_length = pad_zeros(x)
    targets = np.array([item[1] for item in batch])
    return [x, targets, seq_length]

def pad_zeros(arr, min_length=None):

    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret), seq_length
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re
from trainers.fusion_trainer import FusionTrainer
from trainers.frozen_trainer import FrozenTrainer
from trainers.mmtm_trainer import MMTMTrainer
from trainers.daft_trainer import DAFTTrainer

from trainers.unimodal_trainer import UnimodalTrainer
from trainers.staged_Trainer import StagedFusionTrainer
from trainers.DHF_trainer import DHFTrainer
from trainers.lstm_trainer import LSTMFusionTrainer

from trainers.ensemble_trainer import EnsembleFusionTrainer
from trainers.hierarchical_ensemble_trainer import HierEnsembleFusionTrainer
from trainers.ensemble_average import EnsembleAverageTrainer
from trainers.ensemble_late import EnsembleLateTrainer
from trainers.triple_ensemble import TripleFusionTrainer
from trainers.weighted_trainer import WeightedAverageFusionTrainer
from trainers.weighted_ensemble import WeightedEnsembleLateTrainer
from trainers.MSMA_trainer import MSMA_Trainer
from trainers.Calibration import calibration


from ehr_utils.preprocessing import Discretizer, Normalizer
from datasets.ehr_dataset import get_datasets
from datasets.cxr_dataset import get_cxr_datasets
from datasets.DataFusion import load_cxr_ehr_rr_dn
from pathlib import Path
import torch
import random

from arguments import args_parser

parser = args_parser()
# add more arguments here ...
args = parser.parse_args()
print(args)

if args.missing_token is not None:
    from trainers.fusion_tokens_trainer import FusionTokensTrainer as FusionTrainer
    
path = Path(args.save_dir)
path.mkdir(parents=True, exist_ok=True)

seed = 1002
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)

# Set PyTorch to deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def read_timeseries(args):
    path = f'{args.ehr_data_dir}/{args.task}/train/14991576_episode3_timeseries.csv'
    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    return np.stack(ret)
    

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')


discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'normalizers/ph_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

print("getting data")

ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(discretizer, normalizer, args)

cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(args)

train_dl, val_dl, test_dl = load_cxr_ehr_rr_dn(args, ehr_train_ds, ehr_val_ds, cxr_train_ds, cxr_val_ds, ehr_test_ds, cxr_test_ds)

print("data gone")

with open(f"{args.save_dir}/args.txt", 'w') as results_file:
    for arg in vars(args): 
        print(f"  {arg:<40}: {getattr(args, arg)}")
        results_file.write(f"  {arg:<40}: {getattr(args, arg)}\n")


if args.calibration == 'calibrate':
    trainer = calibration(train_dl, 
        val_dl, 
        args,
        test_dl)
elif args.fusion_type == 'relevancy-based-hierarchical' or args.fusion_type == 'predefined-hierarchical':
    trainer = DHFTrainer(
        train_dl, 
        val_dl, 
        args,
        test_dl
        )
elif args.fusion_type == 'triple-early' or args.fusion_type == 'triple-joint' or args.fusion_type == 'triple-late':
    trainer = TripleFusionTrainer(
        train_dl, 
        val_dl, 
        args,
        test_dl
        )
elif args.fusion_type == 'weighted':
    trainer = WeightedAverageFusionTrainer(
        train_dl, 
        val_dl, 
        args,
        test_dl
        )
elif args.fusion_type == 'mmtm':
    trainer = MMTMTrainer(
        train_dl, 
        val_dl, 
        args,
        test_dl=test_dl
        )
elif args.fusion_type == 'daft':
    trainer = DAFTTrainer(train_dl, 
        val_dl, 
        args,
        test_dl=test_dl)
elif 'temp_c-unimodal' in args.fusion_type:
    trainer = calibration(train_dl, 
        val_dl, 
        args,
        test_dl)
else:
    print("running")
    trainer = MSMA_Trainer(
        train_dl, 
        val_dl, 
        args,
        test_dl
        )
if args.mode == 'train':
    print("==> training")
    trainer.train()
    trainer.eval()
elif args.mode == 'eval':
    trainer.eval()
else:
    raise ValueError("not Implementation for args.mode")

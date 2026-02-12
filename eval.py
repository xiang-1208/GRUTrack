import yaml, argparse, time, os, json, multiprocessing
from dataloader.nusc_loader_self import NuScenesloader
from tracking.nusc_tracker import Tracker
from tqdm import tqdm
import pdb
import torch.nn.init as init
from KalmanNet_nn import KalmanNetNN

import torch

import warnings
warnings.filterwarnings("ignore")

from utils.io import load_file

parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--process', type=int, default=1)
# paths
localtime = ''.join(time.asctime(time.localtime(time.time())).split(' '))
parser.add_argument('--nusc_path', type=str, default='/data/wxx/nuscenes/v1.0-trainval/')
parser.add_argument('--config_path', type=str, default='config/nusc_config.yaml')
parser.add_argument('--detection_path', type=str, default='/home/wxx/Poly-MOT/infos_train_10sweeps_withvelo_filter_True.json')
parser.add_argument('--first_token_path', type=str, default='data/utils/first_token_table/train/nusc_first_token.json')
parser.add_argument('--final_token_path', type=str, default='data/utils/final_token_table/train/nusc_final_token.json')
parser.add_argument('--token_path', type=str, default='/home/wxx/Poly-MOT/data/utils/token_table/train/nusc_train_token.json')
parser.add_argument('--model', type=str, default='/home/wxx/Poly-MOT/model/TueJan3011:25:322024/step_4800.pth')
parser.add_argument('--result_path', type=str, default='result/' + localtime)
parser.add_argument('--eval_path', type=str, default='eval_result2/')
args = parser.parse_args()

path = "/home/wxx/Poly-MOT/data/utils/token_table/train/nusc_train_token.json"
seq_token = load_file(path)[0:10]
flat_list = []
for sublist in seq_token:
    flat_list.extend(sublist)

from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.common.config import config_factory as track_configs
from nuscenes.eval.tracking.data_classes import TrackingMetrics, TrackingMetricDataList, TrackingConfig, TrackingBox, \
    TrackingMetricData
from typing import Tuple, List, Dict, Any


class TrackingEval_self(TrackingEval):
    def __init__(self,
                config: TrackingConfig,
                result_path: str,
                eval_set: str,
                output_dir: str,
                nusc_version: str,
                nusc_dataroot: str,
                verbose: bool = True,
                render_classes: List[str] = None,
                tokens:list = None):
        super().__init__(config, result_path, eval_set, output_dir, nusc_version, nusc_dataroot, verbose,render_classes,tokens)


        print ("!!!")

def eval(result_path, eval_path, nusc_path):
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.eval.common.config import config_factory as track_configs
    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval_self(
        config=cfg,
        result_path=result_path,
        eval_set="train",
        output_dir=eval_path,
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=nusc_path,
        tokens = flat_list,
    )
    print("result in " + result_path)
    metrics_summary = nusc_eval.main()


if __name__ == "__main__":
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(args.eval_path, exist_ok=True)

    # load and keep config
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.Loader)
    valid_cfg = config
    json.dump(valid_cfg, open(args.eval_path + "/config.json", "w"))
    print('writing config in folder: ' + os.path.abspath(args.eval_path))

    # load dataloader
    nusc_loader = NuScenesloader(args.detection_path,
                                 args.first_token_path,
                                 args.final_token_path,
                                 args.token_path,
                                 config,
                                 suffle = False,
                                 token=flat_list)
    print('writing result in folder: ' + os.path.abspath(args.result_path))

    # eval result
    eval(os.path.join(args.result_path, 'results.json'), args.eval_path, args.nusc_path)

    print (args.model)
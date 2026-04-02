import yaml
import argparse
import time
import os
import json
import multiprocessing
from dataloader.nusc_loader_self import NuScenesloader
from tracking.nusc_tracker import Tracker
from tqdm import tqdm
import torch.nn.init as init
from KalmanNet_nn import KalmanNetNN
from torch.utils.tensorboard import SummaryWriter
import torch
import logging
import datetime
from utils.io import load_file
from multiprocessing import Process
import warnings

warnings.filterwarnings("ignore")

from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.common.config import config_factory as track_configs
from nuscenes.eval.tracking.data_classes import TrackingConfig, TrackingBox
from typing import List


def parse_config():
    parser = argparse.ArgumentParser()
    localtime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    # running configurations
    parser.add_argument('--process', type=int, default=1)
    parser.add_argument('--config_path', type=str, default='config/nusc_config.yaml')

    # paths
    parser.add_argument('--train_path', type=str, default='data/utils/train')
    parser.add_argument('--val_path', type=str, default='data/utils/val')
    parser.add_argument('--nusc_path', type=str, default='data/nuscenes/v1.0-trainval/')
    parser.add_argument('--output_path', type=str, default='output/' + localtime)
    parser.add_argument('--tensorboard_path', type=str, default='output/tensorboard/' + localtime)
    parser.add_argument('--model', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()

    # Build derived paths
    utils_path = args.train_path
    args.detection_path = f"{utils_path}/detector/infos_train_10sweeps_withvelo_filter_True.json"
    args.first_token_path = f"{utils_path}/nusc_first_token.json"
    args.final_token_path = f"{utils_path}/nusc_final_token.json"
    args.token_path = f"{utils_path}/nusc_token.json"
    args.gt_path = f"{utils_path}/gt_train.json"

    val_path = args.val_path
    args.detection_val_path = f"{val_path}/detector/infos_val_10sweeps_withvelo_filter_True.json"
    args.first_val_token_path = f"{val_path}/nusc_first_token.json"
    args.final_val_token_path = f"{val_path}/nusc_final_token.json"
    args.token_val_path = f"{val_path}/nusc_token.json"

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    args.log_file = f"{output_path}/train.txt"
    args.result_path = f"{output_path}/result/"
    args.eval_path = f"{output_path}/eval_result/"

    return args

def main():
    args = parse_config()

    # 创建信号队列和结果队列
    signal_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    logger = create_logger(args.log_file)

    # load and keep config
    os.makedirs(args.eval_path, exist_ok=True)
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.Loader)
    valid_cfg = config
    json.dump(valid_cfg, open(args.eval_path + "/config.json", "w"))
    logger.info('writing config in folder: ' + os.path.abspath(args.eval_path))

    # load dataloader
    nusc_loader_train = NuScenesloader(args.detection_path,
                                 args.first_token_path,
                                 args.final_token_path,
                                 args.token_path,
                                 config,
                                 suffle = True,
                                 logger = logger)

    # load val dataloader
    seq_token = load_file(args.token_val_path)  #[0:10]
    flat_list = []
    for sublist in seq_token:
        flat_list.extend(sublist)
    nusc_loader_val = NuScenesloader(args.detection_val_path,
                                 args.first_val_token_path,
                                 args.final_val_token_path,
                                 args.token_val_path,
                                 config,
                                 suffle = False,
                                 logger = logger,
                                 token=flat_list)

    KNet_model_CTRA = KalmanNetNN(config=nusc_loader_train.config)
    KNet_model_CTRA.NNBuild(MD=9, SD=10)
    KNet_model_CTRA.cuda()

    if args.model:
        KNet_model_CTRA.load_state_dict(torch.load(args.model))

    for name, param in KNet_model_CTRA.named_parameters():
        if 'weight' in name:
            if len(param.shape) < 2:
                init.xavier_normal_(param.unsqueeze(0))
            else:
                init.xavier_normal_(param)
        elif 'bias' in name:
            init.constant_(param, 0)

    nusc_tracker = Tracker(config=nusc_loader_train.config, model=KNet_model_CTRA)
    gt_boxes = load_file(args.gt_path, logger)

    num_step = 0

    # save init model
    PATH = f"{args.output_path}/model/"
    os.makedirs(PATH, exist_ok=True)
    torch.save(KNet_model_CTRA.state_dict(), f"{PATH}/step_{num_step}.pth")


    total_epochs = args.epochs

    result_process = multiprocessing.Process(target=result_function, args=(args,signal_queue, result_queue))
    result_process.start()


    for i in range(total_epochs):
        nusc_loader_train.suffle_list()
        for i_seq, frame_data in enumerate(nusc_loader_train):
            sample_token = frame_data['sample_token']
            gt_data = gt_boxes[sample_token]

            loss = [0, 0]
            if nusc_tracker.tracking(frame_data, gt_data, logger, loss, i):
                result_queue.put([num_step, loss[0], loss[1]])
                signal_queue.put("loss")

                if num_step % 100 == 0:
                    PATH = f"{args.output_path}/model/step_{num_step}.pth"
                    torch.save(KNet_model_CTRA.state_dict(), PATH)
                    Process(target=test, args=(args, nusc_loader_val, flat_list, PATH, num_step, signal_queue, result_queue)).start()

                logger.info("===========================================================================")
                logger.info("%d/%d epochs, %d/%d frames", i+1, total_epochs, i_seq+1, len(nusc_loader_train))
                logger.info("frame_id: %d, seq_id: %d", frame_data['frame_id'], frame_data['seq_id'])
                num_step += 1
                assert (num_step == frame_data['seq_id'])

    result_process.terminate()


def result_function(args, signal_queue, result_queue):
    os.makedirs(args.tensorboard_path, exist_ok=True)
    writer = SummaryWriter(args.tensorboard_path)
    while True:
        signal = signal_queue.get()

        if signal == "Signal":
            result = result_queue.get()
            writer.add_scalar('amota', result[1], result[0])
            writer.add_scalar('amotp', result[2], result[0])
            writer.add_scalar('ids', result[3], result[0])
        if signal == "loss":
            result = result_queue.get()
            writer.add_scalar('loss', result[1], result[0])
            writer.add_scalar('lr', result[2], result[0])

def test(args, nusc_loader_val, flat_list, path, num_step, signal_queue, result_queue):
    os.makedirs(f"{args.result_path}{num_step}", exist_ok=True)
    os.makedirs(f"{args.eval_path}{num_step}", exist_ok=True)

    val(f"{args.result_path}{num_step}", nusc_loader_val, path)
    [amota, amotp, ids] = eval(
        os.path.join(f"{args.result_path}{num_step}", 'results.json'),
        f"{args.eval_path}{num_step}",
        args.nusc_path,
        flat_list
    )

    result_queue.put([num_step, amota, amotp, ids])
    signal_queue.put("Signal")


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
                tokens: list = None):
        super().__init__(config, result_path, eval_set, output_dir, nusc_version, nusc_dataroot, verbose, render_classes, tokens)


def eval(result_path, eval_path, nusc_path, flat_list):
    from nuscenes.eval.common.config import config_factory as track_configs
    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval_self(
        config=cfg,
        result_path=result_path,
        eval_set="val",
        output_dir=eval_path,
        verbose=False,
        nusc_version="v1.0-trainval",
        nusc_dataroot=nusc_path,
        tokens=flat_list,
    )
    metrics_summary = nusc_eval.main()

    return [metrics_summary['amota'], metrics_summary['amotp'], metrics_summary['ids']]

def val(result_path, nusc_loader, path):
    result = {
        "results": {},
        "meta": {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
    }

    KNet_model_CTRA = KalmanNetNN(config=nusc_loader.config)
    KNet_model_CTRA.NNBuild(MD=9, SD=10)
    KNet_model_CTRA.cuda()
    KNet_model_CTRA.load_state_dict(torch.load(path))
    KNet_model_CTRA.eval()

    nusc_tracker = Tracker(config=nusc_loader.config, model=KNet_model_CTRA)

    for frame_data in nusc_loader:
        sample_token = frame_data['sample_token']

        if nusc_tracker.tracking(frame_data):
            break

        sample_results = []
        if 'no_val_track_result' not in frame_data:
            for predict_box in frame_data['box_track_res']:
                box_result = {
                    "sample_token": sample_token,
                    "translation": [float(predict_box.center[0]), float(predict_box.center[1]),
                                    float(predict_box.center[2])],
                    "size": [float(predict_box.wlh[0]), float(predict_box.wlh[1]), float(predict_box.wlh[2])],
                    "rotation": [float(predict_box.orientation[0]), float(predict_box.orientation[1]),
                                float(predict_box.orientation[2]), float(predict_box.orientation[3])],
                    "velocity": [float(predict_box.velocity[0]), float(predict_box.velocity[1])],
                    "tracking_id": str(predict_box.tracking_id),
                    "tracking_name": predict_box.name,
                    "tracking_score": predict_box.score,
                }
                sample_results.append(box_result.copy())

        if sample_token in result["results"]:
            result["results"][sample_token] = result["results"][sample_token] + sample_results
        else:
            result["results"][sample_token] = sample_results

    # sort track result by the tracking score
    for sample_token in result["results"].keys():
        confs = sorted(
            [(-d["tracking_score"], ind) for ind, d in enumerate(result["results"][sample_token])]
        )
        result["results"][sample_token] = [
            result["results"][sample_token][ind]
            for _, ind in confs[: min(500, len(confs))]
        ]

    json.dump(result, open(f"{result_path}/results.json", "w"))

def create_logger(log_file=None,log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()

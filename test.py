"""
GRUTrack Testing Script
Supports running tracking with optional evaluation on nuScenes dataset.

Modes:
    test    - Run tracking only (default)
    eval    - Run tracking + nuScenes evaluation
"""
import argparse
import time
import os
import json
import multiprocessing
from dataloader.nusc_loader_self import NuScenesloader
from tracking.nusc_tracker import Tracker
from tqdm import tqdm
from KalmanNet_nn import KalmanNetNN
import torch
import yaml
import warnings

warnings.filterwarnings("ignore")

from utils.io import load_file


def parse_args():
    parser = argparse.ArgumentParser(description='GRUTrack Testing')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['test', 'eval'],
                        help='Mode: "test" for tracking only, "eval" for tracking + evaluation')
    parser.add_argument('--process', type=int, default=1)
    parser.add_argument('--nusc_path', type=str, default='data/nuscenes/v1.0-trainval/')
    parser.add_argument('--config_path', type=str, default='config/nusc_config.yaml')
    parser.add_argument('--model', type=str, required=True)

    # Data paths - can be overridden with --data_split
    parser.add_argument('--detection_path', type=str, default=None)
    parser.add_argument('--first_token_path', type=str, default=None)
    parser.add_argument('--final_token_path', type=str, default=None)
    parser.add_argument('--token_path', type=str, default=None)

    # Data split preset (overrides individual path args)
    parser.add_argument('--data_split', type=str, default='test',
                        choices=['test', 'val', 'train'],
                        help='Preset data split: "test", "val", or "train"')

    parser.add_argument('--result_path', type=str, default=None)
    parser.add_argument('--eval_path', type=str, default='eval_result/')
    parser.add_argument('--eval_set', type=str, default='val',
                        choices=['train', 'val'],
                        help='Dataset split for evaluation')

    args = parser.parse_args()

    # Set default paths based on data_split
    localtime = '_'.join(time.asctime(time.localtime(time.time())).split(' '))
    split_base = f"data/utils/{args.data_split}"

    if args.detection_path is None:
        if args.data_split == 'val':
            args.detection_path = f"{split_base}/detector/infos_val_10sweeps_withvelo_filter_True.json"
        else:
            args.detection_path = f"{split_base}/detection.json"

    if args.first_token_path is None:
        args.first_token_path = f"{split_base}/nusc_first_token.json"
    if args.final_token_path is None:
        args.final_token_path = f"{split_base}/nusc_final_token.json"
    if args.token_path is None:
        args.token_path = f"{split_base}/nusc_token.json"
    if args.result_path is None:
        args.result_path = f"result/{localtime}"

    return args


class TrackingEval_self:
    """Wrapper for nuScenes TrackingEval to handle evaluation."""
    def __init__(self, config, result_path, eval_set, output_dir,
                 nusc_version, nusc_dataroot, verbose=True, tokens=None):
        from nuscenes.eval.tracking.evaluate import TrackingEval
        from nuscenes.eval.tracking.data_classes import TrackingConfig, TrackingBox
        from typing import List

        super().__init__()
        self.config = config
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.nusc_version = nusc_version
        self.nusc_dataroot = nusc_dataroot
        self.verbose = verbose
        self.tokens = tokens

    def run(self):
        from nuscenes.eval.tracking.evaluate import TrackingEval
        from nuscenes.eval.common.config import config_factory as track_configs
        from nuscenes.eval.tracking.data_classes import TrackingConfig, TrackingBox
        from typing import List

        cfg = track_configs("tracking_nips_2019")
        nusc_eval = TrackingEval(
            config=cfg,
            result_path=self.result_path,
            eval_set=self.eval_set,
            output_dir=self.output_dir,
            verbose=self.verbose,
            nusc_version=self.nusc_version,
            nusc_dataroot=self.nusc_dataroot,
            tokens=self.tokens,
        )
        return nusc_eval.main()


def run_tracking(args, nusc_loader, result_dict=None):
    """Run tracking and return results."""
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
    KNet_model_CTRA.load_state_dict(torch.load(args.model))
    KNet_model_CTRA.eval()

    nusc_tracker = Tracker(config=nusc_loader.config, model=KNet_model_CTRA)

    for frame_data in tqdm(nusc_loader, desc='Tracking'):
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

    return result


def main():
    args = parse_args()

    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(args.eval_path, exist_ok=True)

    # Load config
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.Loader)
    json.dump(config, open(f"{args.eval_path}/config.json", "w"))

    # Load tokens
    seq_token = load_file(args.token_path)
    if args.mode == 'eval':
        seq_token = seq_token[:10]  # Limit to 10 sequences for eval mode
    flat_list = []
    for sublist in seq_token:
        flat_list.extend(sublist)

    # Load dataloader
    nusc_loader = NuScenesloader(
        args.detection_path,
        args.first_token_path,
        args.final_token_path,
        args.token_path,
        config,
        suffle=False,
        token=flat_list
    )

    print(f"Result path: {os.path.abspath(args.result_path)}")

    if args.mode == 'eval':
        # Run tracking
        print("Running tracking for evaluation...")
        result = run_tracking(args, nusc_loader)
        json.dump(result, open(f"{args.result_path}/results.json", "w"))

        # Run evaluation
        print("Running nuScenes evaluation...")
        eval_path = f"{args.eval_path}/{args.eval_set}"
        os.makedirs(eval_path, exist_ok=True)

        evaluator = TrackingEval_self(
            config=None,
            result_path=args.result_path,
            eval_set=args.eval_set,
            output_dir=eval_path,
            nusc_version="v1.0-trainval",
            nusc_dataroot=args.nusc_path,
            verbose=True,
            tokens=flat_list,
        )
        metrics = evaluator.run()
        print(f"\nEvaluation Results:")
        print(f"  AMOTA: {metrics['amota']:.4f}")
        print(f"  AMOTP: {metrics['amotp']:.4f}")
        print(f"  IDS:   {metrics['ids']:.4f}")
    else:
        # Just run tracking
        result = run_tracking(args, nusc_loader)
        json.dump(result, open(f"{args.result_path}/results.json", "w"))
        print(f"Tracking complete. Results saved to {os.path.abspath(args.result_path)}")


if __name__ == "__main__":
    main()

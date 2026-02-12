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
parser.add_argument('--detection_path', type=str, default="/data1/wxx/MCTrack/nuscenes/detectors/largekernel/test.json")
parser.add_argument('--first_token_path', type=str, default='data/utils/test/nusc_first_token.json')
parser.add_argument('--final_token_path', type=str, default='data/utils/test/nusc_final_token.json')
parser.add_argument('--token_path', type=str, default='data/utils/test/nusc_token.json')
parser.add_argument('--model', type=str, default="/data1/wxx/vel_semi/model_4_3input/step_13700.pth")
parser.add_argument('--result_path', type=str, default='result/' + localtime)
parser.add_argument('--eval_path', type=str, default='eval_result2/')
args = parser.parse_args()

seq_token = load_file(args.token_path)
flat_list = []
for sublist in seq_token:
    flat_list.extend(sublist)


def main(result_path, token, process, nusc_loader,nusc_path):
    # PolyMOT modal is completely dependent on the detector modal
    result = {
        "results": {},
        "meta": {
            "use_camera": True,
            "use_lidar": False,
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

    # tracking and output file
    nusc_tracker = Tracker(config=nusc_loader.config,model=KNet_model_CTRA)

    # from nuscenes import NuScenes
    # from nuscenes.eval.common.loaders import load_gt
    # from nuscenes.eval.tracking.data_classes import TrackingBox
    # from nuscenes.eval.common.config import config_factory as track_configs
    # nusc = NuScenes(version="v1.0-trainval",verbose=True, dataroot=nusc_path)
    # cfg = track_configs("tracking_nips_2019")
    # gt_boxes = load_gt(nusc, "trainval", TrackingBox, verbose=True)

    # assert (nusc_loader.all_sample_token == gt_boxes.sample_tokens)

    for frame_data in tqdm(nusc_loader, desc='Running', total=len(nusc_loader) // process, position=token):\

        if process > 1 and frame_data['seq_id'] % process != token:
            continue
        sample_token = frame_data['sample_token']

        # gt_data = gt_boxes.boxes[sample_token]

        # track each sequence
        if nusc_tracker.tracking(frame_data):
            break
        """
        only for debug
        {
            'np_track_res': np.array, [num, 17] add 'tracking_id', 'seq_id', 'frame_id'
            'box_track_res': np.array[NuscBox], [num,]
            'no_val_track_result': bool
        }
        """
        # output process
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

        # add to the output file
        if sample_token in result["results"]:
            result["results"][sample_token] = result["results"][sample_token] + sample_results
        else:
            result["results"][sample_token] = sample_results


    # sort track result by the tracking score
    for sample_token in result["results"].keys():
        confs = sorted(
            [
                (-d["tracking_score"], ind)
                for ind, d in enumerate(result["results"][sample_token])
            ]
        )
        result["results"][sample_token] = [
            result["results"][sample_token][ind]
            for _, ind in confs[: min(500, len(confs))]
        ]

    # write file
    if process > 1:
        json.dump(result, open(result_path + str(token) + ".json", "w"))
    else:
        json.dump(result, open(result_path + "/results.json", "w"))

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

    if args.process > 1:
        result_temp_path = args.result_path + '/temp_result'
        os.makedirs(result_temp_path, exist_ok=True)
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            pool.apply_async(main, args=(result_temp_path, token, args.process, nusc_loader))
        pool.close()
        pool.join()
        results = {'results': {}, 'meta': {}}
        # combine the results of each process
        for token in range(args.process):
            result = json.load(open(os.path.join(result_temp_path, str(token) + '.json'), 'r'))
            results["results"].update(result["results"])
            results["meta"].update(result["meta"])
        json.dump(results, open(args.result_path + '/results.json', "w"))
        print('writing result in folder: ' + os.path.abspath(args.result_path))
    else:
        main(args.result_path, 0, 1, nusc_loader,args.nusc_path)
        print('writing result in folder: ' + os.path.abspath(args.result_path))

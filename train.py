import yaml, argparse, time, os, json, multiprocessing
from dataloader.nusc_loader_self import NuScenesloader
from tracking.nusc_tracker import Tracker
from tqdm import tqdm
import pdb
import torch.nn.init as init
from KalmanNet_nn import KalmanNetNN

from torch.utils.tensorboard import SummaryWriter   

# from torch.utils.tensorboard import 
# from torch.utils.tensorboard import GlobalSummaryWriter    

import torch

import logging

import datetime

from utils.io import load_file

import time
from multiprocessing import Process
from multiprocessing import Pool



import warnings
warnings.filterwarnings("ignore")


from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.common.config import config_factory as track_configs
from nuscenes.eval.tracking.data_classes import TrackingMetrics, TrackingMetricDataList, TrackingConfig, TrackingBox, \
    TrackingMetricData
from typing import Tuple, List, Dict, Any


def parse_config():
    parser = argparse.ArgumentParser()
    # running configurations
    parser.add_argument('--process', type=int, default=1)
    # paths
    # localtime = ''.join(time.asctime(time.localtime(time.time())).split(' '))
    localtime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    parser.add_argument('--config_path', type=str, default='config/nusc_config.yaml')

    parser.add_argument('--train_path', type=str, default='/home/wxx/Poly-MOT/data/utils/train')
    utils_path = parser.parse_args().train_path
    # detection_path = utils_path+"/infos_10sweeps_withvelo_filter_True.json"
    detection_path = "/home/wxx/Poly-MOT/data/detector/centerPoint/infos_train_10sweeps_withvelo_filter_True.json"
    first_token_path = utils_path+"/nusc_first_token.json"
    final_token_path = utils_path+"/nusc_final_token.json"
    token_path = utils_path+"/nusc_token.json"
    gt_path = utils_path+"/gt_train.json"
    parser.add_argument('--detection_path', type=str, default=detection_path)
    parser.add_argument('--first_token_path', type=str, default=first_token_path)
    parser.add_argument('--final_token_path', type=str, default=final_token_path)
    parser.add_argument('--token_path', type=str, default=token_path)
    parser.add_argument('--gt_path', type=str, default=gt_path)

    parser.add_argument('--val_path', type=str, default='/home/wxx/Poly-MOT/data/utils/val')
    val_path = parser.parse_args().val_path
    # detection_path = val_path+"/infos_10sweeps_withvelo_filter_True.json"
    detection_path = "/home/wxx/Poly-MOT/data/detector/centerPoint/infos_val_10sweeps_withvelo_filter_True.json"
    first_token_path = val_path+"/nusc_first_token.json"
    final_token_path = val_path+"/nusc_final_token.json"
    token_path = val_path+"/nusc_token.json"
    parser.add_argument('--detection_val_path', type=str, default=detection_path)
    parser.add_argument('--first_val_token_path', type=str, default=first_token_path)
    parser.add_argument('--final_val_token_path', type=str, default=final_token_path)
    parser.add_argument('--token_val_path', type=str, default=token_path)

    parser.add_argument('--output_path', type=str, default='output/' + localtime)
    output_path = parser.parse_args().output_path
    os.makedirs(output_path, exist_ok=True)
    parser.add_argument('--log_file', type=str, default= output_path+'/train.txt')
    parser.add_argument('--result_path', type=str, default=output_path + '/result/')
    parser.add_argument('--eval_path', type=str, default=output_path+'/eval_result2/')
    parser.add_argument('--nusc_path', type=str, default='/data1/wxx/nuscenes/v1.0-trainval/')
    parser.add_argument('--tensorboard_path', type=str, default='output/tensorboard/'+localtime)

    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--model', type=str, default='/home/wxx/Poly-MOT/output/20240904-183025/latest.pth')
    
    args = parser.parse_args()

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

    KNet_model_CTRA = KalmanNetNN(config=nusc_loader_train.config)
    KNet_model_CTRA.NNBuild(MD=9, SD=10)
    KNet_model_CTRA.cuda()

    # KNet_model_CTRA.load_state_dict(torch.load(args.model))

    for name, param in KNet_model_CTRA.named_parameters():
        if 'weight' in name:
            if len(param.shape) < 2:
                init.xavier_normal_(param.unsqueeze(0))  # 这里使用了Xavier正态分布初始化权重
            else:
                init.xavier_normal_(param)
        elif 'bias' in name:
            init.constant_(param, 0)  # 这里初始化偏置为0

    # KNet_model_CTRA.train()

    # tracking and output file
    nusc_tracker = Tracker(config=nusc_loader_train.config,model=KNet_model_CTRA)
    
    gt_boxes = load_file(args.gt_path,logger)

    # values_list = list(my_dict.values())

    # gt_boxes = TrackingBox.deserialize(gt_boxes)

    # assert (nusc_loader_train.all_sample_token == list(gt_boxes.keys()))
    num_step = 0

    # save init model
    PATH = args.output_path + '/model/'
    os.makedirs(PATH, exist_ok=True)
    Last_PATH = PATH+'/last_step.pth'
    PATH = PATH+'/step_%s.pth' %num_step
    torch.save(KNet_model_CTRA.state_dict(), PATH)


    total_epochs = args.epochs

    result_process = multiprocessing.Process(target=result_function, args=(args,signal_queue, result_queue))
    result_process.start()


    for i in range(total_epochs):
        nusc_loader_train.suffle_list()
        for i_seq , frame_data in enumerate(nusc_loader_train):

            sample_token = frame_data['sample_token']

            gt_data = gt_boxes[sample_token]
            # logger.info("===========================================================================")

            # logger.info("%d/%d epochs, %d/%d frames", i+1, total_epochs, i_seq+1, len(nusc_loader_train))
            # logger.info("frame_id: %d, seq_id: %d", frame_data['frame_id'], frame_data['seq_id'])

            # track each sequence
            loss = [0,0]
            if nusc_tracker.tracking(frame_data,gt_data,logger,loss,i):
                result_queue.put([num_step,loss[0],loss[1]])
                signal_queue.put("loss")

                if num_step % 100 == 0:
                    
                    # PATH = args.output_path + '/model/step_%s.pth' %num_step
                    # PATH = args.output_path + '/latest.pth'
                    PATH = "/data1/wxx/vel_semi" + '/model_4_3input_nosemi/step_%s.pth' %num_step
                    # torch.save(KNet_model_CTRA.state_dict(), PATH)
                    Process(target=test,args=(args,nusc_loader_val,flat_list,PATH,num_step,signal_queue,result_queue)).start()
                    # po.apply_async(func=test, args=(args,nusc_loader_val,flat_list,PATH,num_step,))
                    # # test_process = Process(target=test,args=(args,nusc_loader_val,flat_list,PATH,num_step,writer,log_to_tensorboard))
                    # # test_process.start()

                    # Process(target=test,args=(args,nusc_loader_val,flat_list,PATH,num_step,lock)).start()
                logger.info("===========================================================================")

                logger.info("%d/%d epochs, %d/%d frames", i+1, total_epochs, i_seq+1, len(nusc_loader_train))
                logger.info("frame_id: %d, seq_id: %d", frame_data['frame_id'], frame_data['seq_id'])
                num_step += 1
                assert (num_step == frame_data['seq_id'])
                
    result_process.terminate()


def result_function(args,signal_queue, result_queue):
    os.makedirs(args.tensorboard_path, exist_ok=True)
    writer = SummaryWriter(args.tensorboard_path)
    while True:
        # 等待信号
        signal = signal_queue.get()

        # 如果收到信号，处理结果
        if signal == "Signal":
            result = result_queue.get()
            writer.add_scalar('amota', result[1], result[0])
            writer.add_scalar('amotp', result[2], result[0])
            writer.add_scalar('ids', result[3], result[0])
        if signal == "loss":
            result = result_queue.get()
            writer.add_scalar('loss', result[1], result[0])
            writer.add_scalar('lr', result[2], result[0])

def test (args,nusc_loader_val,flat_list,path,num_step,signal_queue,result_queue):   
    os.makedirs(args.result_path+str(num_step), exist_ok=True)
    os.makedirs(args.eval_path+str(num_step), exist_ok=True)
    
    val(args.result_path+str(num_step), nusc_loader_val,path)
    # eval result
    [amota,amotp,ids] = eval(os.path.join(args.result_path+str(num_step), 'results.json'), args.eval_path+str(num_step), args.nusc_path,flat_list)

    # 将结果放入结果队列
    result_queue.put([num_step,amota,amotp,ids])


    # 发送信号给另一个进程
    signal_queue.put("Signal")
    # with lock:
    #     writer.add_scalar('amota', amota, num_step)


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


def eval(result_path, eval_path, nusc_path,flat_list):
    from nuscenes.eval.tracking.evaluate import TrackingEval
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
        tokens = flat_list,
    )
    metrics_summary = nusc_eval.main()

    return [metrics_summary['amota'],metrics_summary['amotp'],metrics_summary['ids']]

def val(result_path, nusc_loader,path):
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

    KNet_model_CTRA.load_state_dict(torch.load(path))

    KNet_model_CTRA.eval()


    # tracking and output file
    nusc_tracker = Tracker(config=nusc_loader.config,model=KNet_model_CTRA)

    for frame_data in nusc_loader:

        sample_token = frame_data['sample_token']

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

    json.dump(result, open(result_path + "/results.json", "w"))

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

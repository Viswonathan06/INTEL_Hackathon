# coding: utf-8
import imageio, os
# os.environ["IMAGEIO_FFMPEG_EXE"] = "./audio-orchestrator-ffmpeg/bin/ffmpeg"

import argparse
import numpy as np
from tqdm import tqdm
import yaml
from collections import deque
import pandas as pd
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
import numpy as np
from utils.render import render
# from utils.render_ctypes import render
from itertools import islice, chain, repeat
from utils.functions import cv_draw_landmark, get_suffix
_no_padding = object()

def make_cols(feats = 204):
    cols = ['video', 'vid_number', 'second']
    for i in range(feats):
        cols.append(i)
    return cols

def chunk(it, size, padval=_no_padding):
    if padval == _no_padding:
        it = iter(it)
        sentinel = ()
    else:
        it = chain(iter(it), repeat(padval))
        sentinel = (padval,) * size
    return iter(lambda: tuple(islice(it, size)), sentinel)

def main(args):

    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    columns = make_cols()
    data = []
    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Given a video path
    fn = args.video_fp.split('/')[-1]
    # print(fn)
    vid_name = fn.split('.')[0]
    vid_num = fn.split('.')[1]

    reader = imageio.get_reader(args.video_fp)

    fps = reader.get_meta_data()['fps']
    fps =np.ceil(fps)
    # print('fps = ', fps)
    suffix = get_suffix(args.video_fp)
    csv_path = f'./features/aggregate_'+vid_name+'_facial.csv'
    if os.path.exists(csv_path):
        return
            # writer = imageio.get_# writer(video_wfp, fps=fps)
    print(fn+' processing!!!*********')

    # the simple implementation of average smoothing by looking ahead by n_next frames
    # assert the frames of the video >= n
    n_pre, n_next = args.n_pre, args.n_next
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_frame = deque()

    # run
    dense_flag = args.opt in ('2d_dense', '3d',)
    pre_ver = None
    for i, frame in tqdm(enumerate(reader)):
        if args.start > 0 and i < args.start:
            continue
        if args.end > 0 and i > args.end:
            break
        # print(i)
        frame_bgr = frame[..., ::-1]  # RGB->BGR

        if i == 0:
            # detect
            boxes = face_boxes(frame_bgr)
            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # refine
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # padding queue
            for _ in range(n_pre):
                queue_ver.append(ver.copy())
            queue_ver.append(ver.copy())

            for _ in range(n_pre):
                queue_frame.append(frame_bgr.copy())
            queue_frame.append(frame_bgr.copy())

        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

            roi_box = roi_box_lst[0]
            # todo: add confidence threshold to judge the tracking is failed
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            queue_ver.append(ver.copy())
            queue_frame.append(frame_bgr.copy())

        pre_ver = ver  # for tracking
        # smoothing: enqueue and dequeue ops
        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)
            flat_ver = np.hstack(ver_ave)
            flat_ver = list(flat_ver)
            # flat_ver.insert(0, i)
            # print('frame = ', i)
            # flat_ver.insert(0, vid_num)
            # flat_ver.insert(0, vid_name)
            
            # print('flat_ver = ', flat_ver)
            # print(flat_ver)
            data.append(flat_ver)

            # if args.opt == '2d_sparse':
            #     img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
            # elif args.opt == '2d_dense':
            #     img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
            # elif args.opt == '3d':
            #     img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
            # else:
            #     raise ValueError(f'Unknown opt {args.opt}')

            # writer.append_data(img_draw[:, :, ::-1])  # BGR->RGB

            queue_ver.popleft()
            queue_frame.popleft()
            

    # we will lost the last n_next frames, still padding
    for j in range(n_next):
        queue_ver.append(ver.copy())
        queue_frame.append(frame_bgr.copy())  # the last frame

        ver_ave = np.mean(queue_ver, axis=0)
        flat_ver = np.hstack(ver_ave)
        flat_ver = list(flat_ver)
        data.append(flat_ver)
        queue_ver.popleft()
        queue_frame.popleft()

    # print(len(data), len(data[0]))
    # print(type(data))
    splits = int(len(data)/fps)
    frame_wise_split = list(chunk(data,int(fps), None))
    # print(len(frame_wise_split))
    removed_frame = frame_wise_split.pop()
    # print(removed_frame)
    second = 0
    frame_agg = []
    for split in frame_wise_split:
        x = np.array(split)
        mean = list(np.mean(x, axis= 0))
        mean.insert(0, second)
        mean.insert(0, vid_num)
        mean.insert(0, vid_name)  
        # print(mean)   
        frame_agg.append(mean)
        second = second+1
    
    face_feats = pd.DataFrame(data = frame_agg, columns=columns)
    print(os.listdir('./'))
    face_feats.to_csv(csv_path)   
    print(f'Dump to {csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The smooth demo of video of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--video_fp', type=str)
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-n_pre', default=1, type=int, help='the pre frames of smoothing')
    parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '2d_dense', '3d'])
    parser.add_argument('-s', '--start', default=-1, type=int, help='the started frames')
    parser.add_argument('-e', '--end', default=-1, type=int, help='the end frame')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)

import os
import cv2
import argparse
from glob import glob
from tqdm import tqdm
import prettytable as pt
import numpy as np

from evaluation.evaluate import evaluator
from config import Config

config = Config()


def do_eval(opt):
    # evaluation for whole dataset
    # dataset first in evaluation
    for _data_name in opt.data_lst.split('+'):
        pred_data_dir = sorted(glob(os.path.join(opt.pred_root, opt.model_lst[0], _data_name)))
        if not pred_data_dir:
            print('Skip dataset {}.'.format(_data_name))
            continue
        gt_src = os.path.join(opt.gt_root, _data_name)
        gt_paths = sorted(glob(os.path.join(gt_src, 'GT', '*')))
        print('#' * 20, _data_name, '#' * 20)
        filename = os.path.join(opt.save_dir, '{}_eval.txt'.format(_data_name))
        tb = pt.PrettyTable()
        tb.vertical_char = '&'
        if config.task == 'DIS5K':
            tb.field_names = ["Dataset", "Method", "maxFm", "wFmeasure", 'MAE', "Smeasure", "meanEm", "HCE", "maxEm",
                              "meanFm", "adpEm", "adpFm"]
        elif config.task == 'COD':
            tb.field_names = ["Dataset", "Method", "Smeasure", "wFmeasure", "meanFm", "maxFm", "meanEm", "maxEm", 'MAE',
                              "adpEm", "adpFm", "HCE"]
        elif config.task == 'HRSOD':
            tb.field_names = ["Dataset", "Method", "Smeasure", "maxFm", "meanEm", 'MAE', "maxEm", "meanFm", "wFmeasure",
                              "adpEm", "adpFm", "HCE"]
        else:
            tb.field_names = ["Dataset", "Method", "Smeasure", 'MAE', "maxEm", "meanEm", "maxFm", "meanFm", "wFmeasure",
                              "adpEm", "adpFm", "HCE"]
        for _model_name in opt.model_lst[:]:
            print('\t', 'Evaluating model: {}...'.format(_model_name))

            # 包含预测结果图片的文件路径 (精确到文件)
            pred_paths = [p.replace(opt.gt_root, os.path.join(opt.pred_root, _model_name)).replace('/GT/', '/') for p in
                          gt_paths]

            # print(pred_paths[:1], gt_paths[:1])
            em, sm, fm, mae, wfm, hce = evaluator(
                gt_paths=gt_paths,
                pred_paths=pred_paths,
                metrics=opt.metrics.split('+'),
                verbose=config.verbose_eval
            )
            if config.task == 'DIS5K':
                scores = [
                    fm['curve'].max().round(3), wfm.round(3), mae.round(3), sm.round(3), em['curve'].mean().round(3),
                    int(hce.round()),
                    em['curve'].max().round(3), fm['curve'].mean().round(3), em['adp'].round(3), fm['adp'].round(3),
                ]
            elif config.task == 'COD':
                scores = [
                    sm.round(3), wfm.round(3), fm['curve'].mean().round(3), fm['curve'].max().round(3),
                    em['curve'].mean().round(3), em['curve'].max().round(3), mae.round(3),
                    em['adp'].round(3), fm['adp'].round(3), int(hce.round()),
                ]
            elif config.task == 'HRSOD':
                scores = [
                    sm.round(3), fm['curve'].max().round(3), em['curve'].mean().round(3), mae.round(3),
                    em['curve'].max().round(3), fm['curve'].mean().round(3), wfm.round(3), em['adp'].round(3),
                    fm['adp'].round(3), int(hce.round()),
                ]
            else:
                scores = [
                    sm.round(3), mae.round(3), em['curve'].max().round(3), em['curve'].mean().round(3),
                    fm['curve'].max().round(3), fm['curve'].mean().round(3), wfm.round(3),
                    em['adp'].round(3), fm['adp'].round(3), int(hce.round()),
                ]

            for idx_score, score in enumerate(scores):
                scores[idx_score] = '.' + format(score, '.3f').split('.')[-1] if score <= 1 else format(score, '<4')
            records = [_data_name, _model_name] + scores
            tb.add_row(records)
            # Write results after every check.
            with open(filename, 'w+') as file_to_write:
                file_to_write.write(str(tb) + '\n')
        print(tb)


if __name__ == '__main__':
    # set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt_root', type=str, help='ground-truth root',
        default=config.dataset_root)
    parser.add_argument(
        '--pred_root', type=str, help='prediction root',
        default='./e_preds')
    parser.add_argument(
        '--data_lst', type=str, help='test dataset',
        default={
            'DIS5K': '+'.join(['DIS-VD', 'DIS-TE1', 'DIS-TE2', 'DIS-TE3', 'DIS-TE4'][:]),
            'COD': '+'.join(
                ['CAMO_TestingDataset', 'CHAMELEON_TestingDataset', 'COD10K_TestingDataset', 'NC4K_TestingDataset'][:]),
            'HRSOD': '+'.join(['DAVIS-S', 'TE-HRSOD', 'TE-UHRSD', 'TE-DUTS', 'DUT-OMRON'][:])
        }[config.task])
    parser.add_argument(
        '--save_dir', type=str, help='candidate competitors',
        default='e_results')
    parser.add_argument(
        '--check_integrity', type=bool, help='whether to check the file integrity',
        default=False)
    parser.add_argument(
        '--metrics', type=str, help='candidate competitors',
        default='+'.join(['S', 'MAE', 'E', 'F', 'WF', 'HCE'][:100 if config.task == 'DIS5K' else -1]))
    opt = parser.parse_args()

    os.makedirs(opt.save_dir, exist_ok=True)
    # ./e_preds 下的文件名列表
    opt.model_lst = [m for m in sorted(os.listdir(opt.pred_root), key=lambda x: int(x.split('ep')[-1]), reverse=True) if
                     int(m.split('ep')[-1]) % 1 == 0]

    # check the integrity of each candidates
    if opt.check_integrity:
        for _data_name in opt.data_lst.split('+'):
            for _model_name in opt.model_lst:
                gt_pth = os.path.join(opt.gt_root, _data_name, 'GT')
                pred_pth = os.path.join(opt.pred_root, _model_name, _data_name)
                if not sorted(os.listdir(gt_pth)) == sorted(os.listdir(pred_pth)):
                    print(len(sorted(os.listdir(gt_pth))), len(sorted(os.listdir(pred_pth))))
                    print('The {} Dataset of {} Model is not matching to the ground-truth'.format(_data_name,
                                                                                                  _model_name))

    else:
        print('>>> skip check the integrity of each candidates')

    # start engine
    do_eval(opt)

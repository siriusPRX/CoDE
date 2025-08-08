import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
from mini_dataloader import DataLoader
from model import select_model
from state import select_state
from utils import cal_auc, cal_iou, cal_f1, AverageMeter, show_result
import numpy as np
from tqdm import tqdm
import torch



#_/_/_/ training parameters _/_/_/
parser = argparse.ArgumentParser(description='forgery_detection_A3C')
# Basic parameters
parser.add_argument('--WEIGHT-PATH', type=str,
                    default='./weights/CoDE_pre-trained.pth',
                    help='Pre-trained model.')
parser.add_argument('--MODEL', type=str, default='AC_CoDE',
                    help='Training model.')
parser.add_argument('--STATE', type=str, default='default',
                    help='State. default:(default)')
parser.add_argument('--NORMALIZE', type=bool, default=False,
                    help='Normalize the forgery image into [-1, 1]. default:(False)')
parser.add_argument('--DATASET', type=str, default='dataset',
                    help='Training dataset.')
parser.add_argument('--RATIO', type=float, default=0,
                    help='The split ratio of train set.')
parser.add_argument('--RANDOM-DIVIDED', type=bool, default=True,
                    help='Randomly shuffle the whole dataset before divided. default:(True)')
parser.add_argument('--TEST-BATCH-SIZE', type=int, default=1,
                    help='The batch of test data. default:(1)')
parser.add_argument('--INPUT-SIZE', type=int, default=512,
                    help='Size of image. default:(512)')
parser.add_argument('--AUGMENT-PROB', type=float, default=0.0,
                    help='Probability of data augmentation(flip, rotation). default:(0.2)')
parser.add_argument('--ENABLE-AUG-TYPES', type=list, default=None,
                    help='Types of data enhancement allowed.(0:horizontal_flip, 1:vertical_flip, 2:rotate, 3:resize,\
                         4:jpeg_compression, 5:gaussian_noise, 6:gaussian_blur), default:([0, 1, 2]])')
parser.add_argument('--INTENSITY', type=dict, default={'rates': None, 'qfs': None, 'sds': None, 'ksizes': None,
                                                       'ratios': None},
                    help='The intensity of corresponding data enhancement.(rates:resize, qfs:jpeg_compression,\
                         sds:gaussian_noise, ksizes:gaussian_blur)')
parser.add_argument('--VISUALIZE-NUMS', type=int, default=10,
                    help='visualize result. default:(0)')

# Parameters of the reinforcement learning framework
parser.add_argument('--EPISODE-LEN', type=int, default=3,
                    help='Total experience of one episode. default:(3)')
parser.add_argument('--FORGERY-THRESHOLD', type=float, default=0.2,
                    help='Threshold of tampering. If the probability is greater than the threshold, '
                         'it is considered as tampering. default:(0.2)')


def test(args):
    # _/_/_/ load val dataset _/_/_/
    test_batch_loader = DataLoader(name=args.DATASET, mode='test', ratio=args.RATIO, batch_size=args.TEST_BATCH_SIZE,
                                   input_size=args.INPUT_SIZE, augment_prob=args.AUGMENT_PROB,
                                   enable_aug_types=args.ENABLE_AUG_TYPES, intensity=args.INTENSITY,
                                   divide_shuffle=args.RANDOM_DIVIDED)

    # initialize state
    current_state = select_state(args.STATE)

    # create model
    model = select_model(args.MODEL)
    model.cuda()

    # load weight
    LOAD_PATH = args.WEIGHT_PATH
    model.load_state_dict(torch.load(LOAD_PATH, map_location='cpu')['state_dict'])
    model.eval()
    print("Loaded pretrained weights with {}".format(LOAD_PATH))

    f1 = AverageMeter()
    auc = AverageMeter()
    iou = AverageMeter()
    t = tqdm(range(len(test_batch_loader.flist)))
    with torch.no_grad():
        for i in t:
            # load a batch of data
            forgery, gt_mask, forgery_name, gt_name = test_batch_loader.test_get_item(i)
            current_state.reset(forgery)
            # iterative update
            for j in range(0, args.EPISODE_LEN):
                statevar = torch.Tensor(current_state.state).cuda()
                # ------- #
                if j == 0:
                    p, v = model.forgery_forward(statevar)
                mu, _, _ = model.prob_forward(statevar, p, v)
                # ------- #
                mu *= 0.5
                actions = mu.detach().cpu().numpy()
                current_state.step(actions)

            if args.VISUALIZE_NUMS != -1 and i % max(1, len(test_batch_loader.flist) // args.VISUALIZE_NUMS) == 0:
                save_forgery = forgery.copy()
                save_forgery = np.transpose(save_forgery.squeeze(), (1, 2, 0))
                save_gt_mask = gt_mask.copy()
                save_mask = np.where(current_state.mask.copy() > args.FORGERY_THRESHOLD, 1., 0.)

                show_result(["forgery", "GT", 'mask'], [save_forgery, save_gt_mask.squeeze(), save_mask.squeeze()])


            save_mask = current_state.mask.copy()
            # save_mask = current_state.mask.copy()[:, :, :ori_height, :ori_width]
            save_gt_mask = gt_mask.copy()
            auc.update(cal_auc(save_mask, save_gt_mask))
            save_mask = np.where(save_mask > args.FORGERY_THRESHOLD, 1., 0)
            f1.update(cal_f1(save_mask, save_gt_mask))
            iou.update(cal_iou(save_mask, save_gt_mask))
            t.set_description('F1: {:.3f} | AUC: {:.3f} | IoU: {:.3f}'.format(f1.avg, auc.avg, iou.avg))


if __name__ == '__main__':
    args = parser.parse_args()
    test(args)


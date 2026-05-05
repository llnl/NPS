import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
# from .guided_diffusion_modules.unet import EmbedBlock, EmbedSequential
#from .guided_diffusion_modules.nn import gamma_embedding
#from NPS.model.common import MLP_

# def register_args(parser):
#     parser.add_argument('--pd_file', type=str, default='', help='predict output')
#     parser.add_argument('--gt_file', type=str, default='', help='ground truth output')
#     parser.add_argument('--mode', type=str, default='train', choices=('train', 'eval', 'valid', 'predict', 'rollout', 'trace'),help='job type: train; eval|valid; predict|rollout')
#     parser.add_argument('--n_threads', type=int, default=3, help='number of threads for data loading')
#     parser.add_argument('--log_mem_train', action='store_true', help='prints max memory usage after each epoch training cycle')
#     parser.add_argument('--log_mem_valid', action='store_true', help='prints max memory usage after each epoch validation/prediction cycle')
#     parser.add_argument('--log_mem_loss', action='store_true', help='prints memory usage every training loss output. Does not reset stats collecting. Good for instantaeous looks.')
#     parser.add_argument('--log_mem', action='store_true', help='enables --log_mem_train and --log_mem_valid')
#     parser.add_argument('--save', type=str, default='test',
#                     help='file name to save')
#     parser.add_argument('--predict_ff', action='store_true', help='Feed forward for prediction. Reduces on memory cost for long-duration predictions.')
#     parser.add_argument('--lr_max', type=float, default=4e-3, help='Maximum learning rate')

#     parser.add_argument('--traj_out', type=str, default='', help='file to save test trajectories')
#     parser.add_argument('--infer_mode', type=str, default='original', choices=['original', 'optimize'])
    
#     args = parser.parse_args()
#     if not args.pd_file: args.pd_file=args.dir+'/pd.npy'
#     if not args.gt_file: args.gt_file=args.dir+'/gt.npy'

#     if args.log_mem: 
#         args.log_mem_train = True
#         args.log_mem_valid = True

#     if not args.data_train:
#         args.data_train = args.data+'/train.npy'
#     if not args.data_valid:
#         args.data_valid = args.data+f'/valid.npy'
#     if not args.data_predict:
#         args.data_predict = args.data+f'/test.npy'

#     if len(args.frame_shape) == 1:
#         args.frame_shape *= args.dim
#     else:
#         assert len(args.frame_shape) == args.dim, ValueError('frame shape mismatch')

#     if not args.test_set:
#         args.test_set = {'predict':'test'}.get(args.mode, 'valid')
#     args.n_out_test = 0
# def post_process_args(args):
#     pass

def register_args(parser):
    parser.add_argument('--pd_file', type=str, default='', help='predict output')
    parser.add_argument('--gt_file', type=str, default='', help='ground truth output')
    parser.add_argument('--mode', type=str, default='train',
                        choices=('train', 'eval', 'valid', 'predict', 'rollout', 'trace'),
                        help='job type: train; eval|valid; predict|rollout')
    parser.add_argument('--n_threads', type=int, default=3, help='number of threads for data loading')
    parser.add_argument('--log_mem_train', action='store_true', help='prints max memory usage after each epoch training cycle')
    parser.add_argument('--log_mem_valid', action='store_true', help='prints max memory usage after each epoch validation/prediction cycle')
    parser.add_argument('--log_mem_loss', action='store_true', help='prints memory usage every training loss output')
    parser.add_argument('--log_mem', action='store_true', help='enables --log_mem_train and --log_mem_valid')
    parser.add_argument('--save', type=str, default='test', help='file name to save')
    parser.add_argument('--predict_ff', action='store_true', help='Feed forward for prediction')
    parser.add_argument('--lr_max', type=float, default=4e-3, help='Maximum learning rate')
    parser.add_argument('--traj_out', type=str, default='', help='file to save test trajectories')
    parser.add_argument('--infer_mode', type=str, default='original', choices=['original', 'optimize'])


def post_process_args(args):
    if not args.pd_file:
        args.pd_file = args.dir + '/pd.npy'
    if not args.gt_file:
        args.gt_file = args.dir + '/gt.npy'

    if args.log_mem:
        args.log_mem_train = True
        args.log_mem_valid = True

    if not args.data_train:
        args.data_train = args.data + '/train.npy'
    if not args.data_valid:
        args.data_valid = args.data + '/valid.npy'
    if not args.data_predict:
        args.data_predict = args.data + '/test.npy'

    if len(args.frame_shape) == 1:
        args.frame_shape *= args.dim
    else:
        assert len(args.frame_shape) == args.dim, ValueError('frame shape mismatch')

    if not args.test_set:
        args.test_set = {'predict': 'test'}.get(args.mode, 'valid')

    args.n_out_test = 0
    return args

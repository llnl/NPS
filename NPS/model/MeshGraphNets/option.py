import argparse
# import template

# cat MeshGraphNets/options.py |sed "s/DEFINE_enum/DEFINE_string/"| sed "s/DEFINE_/ /;s/integer/int/;s/boolean/bool/;s/string/str/; s/('/ '--/" |awk 'BEGIN {FPAT = "([^ ]+)|(\"[^\"]+\")|('"'"'[^'"'"']+'"'"')"} {printf("parser.add_argument(%s type=%s, default=%s help=%s)\n", $3,$2,$4,$5)}'
parser = argparse.ArgumentParser(description='GNN NPS')
#################### Book-keeping/misc ####################
parser.add_argument('--dir', type=str, default=None, help='Directory of job')
parser.add_argument('--seed', type=int, default=54321, help='Rand seed')
#################### Data ####################
parser.add_argument('--data', type=str, default='', help='Directory to load dataset from.')
parser.add_argument('--dataset', type=str, default='longclip', help='data type')
parser.add_argument('--dim', type=int, default=2, help='NPS dimension')
parser.add_argument('--periodic', type=int, default=0, help='NPS periodic boundary condition')
parser.add_argument('--nfeat_in', type=int, default=1, help='nfeat_in')
parser.add_argument('--nfeat_out', type=int, default=-1, help='nfeat_out')
parser.add_argument('--frame_shape', type=str, default='64', help='frame shape')
parser.add_argument('--cache', type=bool, default=False, help='Cache whole dataset into memory')
parser.add_argument('--n_in_frame', type=int, default=1, help='no. of input frames')
parser.add_argument('--n_out_frame', type=int, default=1, help='no. of output frames')
parser.add_argument('--clip_step', type=int, default=1, help='clip_step')
parser.add_argument('--nfeat_out_global', type=int, default=0, help='no. of global output channels per graph')
#################### Model ####################
parser.add_argument('--model', type=str, default='NPS', help='Select model to run.')
parser.add_argument('--RNN', type=int, default=0, help='1 = recurrent (default), 0=feedforward (empty memory)')
parser.add_argument('--core', type=str, default='base_mp_gnn', help='core model')
parser.add_argument('--nfeat_latent_node', type=int, default=128, help='nfeat latent of node in GNN')
parser.add_argument('--nfeat_latent_edge', type=int, default=-1,  help='nfeat_latent of edge in GNN')
parser.add_argument('--n_mpassing', type=int, default=2, help='num. of message passing')
parser.add_argument('--nlayer_mlp', type=int, default=2, help='No. of layer in MLP')
parser.add_argument('--unique_op', action='store_true', default=False, help='Apply tf.unique operator in processing edges. Turn off to speed up but be sure the specified edges have no duplicates')
parser.add_argument('--mlp_activation', type=str, default='relu', help='Activation in MLP, e.g. relu')
parser.add_argument('--evaluator', type=str, default='cfd_eval', help='Select rollout method.')
### Mesh
parser.add_argument('--amr_N', type=int, default=64, help='system size, i.e. how many (fine) grids totally')
parser.add_argument('--amr_N1', type=int, default=1, help='how many (fine) grids to bin into one, 1 to disable')
parser.add_argument('--amr_buffer', type=int, default=1, help='how many buffer grids (must be 0 or 1)')
parser.add_argument('--amr_eval_freq', type=int, default=1, help='Call AMR in eval every this many times (default 1)')
parser.add_argument('--amr_threshold', type=float, default=1e-3, help='threshold to coarsen regions if values are close')
#################### Job ####################
parser.add_argument('--mode', type=str, default='train', help='Train model, or run evaluation.')
### Train
parser.add_argument('--nepoch', type=int, default=8000000, help='nepoch')
parser.add_argument('--batch', type=int, default=4, help='batch size')
parser.add_argument('--noise', type=float, default=0.0, help='noise magnitude')
parser.add_argument('--keep_ckpt', type=int, default=-1, help='number of checkpoints to keep. -1 to default(5)')
parser.add_argument('--print_freq', type=int, default=1000, help='')
parser.add_argument('--valid_freq', type=int, default=2000, help='Perform validation/checkpoint every this many steps')
parser.add_argument('--n_training_steps', type=int, default=int(10e6), help='No. of training steps')
parser.add_argument('--randommesh', type=bool, default=False, help='Data augmentation by generating random points and associated mesh')
parser.add_argument('--data_aug', type=str, default='', help='Data augmentation by pointgroup operation')
# optimizer
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
parser.add_argument('--train_split', type=float, default=0.9, help='train set ratio')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='beta1')
parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon')
parser.add_argument('--momentum', type=float, default=0, help='momentum')
# scheduler
parser.add_argument('--scheduler', type=str, default='plateau', help='lr reduction schedule')
parser.add_argument('--lr_decay_step', type=int, default=5000000, help='Learning rate decay factor.')
parser.add_argument('--lr_decay_factor', type=float, default=0.3, help='Learning rate decay steps.')
parser.add_argument('--lr_decay_patience', type=int, default=20, help='Learning rate decay steps.')
### Predict
parser.add_argument('--rollout', type=str, default='', help='Pickle file to save eval trajectories')
parser.add_argument('--rollout_split', type=str, default='valid', help='Dataset split to use for rollouts.')
parser.add_argument('--n_rollout', type=int, default=1, help='No. of rollout trajectories')
parser.add_argument('--ds_pred', type=str, default='', help='Dataset for prediction')
parser.add_argument('--n_out_pred', type=int, default=-1, help='no. of output frames in prediction')
parser.add_argument('--clip_step_test', type=int, default=-1, help='clip_step_pred')

args = parser.parse_args()
# template.set_template(args)

def str2list(x, typ=int): return list(map(typ, filter(bool, x.split(','))))

if args.nfeat_latent_edge == -1:
    args.nfeat_latent_edge = args.nfeat_latent_node
if args.nfeat_out == -1:
    args.nfeat_out = args.nfeat_in
args.periodic = bool(args.periodic)
args.RNN = bool(args.RNN)
args.frame_shape = str2list(args.frame_shape)
if len(args.frame_shape) == 1:
    args.frame_shape *= args.dim
else:
    assert len(args.frame_shape) == args.dim, ValueError('frame shape mismatch')

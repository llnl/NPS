import sys, importlib
import shlex
from NPS_common.io_utils import co

def get_NPS_model(model_dir, other_op="", return_args=False):
    import matplotlib
    backend = matplotlib.get_backend()

    dir_args = co(f'grep mode=train {model_dir}/config.txt |tail -n1| sed "s/mode=train/mode=eval/" {other_op}')
    dir_args = dir_args.replace('="', ' ').replace('"', '')
    dir_args = dir_args.replace('lambda ', 'lambda,')
    dir_args = dir_args.strip().split(' ')
    for i in range(len(dir_args)):
        dir_args[i] = dir_args[i].replace('lambda,', 'lambda ')
    # dir_args = dir_args.replace('="', ' "')
    # print(dir_args)
    # dir_args = shlex.split(dir_args.strip(), posix=False)
    # print(dir_args)

    from NPS import utility
    # from NPS import data
    # from NPS.data import Data
    from NPS import model
    # from NPS import loss
    from unittest.mock import patch
    with patch.object(sys, 'argv', dir_args):
        import NPS.option
        importlib.reload(NPS.option)
        from NPS.option import args

    if return_args: return args
    # torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    if not checkpoint.ok:
        exit()
    # loader = Data(args)
    model = model.Model(args, checkpoint)
    matplotlib.use(backend)
    return model



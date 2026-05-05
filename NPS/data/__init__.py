__author__ = 'Fei Zhou'

from importlib import import_module
from sklearn.model_selection import train_test_split

def load_data(args, typ, datf, split='train'):
    # if typ in ('longclip', 'graph_clip', 'biased_clip', 'vartime_clip'):
    if args.data_is_time_series and (typ != 'single_array'):
        if split == 'train':
            n_in, n_out, clip_step, nskip = args.n_in, args.n_out, args.clip_step, args.nskip
        elif split == 'predict':
            n_in, n_out, clip_step, nskip = args.n_in_test, 0, args.clip_step_test, args.nskip_test
        else:
            n_in, n_out, clip_step, nskip = args.n_in_test,args.n_out_test,args.clip_step_test, args.nskip_test
        # nskip = args.nskip
    # if typ == 'longclip':
    #     from NPS.data.longclip import longclip
        m = getattr(import_module('NPS.data.' + typ), typ)
        ds = m(args, datf, n_in+n_out, clip_step, nskip=nskip, split=split)
        print(f'Loaded {typ} {datf} size {ds.flat.shape if hasattr(ds.flat, "shape") else len(ds.flat)} start_pos {ds.start_pos.shape} stat {ds.statistics}')
        assert ds.start_pos.size > 0
        return ds
    # elif typ == 'graph_clip':
    #     from NPS.data.graph_clip import graph_clip
    #     ds = graph_clip(args, datf, n_in+n_out, clip_step, nskip=nskip)
    #     print(f'Loaded graph_clip {datf} size {len(ds.flat)} start_pos {ds.start_pos.shape} stat {ds.statistics}')
    #     return ds
    elif typ == 'hubbard1band':
        from NPS.data.hubbard1band import load_hubbard1band
        # print(f'debug data', args.data, load_hubbard1band(args.data, cache_data=args.cache))
        return load_hubbard1band(args.data, cache_data=args.cache, filter=args.datatype[12:])
    else:
        try:
            module_name = typ
            m = import_module('NPS.data.' + module_name)
            return getattr(m, module_name)(args, datf)
        except:
            idx = typ.rfind('.')
            module_name, ds_name = typ[:idx], typ[idx+1:]
            m = import_module(typ)
            return getattr(m, ds_name)(args, datf)


class _Data:
    def __init__(self, args):
        self.statistics = {}
        if args.single_dataset:
            ds = load_data(args, args.datatype, args.data)
            ds_train = -args.train_split if args.train_split < 0 else int(args.train_split)
            self.train, self.test = train_test_split(ds, train_size=ds_train, random_state=args.dataset_seed)
            try:
                self.statistics = ds.statistics
            except:
                self.statistics = None
        else:
            self.train = load_data(args, args.datatype, args.data_train, 'train') if (args.mode == 'train') else None
            self.test = load_data(args, args.datatype_test, args.data_test, 'predict' if args.mode == 'predict' else 'valid')
            try:
                self.statistics = self.train.statistics
            except:
                pass
        if args.dataloader == 'torch':
            from torch.utils.data import DataLoader as loader
        elif args.dataloader == 'geometric':
            from torch_geometric.loader import DataLoader as loader
        else:
            raise ValueError(f'Unknown dataloader {args.dataloader}')
        self.loader_func = loader
        if args.dataloader and (not args.model_preprocess_data):
            if self.train is not None:
                self.train = loader(self.train, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
            self.test = loader(self.test, batch_size=args.batch_test, shuffle=False, drop_last=False, num_workers=args.num_workers)


class Data:
    def __init__(self, args):
        ds = _Data(args)
        self.statistics = ds.statistics; self.train = ds.train; self.test = ds.test; self.loader_func = ds.loader_func
        if args.data2:
            data_sav = args.data; datatype_sav = args.datatype; dataloader_sav = args.dataloader; data_train_sav = args.data_train; data_test_sav = args.data_test; fshape_sav = args.frame_shape
            args.data = args.data2; args.datatype = args.datatype2; args.dataloader = args.dataloader2; args.data_train = args.data+'/train'; args.data_test = args.data+'/valid'; args.frame_shape=""; args.datatype_test=args.datatype2
            ds = _Data(args)
            self.statistics2 = ds.statistics; self.train2 = ds.train; self.test2 = ds.test; self.loader_func2 = ds.loader_func
            if self.train2 is None: self.train2 = self.test2 # load test2 in validation jobs
            args.data = data_sav; args.datatype = datatype_sav; args.dataloader = dataloader_sav; args.data_train = data_train_sav; args.data_test = data_test_sav; args.frame_shape = fshape_sav

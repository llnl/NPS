#!/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from NPS_common.utils import load_array, str2slice

def parse_cmd():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--DIM", "-d", type=int, default=0, help="Dimension of array. DO NOT SET")
    parser.add_argument("--tskip", type=int, default=1, help="time skip")
    parser.add_argument("--tbegin", type=int, default=0, help="time begin")
    parser.add_argument("--tend", type=int, default=None, help="time end")
    parser.add_argument("-o", default='', help="save as gif")
    parser.add_argument("data", help="data file(s) (.npy or .npz)", nargs='+')
    parser.add_argument("--range", default='', help="Default '' to auto detect range; set to e.g. '0,1' to override; ',' to set to None")
    parser.add_argument("--per_file_range", action='store_true', help="Auto determine min/max separately for each data file instead of using global min/max")
    parser.add_argument("--ichannel", "-i", "-c", default='0', help="which channel to show, default 0 (single channel), 0:2 or 1:4 RGB multi-channel")
    parser.add_argument("--ichannel_list", default='', help="per-file channel selection, e.g. '0,3' to select channel 0 from 1st file and channel 3 from 2nd file")
    parser.add_argument("--rv", action='store_true', help="invert RGB color")
    parser.add_argument("--norm", default='', help="convert specified channels to norm, e.g. '0:3'. Other channels not affected")
    parser.add_argument("--norm_pow", type=float, default=1, help="plot norm to this power")
    parser.add_argument("--cmap", default='bwr', help="cmap of imshow")
    parser.add_argument("--channel_index", type=int, default=-1, help="channel position, -1 for channel last (default) (-999 means no channel)")
    parser.add_argument("--slice", type=str, default=':', help="e.g. ':', '...,-1'")
    parser.add_argument("--delay", type=int, default=25, help="Delay between frames")
    parser.add_argument("--tick", type=int, default=1, help="ticks on/off")
    parser.add_argument("--axis", type=int, default=1, help="axis/frame on/off")
    parser.add_argument("--nrow", type=int, default=1, help="N rows")
    parser.add_argument("--stamp", type=int, default=1, help="time stamp on/off")
    parser.add_argument("--spacing", type=str, default='', help="between subplots. e.g. '0,0' to remove spacing")
    parser.add_argument("--use_RGB", action='store_true', help="plot with RGB")
    parser.add_argument("--RGB_rescale_min_std", default='0,1', help="rescale values for RGB")
    parser.add_argument("--figsize", type=str, default="1,1", help="figsize multiplier e.g. '1,2'")
    return parser

def process_parser(options):
    if options.o: matplotlib.use('Agg')
    # options.ichannel = list(map(int, options.ichannel.split(':')))

    # Handle per-file channel selection
    if options.ichannel_list:
        options.ichannel_list = list(map(str2slice, filter(bool, options.ichannel_list.split(','))))
    else:
        options.ichannel_list = []

    options.ichannel = list(map(str2slice, filter(bool, options.ichannel.split(','))))
    # if len(options.ichannel) == 1: options.ichannel *= nplot
    if (len(options.ichannel) == 1) and (not bool(options.norm)):
        # options.ichannel = slice(options.ichannel[0], options.ichannel[0]+1)
        options.ichannel = options.ichannel[0]
        # options.use_RGB = False
    # else: ## not sure what I was doing here
    #     options.ichannel = slice(*options.ichannel)#eval(f'slice({options.ichannel})')
        # options.use_RGB = True
    if not options.tick:
        plt.rcParams.update({"xtick.bottom" : False,
      "xtick.labelbottom" : False,
      "ytick.labelleft" : False,
      "ytick.left" : False})
    if not options.axis:
        plt.rcParams.update({"axes.spines.left" : False,
      "axes.spines.right" : False,
      "axes.spines.bottom" : False,
      "axes.spines.top" : False,})
    options.slice = eval(f'lambda x: x[{options.slice}]')

    # data
    nplot = len(options.data)
    data = []
    dat_minmax = []
    for i in range(nplot):
        # data.append(load_array(options.data[i]).astype('float32'))
        d = load_array(options.data[i]).astype('float32')
        d = options.slice(d)
        if options.channel_index == -999:
            d = d[...,None]
        elif options.channel_index != -1:
            if options.channel_index < 0:
                options.channel_index = d.ndim + options.channel_index
            new_ax = list(range(0,options.channel_index))+list(range(options.channel_index+1,d.ndim))+[options.channel_index]
            d = np.transpose(d, new_ax)
        if options.DIM > 0:
            d = d.reshape((-1,)+d.shape[-(options.DIM+1):])[options.tbegin:options.tend:options.tskip]
        else:
            d = d[:,options.tbegin:options.tend:options.tskip]
            # dat_minmax.append([np.amin(data[i],(0,1,2)), np.amax(data[i],(0,1,2))])
        # Use per-file channel selection if specified, otherwise use global ichannel
        ichannel_for_file = options.ichannel_list[i] if options.ichannel_list and i < len(options.ichannel_list) else options.ichannel
        d = d[...,ichannel_for_file]
        if options.norm:
            norm_slice = list(map(int, filter(bool, options.norm.split(':'))))
            d_norm = np.linalg.norm(d[...,norm_slice[0]:norm_slice[1]], axis=-1, keepdims=True)
            d_norm = np.power(d_norm, options.norm_pow)
            d = np.concatenate((d[...,:norm_slice[0]], d_norm, d[...,norm_slice[1]:]), -1)
        if options.DIM > 0:
            if options.use_RGB:
                rgb_min, rgb_std = tuple(map(float, options.RGB_rescale_min_std.split(',')))[:2]
                d = (d[...,:3]-rgb_min)/rgb_std
                if d.shape[-1] == 2:
                    d = np.concatenate((d, np.zeros_like(d)[...,:1]), -1)
                if (d.shape[-1] == 3) and options.rv:
                    d = 1 - d
        dat_minmax.append([float(np.amin(d)), float(np.amax(d))])
        print(options.data[i], 'value range', dat_minmax[i])
        data.append(d)
    dat_minmax=np.array(dat_minmax)
    if options.per_file_range and not options.range:
        # Use per-file min/max
        options.range_per_file = [[dat_minmax[i,0], dat_minmax[i,1]] for i in range(len(dat_minmax))]
        allmin = None  # Not used in per-file mode
        allmax = None
        options.range = [None, None, '']
    else:
        # Use global min/max (original behavior)
        if options.range:
            range_vals = options.range.split(',')
            allmin, allmax = [float(x) if x else None for x in range_vals][:2]
        else:
            allmin=np.amin(dat_minmax[:,0])
            allmax=np.amax(dat_minmax[:,1])
        options.range = [allmin, allmax, options.range]
        options.range_per_file = None
    options.figsize = list(map(float, options.figsize.split(',')))
    if len(options.figsize) == 1:
        options.figsize *= 2
    return options, data

def setup_plots(options):
    nrow = options.nrow
    nplot = len(options.data)
    ncol = int(np.ceil(nplot/nrow))
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol*4, nrow*4), squeeze=False,\
        **({"gridspec_kw": {'wspace':float(options.spacing.split(',')[0]), 'hspace':float(options.spacing.split(',')[1])}} if options.spacing else {}))
    axs = axs.ravel()
    return fig, axs

def run_animation(anim, fig):
    anim_running = True

    def onClick(event):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False
        else:
            anim.event_source.start()
            anim_running = True

    fig.canvas.mpl_connect('button_press_event', onClick)


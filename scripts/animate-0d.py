#!/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from NPS_common.animateND import parse_cmd, process_parser

parser = parse_cmd()
parser.add_argument("--phase", type=int, default=1, help="Plot phase space (x,y), 0 to disable, 2 to plot phase only and skip other plots")
parser.add_argument("--phase_joined", type=int, default=1, help="In phase space (x,y) plot: 1=joined (default), 0=scatter plot")
for action in parser._actions:
    if action.dest == 'ichannel':
        action.default = ':'
options = parser.parse_args()
options.DIM = 0
options, data = process_parser(options)
if data[0].shape[-1] < 2:
    options.phase = 0
if len(options.range) >=3 and options.range[2]:
    options.range.append(list(map(float, options.range[2].split(','))))
nplot_per_dat = data[0].shape[-1]
if options.phase:
    nplot_per_dat += 1
nplot = len(options.data)
nrow = options.nrow * nplot_per_dat
ncol = int(np.ceil(nplot* nplot_per_dat/nrow))
fig, axs = plt.subplots(nrow, ncol, figsize=(nrow*4, ncol*4), squeeze=False)
axs = axs.reshape(-1, nplot)
for i in range(nplot):
    if data[i].ndim ==2: data[i]=data[i][None,:]
    print(f" data {i} found {len(data[i])} batches")
    for dat in data[i]:
        if options.phase != 0: axs[-1, i].plot(dat[:,0].ravel(), dat[:, 1].ravel(), {0:".",1:"-"}[options.phase_joined])
        if options.phase == 2: continue
        for j in range(dat.shape[-1]):
            axs[j, i].plot(np.arange(len(dat[...,0].ravel())), dat[...,j].ravel())
        if len(options.range) >=3 and options.range[2]:
            axs[0, i].set_ylim(options.range[-1][:2])
            axs[1, i].set_ylim(options.range[-1][2:])
            axs[2, i].set_xlim(options.range[-1][:2])
            axs[2, i].set_ylim(options.range[-1][2:])


if options.o:
    fig.save(options.o, writer='imagemagick', fps=6)
else:
    plt.show()

#!/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from NPS_common.animateND import parse_cmd, process_parser, setup_plots, run_animation

parser = parse_cmd()
parser.add_argument("--interp", type=str, default='antialiased', help="Interpolation method: antialiased, nearest, etc")
options = parser.parse_args()
options.DIM = 2
options, data = process_parser(options)
nplot = len(options.data)
fig, axs = setup_plots(options)

# Handle per-file range
if options.range_per_file:
    vmin_list = [r[0] for r in options.range_per_file]
    vmax_list = [r[1] for r in options.range_per_file]
else:
    vmin_list = [options.range[0]] * nplot
    vmax_list = [options.range[1]] * nplot

ims=[]
for i in range(nplot):
    ax = axs[i]
    # note using per-file or global vmin, vmax
    ims.append(ax.imshow(data[i][0,:,:], cmap=plt.get_cmap(options.cmap), vmin=vmin_list[i], vmax=vmax_list[i], interpolation=options.interp))
#        fig.colorbar(ims[i], ax=ax)
    ax.set_xlim((0, data[i].shape[2]))
    ax.set_ylim((0, data[i].shape[1]))
    if options.stamp: ax.set_title('0')
    if not options.axis: ax.set_axis_off()
fig.tight_layout()

# animation function. This is called sequentially
def animate(t):
    for i in range(nplot):
        ims[i].set_data(data[i][t])
        print('step t', t)
        ax = axs[i]
        if options.stamp: ax.set_title(str(t))
    return ims

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=len(data[0]), interval=options.delay, blit=True)

run_animation(anim, fig)

if options.o:
    anim.save(options.o, writer='imagemagick', fps=6)
else:
    plt.show()

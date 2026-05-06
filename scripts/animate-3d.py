#!/bin/env python
#from IPython.display import HTML
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# from skimage import measure
# import argparse
from NPS_common.utils import load_array, str2slice
from NPS_common.animateND import parse_cmd, process_parser, setup_plots, run_animation

# parser = argparse.ArgumentParser()
parser = parse_cmd()
parser.add_argument("--type", default='slice2d', choices=("slice2d", "slice", "iso", "all", "hist", "hist_map", "sum_map", "voxel", "quiver", "hist2d"))
parser.add_argument("--iso_val", type=float, default=0.5, help="isosurface value")
parser.add_argument("--index2d", default='z=0', help="which 2d slice, e.g. z=0 slice. Comma separates multiple values, e.g. animate-3d.py a.npy a.npy --index2d 'z=0,x=50'")
parser.add_argument("--voxel_filter", default='', help="filter for voxels")
parser.add_argument("--size", type=int, default=32, help="array size of flat .bin")
parser.add_argument("--nbins", type=int, default=64, help="no. bins for histogram (type=hist)")
parser.add_argument("--hist_ylinear", action='store_true', help="Switching to linear scale plot (default is log)")
parser.add_argument("--view3d", type=str, default="210,165", help="azim=,elev")
options = parser.parse_args()
options.DIM = 3
if options.type in ("hist2d",) and (options.ichannel != ":"):
    options.ichannel = ":"
    print("WARNING: for hist2d plots, please set --ichannel to ':' and select channels with --slice")
options, data = process_parser(options)
(allmin, allmax) = options.range[:2]
# For per-file range mode
if options.range_per_file:
    allmin_list = [r[0] for r in options.range_per_file]
    allmax_list = [r[1] for r in options.range_per_file]
    # For backward compatibility, use first file's range as default
    allmin = allmin_list[0] if allmin is None else allmin
    allmax = allmax_list[0] if allmax is None else allmax
else:
    allmin_list = [allmin] * len(data)
    allmax_list = [allmax] * len(data)
nframe = len(data[0])
nplot = len(options.data)
nrow = options.nrow
ncol = int(np.ceil(nplot/nrow))
if options.type in ('slice', 'iso'):
    for i in range(len(data)):
        if len(data[i].shape) == 5:
            assert data[i].shape[-1] == 1
            data[i] = data[i][..., 0]
def hide_inside(arr):
    N123 = arr.shape[:3]
    flag = np.zeros(N123, dtype=bool)
    flag[[0, -1], :, :] = True
    flag[:, [0, -1], :] = True
    flag[:, :, [0, -1]] = True
    return flag
    # return flag[...,None].tile((1,1,1,arr.shape[3])) if arr.ndim==4 else flag
options.voxel_filter = eval(options.voxel_filter) if options.voxel_filter else hide_inside
cmap = eval(f"plt.cm.{options.cmap}")
options.view3d = [None,None] if options.view3d=="" else list(map(int, options.view3d.split(",")))

options.index2d = list(filter(bool, options.index2d.split(',')))
if len(options.index2d) == 1: options.index2d *= nplot
options.index2d= [[x.split('=')[0], int(x.split('=')[1])] for x in options.index2d]
# options.ichannel = list(map(str2slice, filter(bool, options.ichannel.split(','))))
# if len(options.ichannel) == 1: options.ichannel *= nplot
figsize = plt.figaspect(nrow/ncol)
fig = plt.figure(figsize=(figsize[0]*options.figsize[0], figsize[1]*options.figsize[1]))
if options.type in ('slice2d', 'hist', 'hist_map', 'quiver', "hist2d", "sum_map"):
    axs=[fig.add_subplot(nrow, ncol, i+1) for i in range(nplot)]
else:
    from mpl_toolkits.mplot3d import Axes3D
    axs=[fig.add_subplot(nrow, ncol, i+1, projection='3d') for i in range(nplot)]
    for iax, ax in enumerate(axs):
        ax.view_init(azim=options.view3d[0], elev=options.view3d[1])
        ax.set_axis_off()
        xmin=0; ymin=0; zmin=0
        xmax=data[iax][0].shape[0]; ymax=data[iax][0].shape[1]; zmax=data[iax][0].shape[2]
        ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
        edges_kw = dict(color='white', linewidth=1, zorder=1e3)
        ax.plot([xmin,xmin,xmax,xmax,xmin,xmin,xmin,xmin], [ymin,ymax,ymax,ymin,ymin,ymin,ymax,ymax], [zmin,zmin,zmin,zmin,zmin,zmax,zmax,zmin], **edges_kw)
        ax.plot([xmax,xmax,xmin], [ymin,ymin,ymin], [zmin,zmax,zmax], **edges_kw)


# assert options.type == 'slice2d' or all([isinstance(x, int) for x in options.ichannel]), ValueError(f'Color output with slicing only available in the slice2d mode')
# fig, axs = setup_plots(options)

# data=[]
# dat_minmax = []
# for i in range(nplot):
#     data.append(load_array(options.data[i]).astype('float32'))
#     if options.channel_index == -999:
#         data[i] = data[i][...,None]
#     elif options.channel_index == -1:
#         pass
#     elif options.channel_index >= 0:
#         new_ax = list(range(0,options.channel_index))+list(range(options.channel_index+1,data[i].ndim))+[options.channel_index]
#         data[i] = np.transpose(data[i], new_ax)
#     else:
#         raise f"Unknown channel_index {options.channel_index}"
#     data[i] = data[i].reshape((-1,)+data[i].shape[-DIM-1:])
#     data[i] = data[i][::options.tskip, ..., options.ichannel[i]]
#     if np.any(np.isnan(data[i])):
#         print("WARNING NAN encountered")
#         np.nan_to_num(data[i], False)
#     print(options.data[i], 'value range', np.min(data[i]), np.max(data[i]))
#     if options.type not in ('slice2d', 'hist', 'hist_map'):
#         axs[i].set_xlim((0, data[i].shape[1]))
#         axs[i].set_ylim((0, data[i].shape[2]))
#         axs[i].set_zlim((0, data[i].shape[3]))
#         axs[i].view_init(25, 65)
#     dat_minmax.append([np.min(data[i]), np.max(data[i])])
# #data = np.array(data)
# dat_minmax=np.array(dat_minmax)
# if options.range:
#     allmin = float(options.range.split(',')[0])
#     allmax = float(options.range.split(',')[1])
# else:
#     allmin=np.min(dat_minmax[:,0])
#     allmax=np.max(dat_minmax[:,1])
# print('Overall min max', allmin, allmax)
# nframe = len(data[0])
# for i in range(nplot):
#     if data[i].shape[-1] == 3:
#         print(f'3 color channels in data {i}, normalizing for color display')
#         data[i] = (data[i]-allmin)/(allmax-allmin)

def plot_3D_slice(array, ax, imin, imax):
    pic = []
    # min_val = array.min()
    # max_val = array.max()
    n_x, n_y, n_z = array.shape
    # cmap = plt.cm.YlOrRd
    nx0=ny0=nz0=0

    x_cut = array[nx0,:,:]
    Y, Z = np.mgrid[0:n_y, 0:n_z]
    X = nx0 * np.ones((n_y, n_z))
    pic.append(ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cmap((x_cut-imin)/(imax-imin))))
    #ax.set_title("x slice")

    y_cut = array[:,ny0,:]
    X, Z = np.mgrid[0:n_x, 0:n_z]
    Y = ny0 * np.ones((n_x, n_z))
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    pic.append(ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cmap((y_cut-imin)/(imax-imin))))
    #ax.set_title("y slice")

    z_cut = array[:,:,nz0]
    X, Y = np.mgrid[0:n_x, 0:n_y]
    Z = nz0 * np.ones((n_x, n_y))
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    pic.append(ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cmap((z_cut-imin)/(imax-imin))))
    #ax.set_title("z slice")
    return pic
    #plt.show()


def plot_3D_iso(arr, ax):
    from skimage import measure
    try:
        verts, faces, _, _ = measure.marching_cubes(arr, options.iso_val)
        return [ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], color=(0,0,0,0.5), lw=1)]
    except:
        return [ax.scatter([0], [0])]

def plot_3D_voxel(arr, ax, imin, imax):
    ax.clear()
    voxel_flag = options.voxel_filter(arr)
    voxel_color = arr if arr.ndim==4 else cmap(((arr-imin)/(imax-imin)))
    # print(f'    voxel count:', voxel_flag.sum(), f"{voxel_flag.shape=} {voxel_color.shape=}")
    # if voxel_color.shape[-1] < 3:
    #     voxel_color = np.pad(voxel_color, ((0,0),(0,0),(0,0),(0,3-voxel_color.shape[-1])))
    # return [ax.voxels(voxel_flag, facecolors= ((voxel_color-imin)/(imax-imin)))]
    ax.set_xlim3d(0, voxel_flag.shape[0])
    ax.set_ylim3d(0, voxel_flag.shape[1])
    ax.set_zlim3d(0, voxel_flag.shape[2])
    return [ax.voxels(voxel_flag, facecolors=voxel_color, cmap=cmap, edgecolors='k', linewidth=0.1, shade=True)]


def frame_slice(frm, xyz, idx):
    if xyz == 'x':
        return frm[idx,:,:]
    elif xyz == 'y':
        return frm[:,idx,:]
    elif xyz == 'z':
        return frm[:,:,idx]


if options.type == 'slice':
    plotfunc=lambda t: [plot_3D_slice(data[i][t], axs[i], allmin_list[i], allmax_list[i]) for i in range(nplot)]
elif options.type == 'iso':
    plotfunc=lambda t: [plot_3D_iso(data[i][t], axs[i]) for i in range(nplot)]
elif options.type == 'voxel':
    plotfunc=lambda t: [plot_3D_voxel(data[i][t], axs[i], allmin_list[i], allmax_list[i]) for i in range(nplot)]
elif options.type == 'sum_map':
    # counts_dat = np.log(np.transpose(np.array([_hist(t) for t in range(nframe)]), (1,2,0)))
    sum_dat = [np.array([data[i][t].sum(axis=(0,1,2)).ravel() for t in range(data[i].shape[0])]) for i in range(nplot)]
    nframe_ = nframe
    plotfunc=lambda t: [axs[i].plot(sum_dat[i]) for i in range(nplot)]#,
    nframe=1
    options.delay=99999
elif options.type == 'hist' or options.type == 'hist_map':
    bins_list = [np.linspace(allmin_list[i]-0.001, allmax_list[i]+0.0001, options.nbins) for i in range(nplot)]
    _hist=lambda t: [np.histogram(data[i][t].ravel(), bins=bins_list[i])[0] for i in range(nplot)]
    plots_last = _hist(-1)
    plots_first = _hist(0)
    ymax=[max(plots_first[i].max(), plots_last[i].max()) for i in range(nplot)]
    if options.type == 'hist':
        plotfunc=lambda t: [(axs[i].hist(data[i][t].ravel(), range=(allmin_list[i], allmax_list[i]), bins=options.nbins),
          None if options.hist_ylinear else axs[i].set_yscale('log'),
          axs[i].set_xlim(allmin_list[i], allmax_list[i]), axs[i].set_ylim(0, ymax[i])) for i in range(nplot)]
    elif options.type == 'hist_map':
        options.delay=999
        # counts_dat = np.log(np.transpose(np.array([_hist(t) for t in range(nframe)]), (1,2,0)))
        # counts_dat = np.transpose(np.array([_hist(t) for t in range(nframe)]), (1,2,0))
        counts_dat = [np.array([np.histogram(x.ravel(), bins=bins_list[i])[0] for x in data[i]]).T for i in range(nplot)]
        if not options.hist_ylinear:
            counts_dat = [np.log(x) for x in counts_dat]
        nframe_ = nframe
        plotfunc=lambda t: [axs[i].imshow(counts_dat[i], origin='lower', aspect='auto',
          interpolation='none', extent=(1,nframe_, allmin_list[i], allmax_list[i]), cmap=options.cmap) for i in range(nplot)]#,
        nframe=1
        options.delay=99999
elif options.type == "hist2d":
    xmin = [data[i][...,0].min() for i in range(nplot)]
    xmax = [data[i][...,0].max() for i in range(nplot)]
    ymin = [data[i][...,1].min() for i in range(nplot)]
    ymax = [data[i][...,1].max() for i in range(nplot)]
    plotfunc=lambda t: [(axs[i].hist2d(data[i][t][...,0].ravel(), data[i][t][...,1].ravel(), range=[[xmin[i], xmax[i]], [ymin[i], ymax[i]]], bins=options.nbins)
        #   None if options.hist_ylinear else axs[i].set_yscale('log')#,
          #axs[i].set_xlim(allmin, allmax), axs[i].set_ylim(0, ymax[i])
          ) for i in range(nplot)]
elif options.type == 'all':
    plotfunc=lambda t: [plot_3D_iso(data[i][t], axs[i]) + plot_3D_slice(data[i][t], axs[i], allmin_list[i], allmax_list[i]) for i in range(nplot)]
elif options.type == 'slice2d':
    plotfunc=lambda t: [axs[i].imshow(frame_slice(data[i][t],*options.index2d[i]), cmap=plt.get_cmap(options.cmap), vmin=allmin_list[i], vmax=allmax_list[i]) for i in range(nplot)]
elif options.type == 'quiver':
    plotfunc=lambda t: [axs[i].quiver(*frame_slice(data[i][t],*options.index2d[i]).transpose((2,0,1)), pivot='mid') for i in range(nplot)]
else:
    raise ValueError("unknown plotting type %s"%options.type)

plots = plotfunc(0)
#plt.show()
# animation function. This is called sequentially
def animate(t):
    if options.type == 'slice2d':
        for i in range(nplot):
            plots[i].set_data(frame_slice(data[i][t],*options.index2d[i]))
    elif options.type == 'quiver':
        for i in range(nplot):
            plots[i].set_UVC(*frame_slice(data[i][t],*options.index2d[i]).transpose((2,0,1)))
    elif options.type in ('hist', 'hist_map', "hist2d"):
        for i in range(nplot):
            axs[i].cla()
        plotfunc(t)
        #for i in range(nplot):
        #    axs[i].cla()
        #    plots[i] = newfigs[i]
    else:
        newfigs = plotfunc(t)
        for i in range(nplot):
            for j in range(len(plots[i])):
                if options.type == 'iso':
                    plots[i][j].remove()
                plots[i][j] = newfigs[i][j]
    print('    step t', t)
    return plots

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=nframe, interval=options.delay, blit=(options.type=='slice2d'))

#run_animation(anim, fig)
if options.o:
    anim.save(options.o, writer='imagemagick', fps=6)
else:
    plt.show()

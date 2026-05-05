import subprocess
import os, glob
import tempfile
import numpy as np
from contextlib import contextmanager
import ase, ase.io

def co(instr, split=False):
    out=subprocess.Popen(instr, stdout=subprocess.PIPE, shell=True, universal_newlines=True).communicate()[0]
    return out.split('\n') if split else out


@contextmanager
def temp_txt_file(data):
    temp = tempfile.NamedTemporaryFile(delete=False, mode='wt')
    temp.write(data)
    temp.close()
    try:
        yield temp.name
    finally:
        os.unlink(temp.name)


def save_traj(fname, traj_arrs, symbols=None, CoM=False):
    from .utils import save_array
    import torch
    if isinstance(traj_arrs, torch.Tensor):
        traj_arrs = traj_arrs.numpy()
    if isinstance(traj_arrs, np.ndarray) and any(map(lambda s:fname.endswith(s),(".bin",".npy",".npz"))):
        save_array(fname, traj_arrs)
        return
    elif isinstance(traj_arrs[0], ase.Atoms):
        ase.io.write(fname, traj_arrs)
        return
    if not isinstance(symbols, (list, tuple)):
        symbols = [symbols] * len(traj_arrs)
    if False:# isinstance(traj_arrs[0][0], (int, float)):
        import torch
        traj_arrs = torch.split(torch.tensor(traj_arrs), [len(s) for s in symbols])
    if CoM:
        traj_arrs = [f - f.mean(0, keepdim=True) for f in traj_arrs]
    from NPS_common.periodic_table import symbol_from_Z
    try:
        symbols = [symbol_from_Z(s) for s in symbols]
    except:
        pass
    if fname.endswith('.xyz'):
        from ase.io.xyz import write_xyz
        with open(fname, 'w') as fh:
            write_xyz(fh, [ase.Atoms(symbol, positions=arr) for arr, symbol in zip(traj_arrs, symbols)])


from collections import namedtuple
MDFrame = namedtuple('MDFrame', ('positions', 'velocities', 'box', 'node_type', 'time'))

class _large_trajectory:
    def __init__(self, fname, read_force=False, sorted=True, force_tags=("fx", "fy", "fz")):
        self.fname = fname
        self.read_force = read_force
        self.sorted= sorted
        self.current = 0
        if fname.endswith((".lammpstraj", ".lammpstrj", ".dump")):
            start = co(f"grep -n TIMESTEP {fname} | sed 's/:.*//'")
            self.start = list(map(int, start.strip().split("\n")))
            # from ase.io.lammpsrun import read_lammps_dump
            # s = read_lammps_dump(fname, index=0)
            # self.natom = len(s)
            # self.cell = s.cell[:]
            # self.pbc = s.get_pbc()
            header = co(f"head -n9 {fname}").strip().split("\n")
            records = header[-1].split(" ")
            try:
                self.index_id_type_xyz = [records.index(x)-2 for x in ("id", "type", "x", "y", "z")]
            except:
                self.index_id_type_xyz = [records.index(x)-2 for x in ("id", "type", "f_avg_pos[1]", "f_avg_pos[2]", "f_avg_pos[3]")]
            if self.read_force:
                self.index_force = [records.index(x)-2 for x in force_tags]
                if self.index_force:
                    print(f"  found force index {self.index_force}")
                else:
                    raise ValueError(f"cannot find force tags {force_tags} in {records}")
            # print(header, len(header))
            self.natom = int(header[3])
            self.cell = np.fromstring(" ".join(header[5:8]), dtype=float, sep=" ").reshape(3, -1)
            if "BOX BOUNDS xy xz yz" in header[4]:
                assert np.all(self.cell[:,-1] == 0.0), NotImplementedError(f"cannot handle triclinic box with large tilt {self.cell}")
            self.cell = np.diag(self.cell[:,1] - self.cell[:,0])
            # print(self.cell)
            self.pbc = [x=="pp" for x in header[4].split(" ")[-3:]]
            # print(self.pbc, self.natom, self.cell, self.index_xyz)
            # print(f"debug lammps trajectory reader ready")
        else:
            raise NotImplementedError(f"unknown large trajectory {fname}")

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.start)

    def __getitem__(self, i):
        if self.fname.endswith((".lammpstraj", ".lammpstrj", ".dump")):
            dat = np.loadtxt(self.fname, max_rows=self.natom, skiprows=self.start[i]-1+9, dtype=float)
            if self.sorted:
                dat = dat[np.argsort(dat[:,self.index_id_type_xyz[0]])]
            s = ase.Atoms(dat[:, self.index_id_type_xyz[1]].astype(int),
                                pbc=self.pbc, cell=self.cell,
                                positions=dat[:, self.index_id_type_xyz[2]:self.index_id_type_xyz[4]+1])
            if self.read_force:
                forces = None
                try:
                    forces = dat[:, self.index_force]
                except:
                    pass
                s.arrays['forces'] = forces
            return s

    def __next__(self):
        if self.current < len(self.start):
            s = self[self.current]
            # print(f"done reading frame {self.current}")
            self.current += 1
            return s
        raise StopIteration


# for even larger trajectory files, do NOT reopen and get arbitrary frame. Rather, open once and read sequentially
class _Large_trajectory(_large_trajectory):
    def __init__(self, fname, **kwx):
        super().__init__(fname, **kwx)
        self.fh = open(fname, "r")

    def __next__(self):
        if self.current >= len(self.start):
            return None
            raise StopIteration

        if self.fname.endswith((".lammpstraj", ".lammpstrj", ".dump")):
            content = " ".join([self.fh.readline() for _ in range(self.natom+9)][9:])
            # string_buffer = io.StringIO(content)
            # dat = np.loadtxt(string_buffer, max_rows=self.natom, skiprows=-1+9, dtype=float)
            # print(content.__class__, content[:999], content[-9:])
            # dat = np.fromstring(content, dtype=float, sep=" "); print(dat, dat.shape)
            dat = np.fromstring(content, dtype=float, sep=" ").reshape(self.natom, -1)
            if self.sorted:
                dat = dat[np.argsort(dat[:,self.index_id_type_xyz[0]])]
            s = ase.Atoms(dat[:, self.index_id_type_xyz[1]].astype(int),
                                pbc=self.pbc, cell=self.cell,
                                positions=dat[:, self.index_id_type_xyz[2]:self.index_id_type_xyz[4]+1])
            if self.read_force:
                forces = None
                try:
                    forces = dat[:, self.index_force]
                except:
                    pass
                s.arrays['forces'] = forces
        else:
            raise NotImplementedError(f"unknown large trajectory {self.fname}")
        # print(f"done reading frame {self.current}")
        self.current += 1
        return s

    def __getitem__(self, i):
        if i == self.current:
            return self.__next__()
        else:
            raise NotImplementedError(f"cannot access frame {i} in Large trajectory reader")

def read_traj(f, to_graph=False, to_dict=False, fastread=0, read_force=False, sorted=True, force_tags=("fx", "fy", "fz")):
    if fastread == 2:
        return _large_trajectory(f, read_force=read_force, sorted=sorted, force_tags=force_tags)
    elif fastread == 1:
        return _Large_trajectory(f, read_force=read_force, sorted=sorted, force_tags=force_tags)
    if f.endswith('.pdb'):
        import MDAnalysis
        return MDAnalysis.coordinates.PDB.PDBReader(f).trajectory
    elif f.endswith(('.pdb', '.xyz', '.extxyz', '.cif', 'vasp', 'XDATCAR')):
        import ase.io
        return ase.io.read(f, index=':')
    elif ('.lammpstraj' in f) or ('.lammpstrj' in f):
        from ase.io.lammpsrun import read_lammps_dump
        return read_lammps_dump(f, index=slice(0,None))
    elif f.endswith('.xtc'):
        f_typ = get_associated_filename(f, ".atomtype.txt", "atomtype.txt")
        try:
            node_type = np.array(list(filter(bool, np.genfromtxt(f_typ, dtype='str'))))
            print(f"  atom types from {f_typ}")
        except:
            print(f"*********** WARNING: ***********\nCannot find atomtype.txt next to {f}")
            node_type = None
        f_bond = get_associated_filename(f, ".fixedbond.txt", "fixedbond.txt")
        try:
            fixedbond = np.array(list(filter(bool, np.loadtxt(f_bond, dtype=int))))
            fixedbond = np.concatenate((fixedbond, fixedbond[:,2:0:-1]), 0).T
            fixedbond_type = np.ones_like(fixedbond[0])
            print(f"  bond types from {f_bond}")
        except:
            print(f"*********** WARNING: ***********\nCannot find fixedbond.txt next to {f}")
            fixedbond = np.array([[]])
            fixedbond_type = np.array([[]])
        from MDAnalysis.lib.formats.libmdaxdr import XTCFile
        with XTCFile(f) as f_traj:
            if node_type is None:
                node_type = np.zeros(len(f_traj[0].x), dtype=int)
            # note: convert nm to Angstrom
            traj = [MDFrame(frame.x*10, None, frame.box, node_type, frame.time) for frame in f_traj]
        if len(traj) >= 2:
            if traj[0].time == traj[1].time: # modified xtc format with pos, velocity
                print(f'    Found velocity in XTC {f}')
                traj = [MDFrame(traj[i].positions, traj[i+1].positions, traj[i].box, traj[i].node_type, traj[i].time) for i in range(0, len(traj)//2*2, 2)]
        if to_graph:
            raise NotImplementedError("reading trajectory to graph is not implemented")
        elif to_dict:
            traj = [{'pos':np.array(x.positions, dtype=np.float32), 'type': node_type,
                     'fixedbond': fixedbond, 'fixedbond_type': fixedbond_type} for x in traj]
        print(f"{fixedbond=} {fixedbond_type=} {f_bond=} {f_bond=}")
        return traj
    elif f.endswith('.npz'):
        f_traj = np.load(f)
        assert to_dict
        traj = [{'pos':np.array(f_traj['x'][it], dtype=np.float32), 'type': f_traj['type'],
                     'fixedbond': f_traj['fixedbond'], 'fixedbond_type': f_traj['fixedbond_type']} for it in range(len(f_traj['x']))]
        return traj
    else:
        raise ValueError(f'Unknown trajectory type in {f}')


def read_topol(f):
    from itertools import permutations

    lines = open(f,'r').read().splitlines()#open(f, 'r').readlines()
    lines = list(filter(lambda x: not x.startswith(';'), map(lambda x: x.strip(), lines)))
    # print(f'debug l 3', lines[:13])
    keys = ('atom', 'bond', 'pair', 'angle', 'dihedral')
    order = (1, 2, 2, 3, 4)
    istart = [lines.index(f'[ {tag}s ]') for tag in keys]
    iend = (np.array(istart)[1:]).tolist()
    # print(iend, lines.index('', istart[-1]+1))
    iend.append(lines.index('', istart[-1]))
    g = {k: np.array(list(map(lambda x: list(map(int, x.split()[:order[i]])), filter(bool, lines[istart[i]+1:iend[i]]))))-1 for i, k in enumerate(keys)}
    nnode = len(g['atom'])
    g['charge'] = np.array(list(map(lambda x: list(map(float, x.split()[6:7])), filter(bool, lines[istart[0]+1:iend[0]]))))
    bond = np.concatenate((g['bond'], g['bond'][:,::-1]))
    g['bond_index'] = bond.T
    g['bond_type'] = np.ones_like(g['bond_index'][0])
    # print(bond.shape, g['bond_index'].shape, g['bond_type'].shape )
    # bond_idx = np.arange(nnode**2).reshape(-1,nnode)
    bond_idx = np.zeros([nnode]*2, dtype=int)
    bond_idx[tuple(bond.T)] = np.arange(len(bond))
    # angle = [[bond_idx[jk[0], i], bond_idx[jk[1], i]] for i in range(nnode) for jk in permutations(bond[:,1][bond[:,0]==i], 2)]
    angle_index = np.array([[bond_idx[i,j], bond_idx[k,j]] for i,j,k in g['angle']])
    angle_index = np.concatenate((angle_index, angle_index[:,::-1]))
    g['angle_idx'] = angle_index
    dihedral_index = np.array([[bond_idx[i,j], bond_idx[l,k]] for i,j,k,l in g['dihedral']])
    dihedral_index = np.concatenate((dihedral_index, dihedral_index[:,::-1]))
    g['dihedral_idx'] = dihedral_index
    # print(istart, iend, [lines[i:j] for i, j in zip(istart, iend)])
    # print(g, "g['angle_idx']", g['angle_idx'].shape, g['angle_idx'])
    return g



def load_nequip_dataset(fname, pbc=False):
    import ase
    ds = np.load(fname)
    if not pbc:
        return [ase.Atoms(#ds['name'],
        positions=pos, symbols=None, numbers=ds['z'], pbc=False, cell=[0,0,0]) for pos in ds['R']]


def gif2npy(fn, mode='RGBA'):
    from PIL import Image, ImageSequence
    with Image.open(fn) as im:
        return np.array([
            np.array(frame.convert(mode))
            for frame in ImageSequence.Iterator(im)
        ])


def get_associated_filename(file, suffix, default, ok_not_found=False):
    fn = file + suffix
    if os.path.exists(fn):
        return fn
    elif os.path.exists(os.path.dirname(os.path.abspath(file))+"/"+default):
        return os.path.dirname(os.path.abspath(file))+"/"+default
    elif os.path.exists(file):
        return file
    else:
        raise ValueError(f"Cannot fine {file} with suffix {suffix} or default {default} in same folder")

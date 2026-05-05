
try:
    from e3nn import o3
    def rand_rot():
        return o3.rand_matrix()
except:
    from scipy.spatial.transform import Rotation as R
    def rand_rot():
        return R.random().as_matrix()


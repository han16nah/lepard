import os, sys, glob, torch
# sys.path.append("../")
[sys.path.append(i) for i in ['.', '..']]
import numpy as np
import torch
import random
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from collections import OrderedDict
from lib.benchmark_utils import to_o3d_pcd, to_tsfm, KDTree_corr, get_correspondences, find_new_corr
from lib.utils import load_obj


class _Plants(Dataset):

    def __init__(self, config, split, data_augmentation=False):
        super(_Plants, self).__init__()

        assert split in ['train','val','test']

        if 'overfit' in config.exp_dir:
            d_slice = config.batch_size
        else :
            d_slice = None

        self.entries = self.read_entries(  config.split[split] , config.data_root, d_slice=d_slice )

        self.base_dir = config.data_root
        self.data_augmentation = data_augmentation
        self.config = config

        self.rot_factor = 1.
        self.augment_noise = config.augment_noise
        self.max_points = 200_000

        self.overlap_radius = 0.0375

        self.cache = OrderedDict()
        self.cache_size = 30_000


    def read_entries (self, split, data_root, d_slice=None, shuffle= False):
        entries = glob.glob(os.path.join(data_root, split, "*/*.npz"))
        if shuffle:
            random.shuffle(entries)
        if d_slice:
            return np.array(entries[:d_slice]).astype(np.bytes_)
        return np.array(entries).astype(np.bytes_)


    def __len__(self):
        return len(self.entries )


    def __getitem__(self, index, debug=False):

        if index in self.cache:
            entry = self.cache.pop(index)
            self.cache[index] = entry  # mark as recently used
        else:
            entry = np.load(self.entries[index])
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)  # evict least recently used
            self.cache[index] = entry


        # get transformation
        rot = entry['rot']
        trans = entry['trans']
        s2t_flow = entry['s2t_flow']
        src_pcd = entry['s_pc']
        tgt_pcd = entry['t_pc']
        correspondences = entry['correspondences']
        src_pcd_deformed = src_pcd + s2t_flow
        if "metric_index" in entry:
            metric_index = entry['metric_index'].squeeze()
        else:
            metric_index = None
        overlap_ratio = correspondences.shape[0] / src_pcd.shape[0]
        num_matches = correspondences.shape[0]
        # Centering (like we do in DeformationPyramid)
        all_points = np.vstack([src_pcd, tgt_pcd])
        center = all_points.mean(axis=0, keepdims=True)
        src_pcd = src_pcd - center
        tgt_pcd = tgt_pcd - center



        # if we get too many points, we do some downsampling
        #if (src_pcd.shape[0] > self.max_points):
        #    idx = np.random.permutation(src_pcd.shape[0])[:self.max_points]
        #    src_pcd = src_pcd[idx]
        #if (tgt_pcd.shape[0] > self.max_points):
        #    idx = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
        #    tgt_pcd = tgt_pcd[idx]

        downsampled = False
        # if we get too many points, we do some downsampling
        if src_pcd.shape[0] > self.max_points:
            print("Downsampling...")
            downsampled = True
            pts_max = min(src_pcd.shape[0], tgt_pcd.shape[0])
            sub_idx_src = np.random.permutation(pts_max)[:self.max_points]
            src_pcd = src_pcd[sub_idx_src]
            s2t_flow = s2t_flow[sub_idx_src]
            # indices of target - no filtering
            sub_idx_tgt = np.arange(tgt_pcd.shape[0])

        if (tgt_pcd.shape[0] > self.max_points):
            print("Downsampling...")
            sub_idx_tgt = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
            tgt_pcd = tgt_pcd[sub_idx_tgt]
            if not downsampled:
                sub_idx_src = np.arange(src_pcd.shape[0])
        
        src_pcd_deformed = src_pcd + s2t_flow
        if downsampled:
            correspondences = find_new_corr(correspondences, sub_idx_src, sub_idx_tgt)
            # assert that none of the important variables are empty
            assert src_pcd.shape[0] > 0, "Source point cloud is empty after downsampling."
            assert tgt_pcd.shape[0] > 0, "Target point cloud is empty after downsampling."
            assert correspondences.shape[0] > 0, "Correspondences are empty after downsampling."
            assert s2t_flow.shape[0] > 0, "Scene flow is empty after downsampling."

        if debug:
            #import mayavi.mlab as mlab
            c_red = (224. / 255., 0 / 255., 125 / 255.)
            c_pink = (224. / 255., 75. / 255., 232. / 255.)
            c_blue = (0. / 255., 0. / 255., 255. / 255.)
            #scale_factor = 0.013
            #src_wrapped = (np.matmul( rot, src_pcd_deformed.T ) + trans ).T
            #mlab.points3d(src_wrapped[:, 0], src_wrapped[:, 1], src_wrapped[:, 2], scale_factor=scale_factor, color=c_pink)
            #mlab.points3d(src_pcd[ :, 0] , src_pcd[ :, 1], src_pcd[:,  2], scale_factor=scale_factor , color=c_red)
            #mlab.points3d(tgt_pcd[ :, 0] , tgt_pcd[ :, 1], tgt_pcd[:,  2], scale_factor=scale_factor , color=c_blue)
            #mlab.show()
            import open3d as o3d
            src_o3d = o3d.geometry.PointCloud()
            tgt_o3d = o3d.geometry.PointCloud()
            src_wrapped = (np.matmul( rot, src_pcd_deformed.T ) + trans ).T
            src_wrapped_o3d = o3d.geometry.PointCloud()
            src_o3d.points = o3d.utility.Vector3dVector(src_pcd)
            tgt_o3d.points = o3d.utility.Vector3dVector(tgt_pcd)
            src_wrapped_o3d.points = o3d.utility.Vector3dVector(src_wrapped)
            src_o3d.paint_uniform_color(c_red)
            tgt_o3d.paint_uniform_color(c_blue)
            src_wrapped_o3d.paint_uniform_color(c_pink)
            o3d.visualization.draw_geometries([src_o3d, tgt_o3d, src_wrapped_o3d])


        # add gaussian noise
        if self.data_augmentation:
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2 / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                src_pcd = np.matmul(rot_ab, src_pcd.T).T
                src_pcd_deformed = np.matmul(rot_ab, src_pcd_deformed.T).T
                rot = np.matmul(rot, rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T
                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)

            src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise
            s2t_flow = src_pcd_deformed - src_pcd


        if debug:
            # wrapp_src = (np.matmul(rot, src_pcd.T)+ trans).T
            src_wrapped = (np.matmul( rot, src_pcd_deformed.T ) + trans ).T
            #mlab.points3d(src_wrapped[:, 0], src_wrapped[:, 1], src_wrapped[:, 2], scale_factor=scale_factor, color=c_red)
            #mlab.points3d(tgt_pcd[:, 0], tgt_pcd[:, 1], tgt_pcd[:, 2], scale_factor=scale_factor, color=c_blue)
            #mlab.show()
            src_wrapped_o3d = o3d.geometry.PointCloud()
            src_wrapped_o3d.points = o3d.utility.Vector3dVector(src_wrapped)
            src_wrapped_o3d.paint_uniform_color(c_red)
            o3d.visualization.draw_geometries([tgt_o3d, src_wrapped_o3d])


        if (trans.ndim == 1):
            trans = trans[:, None]


        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)


        #R * ( Ps + flow ) + t  = Pt
        return src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, overlap_ratio, num_matches, rot, trans, s2t_flow, metric_index



if __name__ == '__main__':
    pass
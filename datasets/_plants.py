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


def plot_corr(s_pc, t_pc, corr, img_path):
    import open3d as o3d
    import copy
    # for plotting, we need to arrange the points pertically
    points_combined = np.vstack([s_pc, t_pc])
    # and then modify the correspondences of the target points
    correspondences_st_plotting = copy.deepcopy(corr)
    correspondences_st_plotting[:, 1] += s_pc.shape[0]  # shift target indices by the number of source points
    cloudf_src = o3d.geometry.PointCloud()
    cloudf_src.points = o3d.utility.Vector3dVector(s_pc)
    cloudf_src.paint_uniform_color([1, 0, 0])  # red
    cloudf_tgt = o3d.geometry.PointCloud()
    cloudf_tgt.points = o3d.utility.Vector3dVector(t_pc)
    cloudf_tgt.paint_uniform_color([0, 0, 1])  # blue
    corres_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_combined),
        lines=o3d.utility.Vector2iVector(correspondences_st_plotting))
    corres_lineset.paint_uniform_color([1, 0, 0])  # red
    #vis = o3d.visualization.Visualizer()
    #vis.create_window(visible=False)
    #vis.add_geometry(cloudf_src)
    #vis.add_geometry(cloudf_tgt)
    #vis.add_geometry(corres_lineset)
    #vis.update_geometry(cloudf_src)
    #vis.update_geometry(cloudf_tgt)
    #vis.update_geometry(corres_lineset)
    #vis.poll_events()
    #vis.update_renderer()
    #vis.capture_screen_image(img_path, do_render=True)
    #vis.destroy_window()
    o3d.visualization.draw_geometries([cloudf_src, cloudf_tgt, corres_lineset])


def plot_flow(s_pc, t_pc, s2t_flow, img_path):
    import open3d as o3d
    cloudf_src = o3d.geometry.PointCloud()
    cloudf_src.points = o3d.utility.Vector3dVector(s_pc)
    cloudf_src.paint_uniform_color([1, 0, 0])  # red
    cloudf_tgt = o3d.geometry.PointCloud()
    cloudf_tgt.points = o3d.utility.Vector3dVector(t_pc)
    cloudf_tgt.paint_uniform_color([0, 0, 1])  # blue
    corres_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack([s_pc, s_pc + s2t_flow])),
        lines=o3d.utility.Vector2iVector(np.arange(2 * s2t_flow.shape[0]).reshape((2, -1)).T))
    corres_lineset.paint_uniform_color([1, 0, 0])  # red
    #vis = o3d.visualization.Visualizer()
    #vis.create_window(visible=False)
    #vis.add_geometry(cloudf_src)
    #vis.add_geometry(cloudf_tgt)
    #vis.add_geometry(corres_lineset)
    #vis.update_geometry(cloudf_src)
    #vis.update_geometry(cloudf_tgt)
    #vis.update_geometry(corres_lineset)
    #vis.poll_events()
    #vis.update_renderer()
    #vis.capture_screen_image(img_path, do_render=True)
    #vis.destroy_window()
    o3d.visualization.draw_geometries([cloudf_src, cloudf_tgt, corres_lineset])


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
        self.max_points = 50_000

        self.overlap_radius = 0.0375

        self.cache = OrderedDict()
        self.cache_size = 1_000


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



        # if we get too many points, we do some downsampling
        #if (src_pcd.shape[0] > self.max_points):
        #    idx = np.random.permutation(src_pcd.shape[0])[:self.max_points]
        #    src_pcd = src_pcd[idx]
        #if (tgt_pcd.shape[0] > self.max_points):
        #    idx = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
        #    tgt_pcd = tgt_pcd[idx]

        downsampled = False
        src_mask = None
        tgt_mask = None
        # if we get too many points, we do some downsampling
        if src_pcd.shape[0] > self.max_points:
            print("Downsampling source...")
            downsampled = True
            # Create boolean mask for source points
            src_mask = np.zeros(src_pcd.shape[0], dtype=bool)
            sub_idx_src = np.random.choice(src_pcd.shape[0], self.max_points, replace=False)
            src_mask[sub_idx_src] = True
            
            src_pcd = src_pcd[src_mask]
            s2t_flow = s2t_flow[src_mask]
            
            # For target, keep all points initially
            tgt_mask = np.ones(tgt_pcd.shape[0], dtype=bool)

        if tgt_pcd.shape[0] > self.max_points:
            print("Downsampling target...")
            # Create boolean mask for target points
            tgt_mask = np.zeros(tgt_pcd.shape[0], dtype=bool)
            sub_idx_tgt = np.random.choice(tgt_pcd.shape[0], self.max_points, replace=False)
            tgt_mask[sub_idx_tgt] = True
            
            tgt_pcd = tgt_pcd[tgt_mask]
            
            # If source wasn't downsampled, create full mask for it
            if not downsampled:
                src_mask = np.ones(src_pcd.shape[0], dtype=bool)
                downsampled = True
        
        src_pcd_deformed = src_pcd + s2t_flow
        if downsampled:
            # might not be needed to recompute correspondences, but just to be safe
            correspondences = find_new_corr(correspondences, src_mask, tgt_mask)
            # assert that none of the important variables are empty
            assert src_pcd.shape[0] > 0, "Source point cloud is empty after downsampling."
            assert tgt_pcd.shape[0] > 0, "Target point cloud is empty after downsampling."
            assert correspondences.shape[0] > 0, "Correspondences are empty after downsampling."
            assert s2t_flow.shape[0] > 0, "Scene flow is empty after downsampling."
            #plot_corr(src_pcd, tgt_pcd, correspondences, "debug_downsampled_corr.png")
            #plot_flow(src_pcd, tgt_pcd, s2t_flow, "debug_downsampled_flow.png")


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
        return src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, rot, trans, s2t_flow, metric_index



if __name__ == '__main__':
    pass
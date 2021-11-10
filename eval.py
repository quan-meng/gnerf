import os
import argparse
import numpy as np
import torch
import json
from kornia.geometry.conversions import rotation_matrix_to_quaternion

from tools.ray_utils import look_at_rotation, r6d2mat

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate estimated poses (blender dataset)')
    parser.add_argument('--ckpt', help='pretrained checkpoint path to load')
    parser.add_argument('--gt', type=str, help='the path of the json file with ground-truth poses')

    args = parser.parse_args()

    out_dir = './evaluation/'
    os.mkdir(out_dir) if not os.path.exists(out_dir) else print('directory exits')

    print('load estimated poses from checkpoint')
    checkpoint = torch.load(args.ckpt)
    raw_poses = checkpoint['t']['poses_embed'].cpu().detach()

    if raw_poses.shape[-1] == 6:
        t = raw_poses[:, :3]  # [N, 3]
        r = raw_poses[:, 3:]
        R = r6d2mat(r)[:, :3, :3]  # [N, 3, 3]
    else:
        t = raw_poses[:, :3]  # [N, 3]
        R = look_at_rotation(t)
    poses_pred = torch.cat((R, t[..., None]), -1)

    t_pred = poses_pred[:, :, 3]  # [N, 3]
    R_pred = poses_pred[:, :3, :3]  # [N, 3, 3]

    gt_qua = rotation_matrix_to_quaternion(R_pred.contiguous())  # [N, 4]

    out_lines = torch.cat((t_pred, gt_qua), -1)  # [N, 7]
    out_lines = out_lines.tolist()
    out_lines = [' '.join(str(e) for e in [i] + v) for i, v in enumerate(out_lines)]
    # output to file stamped_traj_estimate.txt
    with open(os.path.join(out_dir, 'stamped_traj_estimate.txt'), 'w') as f:
        f.writelines('\r\n'.join(out_lines))

    print('load ground-truth poses from the dataset')
    with open(args.gt, 'r') as f:
        meta = json.load(f)

    gt_poses = []
    for i, frame in enumerate(meta['frames']):
        pose = np.array(frame['transform_matrix'])[:3, :4]
        c2w = torch.FloatTensor(pose)

        gt_poses.append(c2w[None])
    gt_poses = torch.cat(gt_poses)  # [N, 3, 4]

    gt_t = gt_poses[:, :, 3]  # [N, 3]
    gt_R = gt_poses[:, :3, :3]  # [N, 3, 3]

    gt_qua = rotation_matrix_to_quaternion(gt_R.contiguous())  # [N, 4]

    out_lines = torch.cat((gt_t, gt_qua), -1)  # [N, 7]

    out_lines = out_lines.tolist()
    out_lines = [' '.join(str(e) for e in [i] + v) for i, v in enumerate(out_lines)]
    # output to file stamped_groundtruth.txt
    with open(os.path.join(out_dir, 'stamped_groundtruth.txt'), 'w') as f:
        f.writelines('\r\n'.join(out_lines))


"""util functions
# many old functions, need to clean up
# homography --> homography
# warping
# loss --> delete if useless
"""
import numpy as np
from pathlib import Path
import datetime
import cv2

import paddle
import paddle.nn.functional as F
import paddle.nn as nn

from collections import OrderedDict
from utils.d2s import DepthToSpace
from utils.d2s import SpaceToDepth


def img_overlap(img_r, img_g, img_gray):
    def to_3d(img):
        if len(img.shape) == 2:
            img = img[np.newaxis, ...]
        return img
    img_r, img_g, img_gray = to_3d(img_r), to_3d(img_g), to_3d(img_gray)
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img


def thd_img(img, thd=0.015):
    img[img < thd] = 0
    img[img >= thd] = 1
    return img


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()


def save_path_formatter(args, parser):
    print('todo: save path')
    return Path('.')
    pass



def tensor2array(tensor, max_value=255, colormap='rainbow', channel_first=True):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if int(cv2.__version__[0]) >= 3:
                color_cvt = cv2.COLOR_BGR2RGB
            else:
                color_cvt = cv2.cv.CV_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255 * tensor.squeeze().numpy() / max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32) / 255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy
                () / max_value).clip(0, 1)
        if channel_first:
            array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert tensor.size(0) == 3
        array = 0.5 + tensor.numpy() * 0.5
        if not channel_first:
            array = array.transpose(1, 2, 0)
    return array


def find_files_with_ext(directory, extension='.npz'):
    list_of_files = []
    import os
    if extension == '.npz':
        for l in os.listdir(directory):
            if l.endswith(extension):
                list_of_files.append(l)
        return list_of_files


def save_checkpoint(save_path, net_state, epoch, filename='checkpoint.pdiparams.tar'):
    file_prefix = ['superPointNet']

    filename = '{}_{}_{}'.format(file_prefix[0], str(epoch), filename)
    paddle.save(net_state, save_path / filename)
    print('save checkpoint to ', filename)
    pass


def load_checkpoint(load_path, filename='checkpoint.pdiparams.tar'):
    file_prefix = ['superPointNet']
    filename = '{}__{}'.format(file_prefix[0], filename)

    checkpoint = paddle.load(load_path / filename)
    print('load checkpoint from ', filename)
    return checkpoint
    pass


def saveLoss(filename, iter, loss, task='train', **options):
    with open(filename, 'a') as myfile:
        myfile.write(task + ' iter: ' + str(iter) + ', ')
        myfile.write('loss: ' + str(loss) + ', ')
        myfile.write(str(options))
        myfile.write('\n')


def saveImg(img, filename):
    import cv2
    cv2.imwrite(filename, img)


def pltImshow(img):
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.show()


def loadConfig(filename):
    import yaml
    with open(filename, 'r') as f:
        config = yaml.load(f)
    return config


def append_csv(file='foo.csv', arr=[]):
    import csv
    with open(file, 'a') as f:
        writer = csv.writer(f)
        if type(arr[0]) is list:
            for a in arr:
                writer.writerow(a)
        else:
            writer.writerow(arr)




def sample_homography(inv_scale=3):
    corner_img = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    img_offset = corner_img
    corner_map = (np.random.rand(4, 2) - 0.5) * 2 / (inv_scale + 0.01
        ) + img_offset
    matrix = cv2.getPerspectiveTransform(np.float32(corner_img), np.float32
        (corner_map))
    return matrix


def sample_homographies(batch_size=1, scale=10, device='cpu'):
    mat_H = [sample_homography(inv_scale=scale) for i in range(batch_size)]
    mat_H = np.stack(mat_H, axis=0)
    mat_H = paddle.to_tensor(mat_H, dtype=paddle.float32)
    mat_H = mat_H.to(device)
    mat_H_inv = paddle.stack([paddle.inverse(mat_H[i, :, :]) for i in range(batch_size)])
    mat_H_inv = paddle.to_tensor(mat_H_inv, dtype=paddle.float32)
    mat_H_inv = mat_H_inv.to(device)
    return mat_H, mat_H_inv


def warpLabels(pnts, homography, H, W):
    import paddle
    """
    input:
        pnts: numpy
        homography: numpy
    output:
        warped_pnts: numpy
    """
    from utils.utils import warp_points
    from utils.utils import filter_points
    pnts = paddle.to_tensor(pnts).long()
    homography = paddle.to_tensor(homography, dtype=paddle.float32)
    warped_pnts = warp_points(paddle.stack((pnts[:, (0)], pnts[:, (1)]),
        axis=1), homography)
    warped_pnts = filter_points(warped_pnts, paddle.to_tensor([W, H])).round(
        ).long()
    return warped_pnts.numpy()


def warp_points_np(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2).
        homography: batched or not (shapes (B, 3, 3) and ... respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.
    """
    batch_size = homographies.shape[0]
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    warped_points = np.tensordot(homographies, points.transpose(), axes=([2
        ], [0]))
    warped_points = warped_points.reshape([batch_size, 3, -1])
    warped_points = warped_points.transpose([0, 2, 1])
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points


def homography_scaling(homography, H, W):
    trans = np.array([[2.0 / W, 0.0, -1], [0.0, 2.0 / H, -1], [0.0, 0.0, 1.0]])
    homography = np.linalg.inv(trans) @ homography @ trans
    return homography


def homography_scaling_torch(homography, H, W):
    trans = paddle.to_tensor([[2.0 / W, 0.0, -1], [0.0, 2.0 / H, -1], [0.0,
        0.0, 1.0]])
    homography = trans.inverse() @ homography @ trans
    return homography


def filter_points(points, shape, return_mask=False):
    points = points.float()
    shape = shape.float()
    mask = (points >= 0) * (points <= shape - 1)
    mask = (paddle.prod(mask, axis=-1) == 1)
    if return_mask:
        return points[mask], mask
    return points[mask]


def warp_points(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2(x, y))).
        homography: batched or not (shapes (B, 3, 3) and ... respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2(x, y)) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.

    """
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies
    batch_size = homographies.shape[0]
    points = paddle.concat((points.float(), paddle.ones((points.shape
        [0], 1)).requires_grad_(False).to(device)), axis=1)
    points = points.to(device)
    homographies = homographies.view(batch_size * 3, 3)
    warped_points = homographies @ points.transpose(0, 1)
    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points[0, :, :] if no_batches else warped_points


def inv_warp_image_batch(img, mat_homo_inv, device='cpu', mode='bilinear'):
    """
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [batch_size, 1, H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [batch_size, 3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [batch_size, 1, H, W]
    """
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1, 1, img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1, 3, 3)
    Batch, channel, H, W = img.shape
    coor_cells = paddle.stack(paddle.meshgrid(paddle.linspace(-1, 1, W), paddle.linspace(-1, 1, H)), axis=2)
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous()
    src_pixel_coords = warp_points(coor_cells.view([-1, 2]), mat_homo_inv,
        device)
    src_pixel_coords = src_pixel_coords.view([Batch, H, W, 2])
    src_pixel_coords = src_pixel_coords.float()
    warped_img = F.grid_sample(img, src_pixel_coords, mode=mode, align_corners=True)
    return warped_img


def inv_warp_image(img, mat_homo_inv, device='cpu', mode='bilinear'):
    """
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [H, W]
    """
    warped_img = inv_warp_image_batch(img, mat_homo_inv, device, mode)
    return warped_img.squeeze()


def labels2Dto3D(labels, cell_size, add_dustbin=True):
    """
    Change the shape of labels into 3D. Batch of labels.

    :param labels:
        tensor [batch_size, 1, H, W]
        keypoint map.
    :param cell_size:
        8
    :return:
         labels: tensors[batch_size, 65, Hc, Wc]
    """
    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    labels = space2depth(labels)
    if add_dustbin:
        dustbin = labels.sum(dim=1)
        dustbin = 1 - dustbin
        dustbin[dustbin < 1.0] = 0
        labels = paddle.concat((labels, dustbin.view(batch_size, 1,
            Hc, Wc)), axis=1)
        dn = labels.sum(dim=1)
        labels = labels.div(paddle.unsqueeze(dn, 1))
    return labels


def labels2Dto3D_flattened(labels, cell_size):
    """
    Change the shape of labels into 3D. Batch of labels.

    :param labels:
        tensor [batch_size, 1, H, W]
        keypoint map.
    :param cell_size:
        8
    :return:
         labels: tensors[batch_size, 65, Hc, Wc]
    """
    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    labels = space2depth(labels)
    dustbin = paddle.ones((batch_size, 1, Hc, Wc)).requires_grad_(False).cuda()
    labels = paddle.concat((labels * 2, dustbin.view(batch_size, 1,
        Hc, Wc)), axis=1)
    labels = paddle.argmax(labels, axis=1)
    return labels


def old_flatten64to1(semi, tensor=False):
    """
    Flatten 3D np array to 2D

    :param semi:
        np [64 x Hc x Wc]
        or
        tensor (batch_size, 65, Hc, Wc)
    :return:
        flattened map
        np [1 x Hc*8 x Wc*8]
        or
        tensor (batch_size, 1, Hc*8, Wc*8)
    """
    if tensor:
        is_batch = len(semi.size()) == 4
        if not is_batch:
            semi = semi.unsqueeze_(0)
        Hc, Wc = semi.size()[2], semi.size()[3]
        cell = 8
        semi.transpose_(1, 2)
        semi.transpose_(2, 3)
        semi = semi.view(-1, Hc, Wc, cell, cell)
        semi.transpose_(2, 3)
        semi = semi.contiguous()
        semi = semi.view(-1, 1, Hc * cell, Wc * cell)
        heatmap = semi
        if not is_batch:
            heatmap = heatmap.squeeze_(0)
    else:
        Hc, Wc = semi.shape[1], semi.shape[2]
        cell = 8
        semi = semi.transpose(1, 2, 0)
        heatmap = np.reshape(semi, [Hc, Wc, cell, cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * cell, Wc * cell])
        heatmap = heatmap[np.newaxis, :, :]
    return heatmap


def flattenDetection(semi, tensor=False):
    """
    Flatten detection output

    :param semi:
        output from detector head
        tensor [65, Hc, Wc]
        :or
        tensor (batch_size, 65, Hc, Wc)

    :return:
        3D heatmap
        np (1, H, C)
        :or
        tensor (batch_size, 65, Hc, Wc)

    """
    batch = False
    if len(semi.shape) == 4:
        batch = True
        batch_size = semi.shape[0]
    if batch:
        dense = nn.functional.softmax(semi, axis=1)
        nodust = dense[:, :-1, :, :]
    else:
        dense = nn.functional.softmax(semi, axis=0)
        nodust = dense[:-1, :, :].unsqueeze(0)
    depth2space = DepthToSpace(8)
    heatmap = depth2space(nodust)
    heatmap = heatmap.squeeze(0) if not batch else heatmap
    return heatmap


def sample_homo(image):
    import tensorflow as tf
    from utils.homographies import sample_homography
    H = sample_homography(tf.shape(image)[:2])
    with tf.Session():
        H_ = H.eval()
    H_ = np.concatenate((H_, np.array([1])[:, np.newaxis]), axis=1)
    mat = np.reshape(H_, (3, 3))
    return mat


import cv2


def getPtsFromHeatmap(heatmap, conf_thresh, nms_dist):
    """
    :param self:
    :param heatmap:
        np (H, W)
    :return:
    """
    border_remove = 4
    H, W = heatmap.shape[0], heatmap.shape[1]
    xs, ys = np.where(heatmap >= conf_thresh)
    sparsemap = heatmap >= conf_thresh
    if len(xs) == 0:
        return np.zeros((3, 0))
    pts = np.zeros((3, len(xs)))
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]
    bord = border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= W - bord)
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= H - bord)
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts


def box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):
    from torchvision.ops import nms
    """Performs non maximum suppression on the heatmap by considering hypothetical
    bounding boxes centered at each pixel's location (e.g. corresponding to the receptive
    field). Optionally only keeps the top k detections.
    Arguments:
    prob: the probability heatmap, with shape `[H, W]`.
    size: a scalar, the size of the bouding boxes.
    iou: a scalar, the IoU overlap threshold.
    min_prob: a threshold under which all probabilities are discarded before NMS.
    keep_top_k: an integer, the number of top scores to keep.
    """
    pts = paddle.nonzero(prob > min_prob).float() # [N, 2]
    prob_nms = paddle.full_like(prob).requires_grad_(False)
    if pts.nelement() == 0:
        return prob_nms
    size = paddle.to_tensor(size / 2.0).cuda()
    boxes = paddle.concat([pts - size, pts + size], axis=1)
    scores = prob[pts[:, 0].long(), pts[:, 1].long()]
    if keep_top_k != 0:
        indices = nms(boxes, scores, iou)
    else:
        raise NotImplementedError
    pts = paddle.index_select(pts, 0, indices)
    scores = paddle.index_select(scores, 0, indices)
    prob_nms[pts[:, 0].long(), pts[:, 1].long()] = scores
    return prob_nms


def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)
    inds = np.zeros((H, W)).astype(int)
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros(1).astype(int)
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    count = 0
    for i, rc in enumerate(rcorners.T):
        pt = rc[0] + pad, rc[1] + pad
        if grid[pt[1], pt[0]] == 1:
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds


def compute_valid_mask(image_shape, inv_homography, device='cpu',
    erosion_radius=0):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.

    Arguments:
        input_shape: Tensor of rank 2 representing the image shape, i.e. `[H, W]`.
        homography: Tensor of shape (B, 8) or (8,), where B is the batch size.
        `erosion_radius: radius of the margin to be discarded.

    Returns: a Tensor of type `tf.int32` and shape (H, W).
    """
    if inv_homography.dim() == 2:
        inv_homography = inv_homography.view(-1, 3, 3)
    batch_size = inv_homography.shape[0]
    mask = paddle.ones([batch_size, 1, image_shape[0], image_shape[1]]
        ).requires_grad_(False).to(device)
    mask = inv_warp_image_batch(mask, inv_homography, device=device, mode='nearest')
    mask = mask.view(batch_size, image_shape[0], image_shape[1])
    mask = mask.cpu().numpy()
    if erosion_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (erosion_radius * 2,) * 2)
        for i in range(batch_size):
            mask[i, :, :] = cv2.erode(mask[i, :, :], kernel, iterations=1)
    return paddle.to_tensor(mask).to(device)


def normPts(pts, shape):
    """
    normalize pts to [-1, 1]
    :param pts:
        tensor (y, x)
    :param shape:
        tensor shape (y, x)
    :return:
    """
    pts = pts / shape * 2 - 1
    return pts


def denormPts(pts, shape):
    """
    denormalize pts back to H, W
    :param pts:
        tensor (y, x)
    :param shape:
        numpy (y, x)
    :return:
    """
    pts = (pts + 1) * shape / 2
    return pts


def descriptor_loss(descriptors, descriptors_warped, homographies,
    mask_valid=None, cell_size=8, lamda_d=250, device='cpu',
    descriptor_dist=4, **config):
    """
    Compute descriptor loss from descriptors_warped and given homographies

    :param descriptors:
        Output from descriptor head
        tensor [batch_size, descriptors, Hc, Wc]
    :param descriptors_warped:
        Output from descriptor head of warped image
        tensor [batch_size, descriptors, Hc, Wc]
    :param homographies:
        known homographies
    :param cell_size:
        8
    :param device:
        gpu or cpu
    :param config:
    :return:
        loss, and other tensors for visualization
    """
    homographies = homographies.to(device)
    from utils.utils import warp_points
    lamda_d = lamda_d
    margin_pos = 1
    margin_neg = 0.2
    batch_size, Hc, Wc = descriptors.shape[0], descriptors.shape[2
        ], descriptors.shape[3]
    H, W = Hc * cell_size, Wc * cell_size
    with paddle.no_grad():
        shape = paddle.to_tensor([H, W], dtype=paddle.float32).to(device)

        coor_cells = paddle.stack(paddle.meshgrid(paddle.arange(Hc), paddle.arange(Wc)), axis=2)
        coor_cells = paddle.to_tensor(coor_cells, dtype=paddle.float32).to(device)
        coor_cells = coor_cells * cell_size + cell_size // 2

        coor_cells = coor_cells.view([-1, 1, 1, Hc, Wc, 2])
        warped_coor_cells = normPts(coor_cells.view([-1, 2]), shape)
        warped_coor_cells = paddle.stack((warped_coor_cells[:, (1)], warped_coor_cells[:, (0)]), axis=1)
        warped_coor_cells = warp_points(warped_coor_cells, homographies, device)

        warped_coor_cells = paddle.stack((warped_coor_cells[:, :, (1)], warped_coor_cells[:, :, (0)]), axis=2)
        shape_cell = paddle.to_tensor([H // cell_size, W // cell_size], dtype=paddle.float32).to(device)

        warped_coor_cells = denormPts(warped_coor_cells, shape)
        warped_coor_cells = warped_coor_cells.view([-1, Hc, Wc, 1, 1, 2])

        cell_distances = coor_cells - warped_coor_cells
        cell_distances = paddle.norm(cell_distances, axis=-1)
        mask = cell_distances <= descriptor_dist

        mask = paddle.to_tensor(mask, dtype=paddle.float32).to(device)

    descriptors = descriptors.transpose(1, 2).transpose(2, 3)
    descriptors = descriptors.view((batch_size, Hc, Wc, 1, 1, -1))
    descriptors_warped = descriptors_warped.transpose(1, 2).transpose(2, 3)
    descriptors_warped = descriptors_warped.view((batch_size, 1, 1, Hc, Wc, -1))
    dot_product_desc = descriptors * descriptors_warped
    dot_product_desc = dot_product_desc.sum(dim=-1)
    positive_dist = paddle.max(margin_pos - dot_product_desc, paddle.to_tensor(0.0).to(device))
    negative_dist = paddle.max(dot_product_desc - margin_neg, paddle.to_tensor(0.0).to(device))

    if mask_valid is None:
        mask_valid = paddle.ones([batch_size, 1, Hc * cell_size, Wc * cell_size]).requires_grad_(False)
    mask_valid = mask_valid.view(batch_size, 1, 1, mask_valid.shape[2], mask_valid.shape[3])

    loss_desc = lamda_d * mask * positive_dist + (1 - mask) * negative_dist
    loss_desc = loss_desc * mask_valid

    normalization = batch_size * (mask_valid.sum() + 1) * Hc * Wc
    pos_sum = (lamda_d * mask * positive_dist / normalization).sum()
    neg_sum = ((1 - mask) * negative_dist / normalization).sum()
    loss_desc = loss_desc.sum() / normalization

    return loss_desc, mask, pos_sum, neg_sum


def sumto2D(ndtensor):
    return ndtensor.sum(dim=1).sum(dim=1)


def mAP(pred_batch, labels_batch):
    pass


def precisionRecall_torch(pred, labels):
    offset = 10 ** -6
    assert pred.size() == labels.size(), 'Sizes of pred, labels should match when you get the precision/recall!'
    precision = paddle.sum(pred * labels) / (paddle.sum(pred) + offset)
    recall = paddle.sum(pred * labels) / (paddle.sum(labels) + offset)
    if precision.item() > 1.0:
        print(pred)
        print(labels)
        import scipy.io.savemat as savemat
        savemat('pre_recall.mat', {'pred': pred, 'labels': labels})
    assert precision.item() <= 1.0 and precision.item() >= 0.0
    return {'precision': precision, 'recall': recall}


def precisionRecall(pred, labels, thd=None):
    offset = 10 ** -6
    if thd is None:
        precision = np.sum(pred * labels) / (np.sum(pred) + offset)
        recall = np.sum(pred * labels) / (np.sum(labels) + offset)
    return {'precision': precision, 'recall': recall}


def getWriterPath(task='train', exper_name='', date=True):
    import datetime
    prefix = 'runs/'
    str_date_time = ''
    if exper_name != '':
        exper_name += '_'
    if date:
        str_date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    return prefix + task + '/' + exper_name + str_date_time


def crop_or_pad_choice(in_num_points, out_num_points, shuffle=False):
    """Crop or pad point cloud to a fixed number; return the indexes
    Args:
        points (np.ndarray): point cloud. (n, d)
        num_points (int): the number of output points
        shuffle (bool): whether to shuffle the order
    Returns:
        np.ndarray: output point cloud
        np.ndarray: index to choose input points
    """
    if shuffle:
        choice = np.random.permutation(in_num_points)
    else:
        choice = np.arange(in_num_points)
    assert out_num_points > 0, 'out_num_points = %d must be positive int!' % out_num_points
    if in_num_points >= out_num_points:
        choice = choice[:out_num_points]
    else:
        num_pad = out_num_points - in_num_points
        pad = np.random.choice(choice, num_pad, replace=True)
        choice = np.concatenate([choice, pad])
    return choice

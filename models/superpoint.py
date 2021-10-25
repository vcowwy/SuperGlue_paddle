from pathlib import Path

import paddle
from paddle import nn


def simple_nms(scores, nms_radius: int):
    assert nms_radius >= 0

    def max_pool(x):
        return paddle.nn.functional.max_pool2d(x, kernel_size=nms_radius*2+1,
                                               stride=1, padding=nms_radius)

    zeros = paddle.full_like(scores).requires_grad_(False)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = paddle.where(supp_mask, zeros, scores)

        new_max_mask = supp_scores == max_pool(supp_scores)
        #max_mask = max_mask | new_max_mask & ~supp_mask
        max_mask = paddle.logical_and(paddle.logical_or(max_mask, new_max_mask),
                                      ~supp_mask)

    return paddle.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = paddle.logical_and((keypoints[:, 0] >= border), (keypoints[:, 0] < height - border))
    mask_w = paddle.logical_and((keypoints[:, 1] >= border), (keypoints[:, 1] < width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = paddle.topk(scores, k, axis=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int=8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= paddle.to_tensor([w * s - s / 2 - 0.5, h * s - s / 2 - 0.5]).to(keypoints)[None]
    keypoints = keypoints * 2 - 1

    #args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    args = {'align_coerners': True}

    descriptors = paddle.nn.functional.grid_sample(descriptors,
                                                   keypoints.view(b, 1, -1, 2),
                                                   mode='bilinear', **args)
    
    descriptors = paddle.nn.functional.normalize(descriptors.reshape(b, c, -1),
                                                 p=2,
                                                 axis=1)

    return descriptors


class SuperPoint(nn.Layer):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    default_config = {'descriptor_dim': 256,
                      'nms_radius': 4,
                      'keypoint_threshold': 0.005,
                      'max_keypoints': -1,
                      'remove_borders': 4}

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        #self.relu = paddle.nn.ReLU(inplace=True)
        self.relu = paddle.nn.ReLU()
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2D(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2D(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2D(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2D(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2D(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2D(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2D(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2D(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2D(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2D(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2D(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2D(c5, self.config['descriptor_dim'],
                                kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent / 'weights/superpoint_v1.pdparams'
        self.load_state_dict(paddle.load(str(path)))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('"max_keypoints" must be positive or "-1"')

        print('Loaded SuperPoint model')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = paddle.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, self.config['nms_radius'])

        keypoints = [paddle.nonzero(s > self.config['keypoint_threshold'])
                     for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h * 8, w * 8)
            for k, s in zip(keypoints, scores)]))

        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[top_k_keypoints(k, s, self.
                config['max_keypoints']) for k, s in zip(keypoints, scores)]))

        keypoints = [paddle.flip(k, [1]).float() for k in keypoints]

        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = paddle.nn.functional.normalize(descriptors, p=2, dim=1)

        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        return {'keypoints': keypoints,
                'scores': scores,
                'descriptors': descriptors}

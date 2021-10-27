"""Old version of SuperpointNet. Use it together with 
logs/magicpoint_synth20/checkpoints/superPointNet_200000_checkpoint.pdiparams.tar

"""
from models.unet_parts import *

import paddle

class SubpixelNet(paddle.nn.Layer):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self, subpixel_channel=1):
        super(SubpixelNet, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        self.inc = inconv(1, c1)
        self.down1 = down(c1, c2)
        self.down2 = down(c2, c3)
        self.down3 = down(c3, c4)
        self.up1 = up(c4 + c3, c2)
        self.up2 = up(c2 + c2, c1)
        self.up3 = up(c1 + c1, c1)
        self.outc = outconv(c1, subpixel_channel)
        self.relu = paddle.nn.ReLU()
        self.convPa = paddle.nn.Conv2D(c4, c5, kernel_size=3, stride=1,
            padding=1)
        self.bnPa = nn.BatchNorm2D(c5)
        self.convPb = paddle.nn.Conv2D(c5, det_h, kernel_size=1, stride=1,
            padding=0)
        self.bnPb = nn.BatchNorm2D(det_h)
        self.convDa = paddle.nn.Conv2D(c4, c5, kernel_size=3, stride=1,
            padding=1)
        self.bnDa = nn.BatchNorm2D(c5)
        self.convDb = paddle.nn.Conv2D(c5, d1, kernel_size=1, stride=1,
            padding=0)
        self.bnDb = nn.BatchNorm2D(d1)

    @staticmethod
    def soft_argmax_2d(patches):
        """
    params:
        patches: (B, N, H, W)
    return:
        coor: (B, N, 2)  (x, y)

    """
        import torchgeometry as tgm
        m = tgm.contrib.SpatialSoftArgmax2d()
        coords = m(patches)
        return coords

    def forward(self, x, subpixel=False):
        """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        cPa = self.bnPa(self.relu(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa))
        cDa = self.bnDa(self.relu(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))
        dn = paddle.norm(desc, p=2, axis=1) # Compute the norm.
        desc = desc.div(paddle.unsqueeze(dn, 1))
        if subpixel:
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            x = self.outc(x)
            return semi, desc, x
        return semi, desc


if __name__ == '__main__':
    device = 'cuda' if paddle.is_compiled_with_cuda() else 'cpu'
    device = device.replace('cuda', 'gpu')
    device = paddle.set_device(device)
    model = SubpixelNet()
    model = model.to(device)

    from paddle import summary
    summary(model, input_size=(1, 240, 320))

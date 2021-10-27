import paddle


def LongTensor(data, dtype=paddle.int64, device=None, requires_grad=True, pin_memory=False):
    return paddle.to_tensor(data, dtype=dtype, place=device, stop_gradient=paddle.logical_not(requires_grad))


def IntTensor(data, dtype=paddle.int8, device=None, requires_grad=True, pin_memory=False):
    return paddle.to_tensor(data, dtype=dtype, place=device, stop_gradient=paddle.logical_not(requires_grad))


def FloatTensor(data, dtype=paddle.int8, device=None, requires_grad=True, pin_memory=False):
    return paddle.to_tensor(data, dtype=dtype, place=device, stop_gradient=paddle.logical_not(requires_grad))
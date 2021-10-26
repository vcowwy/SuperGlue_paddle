import paddle


def Parameter(shape_or_tensor, fill_value=None, requires_grad=True):
    if isinstance(shape_or_tensor, paddle.Tensor):
        X = Parameter(shape_or_tensor.shape, 0.0)
        paddle.assign(shape_or_tensor.astype("float32"), X)
    else:
        if isinstance(shape_or_tensor, int):
            shape_or_tensor = [shape_or_tensor]

        X = paddle.create_parameter(
                        shape=shape_or_tensor, dtype="float32",
                        attr=paddle.ParamAttr(name=None, initializer=paddle.nn.initializer.Constant(value=fill_value)),
                        is_bias=False)
    if not requires_grad:
        X.stop_gradient = True

    return X
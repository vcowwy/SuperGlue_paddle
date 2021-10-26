import torch
import paddle


def torch2paddle():
    torch_path = "./superglue_indoor.pth"
    paddle_path = "./pd/superglue_indoor.pdparams"
    torch_state_dict = torch.load(torch_path)
    paddle_state_dict = {}
    state_dict = {k:v for k,v in torch_state_dict.items()}

    for k, v in state_dict.items():
        v = state_dict[k].detach().cpu().numpy()
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        paddle_state_dict[k] = v
    paddle.save(paddle_state_dict, paddle_path)


if __name__ == "__main__":
    torch2paddle()
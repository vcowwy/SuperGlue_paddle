import torch
import paddle


def torch2paddle():
    #torch_path = "../../../SuperGlue_paddle/models/weights/superglue_outdoor.pdparams"
    #torch_state_dict = paddle.load(torch_path)
    #print(torch_state_dict['bin_score'])
    torch_path = "superpoint_v1.pth"
    paddle_path = "superpoint_v1.pdparams"
    torch_state_dict = torch.load(torch_path)

    #fc_names = ["fc"]
    paddle_state_dict = {}
    print(torch_state_dict)
    state_dict = {k:v for k,v in torch_state_dict.items()}

    for k, v in state_dict.items():
        v = state_dict[k].detach().cpu().numpy()
        #flag = [i in k for i in fc_names]
        #if any(flag):
        #    v = v.transpose()
        #k = k.replace("running_var", "_variance")
        #k = k.replace("running_mean", "_mean")
        #if k not in model_state_dict:
        #    print(k)
        #else:
        paddle_state_dict[k] = v
    paddle.save(paddle_state_dict, paddle_path)
    #diff = list(set(model_state_dict).difference(set(torch_state_dict)))
    #print(diff)
    #diff = list(set(torch_state_dict).difference(set(model_state_dict)))
    #print(diff)

    #model.set_dict(paddle_state_dict)

if __name__ == "__main__":
    torch2paddle()
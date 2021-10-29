from reprod_log import ReprodDiffHelper



if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("torch/eval_torch.npy")
    paddle_info = diff_helper.load_info("paddle/eval_pd.npy")

    diff_helper.compare_info(torch_info, paddle_info)

    diff_helper.report(path="eval_diff.log")
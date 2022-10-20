import sys
import pickle
import paddle
import torch


def load_paddle_model(filename):
    state_dict = paddle.load(filename)
    for k in state_dict.keys():
        state_dict[k] = state_dict[k].numpy()
    return state_dict

def load_torch_model(filename):
    state_dict = torch.load(filename)
    for k in state_dict.keys():
        state_dict[k] = state_dict[k].cpu().numpy()
    return state_dict

def match(paddle_weight, torch_weight):
    paddle_state_dict = load_paddle_model(paddle_weight)
    torch_state_dict = load_torch_model(torch_weight)
    with open('paddle_state_dict.txt', 'w') as f:
        for k in paddle_state_dict.keys():
            f.write(k + ' ' + str(paddle_state_dict[k].shape) + '\n')
    print("output paddle_state_dict.txt done.")
    with open('torch_state_dict.txt', 'w') as f:
        for k in torch_state_dict.keys():
            f.write(k + ' ' + str(torch_state_dict[k].shape) + '\n')
    print("output torch_state_dict.txt done.")


if __name__ == "__main__":
    paddle_weight_path = sys.argv[1]
    torch_weight_path = sys.argv[2]
    match(paddle_weight_path, torch_weight_path)

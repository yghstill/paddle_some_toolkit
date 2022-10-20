import sys
import pickle
import paddle
import torch
import numpy as np

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

def convert(weights, torch_weight_name_file, paddle_weight_name_file, target_name):
    weight_name_map = {}
    with open(torch_weight_name_file) as tf:
        with open(paddle_weight_name_file) as pf:
            torch_line  =  tf.readlines()
            paddle_line  =  pf.readlines()
            for tk, pk in zip(torch_line, paddle_line):
                weight_name_map[pk.split()[0]] = tk.split()[0]

    # print(weight_name_map)
    dst = {}
    src = load_torch_model(weights)
    for k, v in weight_name_map.items():
        if k == 'linear.weight':
            src[v] = src[v].transpose(1, 0)
        dst[k] = src[v]

    pickle.dump(dst, open(target_name, 'wb'), protocol=2)
    print('Convert done. Saved in {target_name}.')


if __name__ == "__main__":
    weight_path = sys.argv[1]
    paddle_weight_name_file = sys.argv[2]
    torch_weight_name_file = sys.argv[3]
    # output name
    target_name = 'output.pdparams'
    convert(weight_path, torch_weight_name_file, paddle_weight_name_file, target_name)

import sys
import pickle
import paddle
import numpy as np

def load_paddle_model(filename):
    state_dict = paddle.load(filename)
    for k in state_dict.keys():
        state_dict[k] = state_dict[k].numpy()
    return state_dict


def convert(weights, ppcls_weight_name_file, ppdet_weight_name_file, target_name):
    weight_name_map = {}
    with open(ppcls_weight_name_file) as tf:
        with open(ppdet_weight_name_file) as pf:
            ppcls_line  =  tf.readlines()
            ppdet_line  =  pf.readlines()
            for tk, pk in zip(ppcls_line, ppdet_line):
                weight_name_map[pk.split()[0]] = tk.split()[0]

    print(weight_name_map)
    dst = {}
    src = load_paddle_model(weights)
    for k, v in weight_name_map.items():
        dst[k] = src[v]

    pickle.dump(dst, open(target_name, 'wb'), protocol=2)


if __name__ == "__main__":
    weight_path = sys.argv[1]
    ppcls_map_name_file = sys.argv[2]
    ppdet_map_name_file = sys.argv[3]
    # output name
    target_name = sys.argv[4]
    convert(weight_path, ppcls_map_name_file, ppdet_map_name_file, target_name)

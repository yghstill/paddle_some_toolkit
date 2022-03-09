import sys
import pickle
import paddle

def load_paddle_model(filename):
    state_dict = paddle.load(filename)
    for k in state_dict.keys():
        state_dict[k] = state_dict[k].numpy()
    return state_dict

def match(old_weight, paddle_weight):
    ppcls_state_dict = load_paddle_model(old_weight)
    ppdet_state_dict = load_paddle_model(paddle_weight)
    with open('ppcls_state_dict.txt', 'w') as f:
        for k in ppcls_state_dict.keys():
            f.write(k + ' ' + str(ppcls_state_dict[k].shape) + '\n')
    print("output ppcls_state_dict.txt done.")
    with open('ppdet_state_dict.txt', 'w') as f:
        for k in ppdet_state_dict.keys():
            f.write(k + ' ' + str(ppdet_state_dict[k].shape) + '\n')
    print("output ppdet_state_dict.txt done.")


if __name__ == "__main__":
    old_weight_path = sys.argv[1]
    paddle_weight_path = sys.argv[2]
    match(old_weight_path, paddle_weight_path)

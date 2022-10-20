import sys
import pickle
import paddle


def load_paddle_model(filename):
    state_dict = paddle.load(filename)
    for k in state_dict.keys():
        state_dict[k] = state_dict[k].numpy()
    return state_dict

def match(paddle_weight, new_paddle_weight):
    paddle_state_dict = load_paddle_model(paddle_weight)
    new_paddle_dict = load_paddle_model(new_paddle_weight)
    with open('paddle_state_dict.txt', 'w') as f:
        for k in paddle_state_dict.keys():
            f.write(k + ' ' + str(paddle_state_dict[k].shape) + '\n')
    print("output paddle_state_dict.txt done.")
    with open('new_paddle_dict.txt', 'w') as f:
        for k in new_paddle_dict.keys():
            f.write(k + ' ' + str(new_paddle_dict[k].shape) + '\n')
    print("output new_paddle_dict.txt done.")


if __name__ == "__main__":
    paddle_weight_path = sys.argv[1]
    new_paddle_weight_path = sys.argv[2]
    match(paddle_weight_path, new_paddle_weight_path)

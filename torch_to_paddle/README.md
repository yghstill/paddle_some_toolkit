# 将Pytorch权重转换至Paddle

## 1. 生成weight name映射list
```shell
python3.7 match_weight_name.py paddle.pdparams torch.pth.tar
```

此时会生成两个txt文件： paddle_state_dict.txt 、torch_state_dict.txt
然后对齐修改每一行weight都相同：
```shell
vimdiff paddle_state_dict.txt torch_state_dict.txt
```

## 2. 转换Torch权重至Paddle
```shell
python3.7 convert_paddle.py torch.pth.tar paddle_state_dict.txt torch_state_dict.txt
```
此时会生成转换好的Paddle权重： output.pdparams

## 3. Check转换的权重正确性
根据Paddle旧权重(paddle.pdparams) 和新权重(output.pdparams) 进行比对。
```shell
python3.7 check_weight.py paddle.pdparams output.pdparams
```

会生成：paddle_state_dict.txt、new_paddle_dict.txt
然后比对每一行weight是否完全一致：
```shell
vimdiff paddle_state_dict.txt new_paddle_dict.txt
```
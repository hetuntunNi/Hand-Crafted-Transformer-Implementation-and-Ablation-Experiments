手工搭建了Transformer并在tiny_shakespeare数据集做消融实验，数据集链接：https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
编程语言：Python 3.9
深度学习框架：PyTorch 2.5.1
依赖库：torch、matplotlib、datasets、numpy、tqdm、pickle
硬件环境：NVIDIA RTX 3090、CPU：Xeon Gold 6330
CUDA版本：11.8

创建conda环境
conda create -n Transformer python=3.9
conda activate Transformer

安装依赖
pip install -r requirements.txt

训练命令
对照组（完整模型）：
python train.py --use_positional_encoding

消融组1（无位置编码）：
python train.py

消融组2（无残差连接）：
python train.py --use_positional_encoding --no_residual

消融组3（单头注意力）：
python train1.py --use_positional_encoding

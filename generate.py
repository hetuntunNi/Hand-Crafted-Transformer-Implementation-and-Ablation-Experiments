import torch
import torch.nn as nn
import os
import pickle
from my_transformer import Transformer


class Config:
    def __init__(self, use_positional_encoding=True, use_residual=True):
        # 数据参数
        self.seq_length = 256
        self.batch_size = 128
        self.stride = 32

        # 模型参数
        self.d_model = 128
        self.max_len = 256
        self.n_heads = 4
        self.ffn_hidden = 512
        self.n_layers = 2
        self.drop_prob = 0.2

        # 消融实验参数
        self.use_positional_encoding = use_positional_encoding
        self.use_residual = use_residual

        # 训练参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 数据路径
        self.data_dir = "datasets"
        self.vocab_file = os.path.join(self.data_dir, "vocab.pkl")
        self.model_dir = os.path.join(self.data_dir, "models")
        self.results_dir = os.path.join(self.data_dir, "results")

        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)


def create_model(vocab_size, config):
    """创建模型实例"""
    model = Transformer(
        src_pad_idx=0,
        trg_pad_idx=0,
        enc_voc_size=vocab_size,
        dec_voc_size=vocab_size,
        d_model=config.d_model,
        max_len=config.max_len,
        n_heads=config.n_heads,
        ffn_hidden=config.ffn_hidden,
        n_layers=config.n_layers,
        drop_prob=config.drop_prob,
        device=config.device,
        use_positional_encoding=config.use_positional_encoding,
        use_residual=config.use_residual
    )
    return model


def load_vocab(config):
    """加载词汇表"""
    if os.path.exists(config.vocab_file):
        with open(config.vocab_file, 'rb') as f:
            vocab_data = pickle.load(f)
        return vocab_data['vocab'], vocab_data['idx_to_char'], vocab_data['vocab_size']
    else:
        raise FileNotFoundError(f"词汇表文件不存在: {config.vocab_file}")


def load_specific_model(exp_name, use_pos, use_res):
    """加载特定实验的模型"""
    # 创建配置
    config = Config(use_positional_encoding=use_pos, use_residual=use_res)

    # 加载词汇表
    vocab, idx_to_char, vocab_size = load_vocab(config)

    # 创建模型
    model = create_model(vocab_size, config)

    # 添加语言建模头
    if not hasattr(model, 'lm_head'):
        model.lm_head = nn.Linear(config.d_model, vocab_size).to(config.device)

    # 加载模型权重
    model_path = os.path.join(config.model_dir, f"{exp_name}_best.pth")

    if os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        state_dict = torch.load(model_path, map_location=config.device)

        # 过滤不匹配的键
        model_state_dict = model.state_dict()
        filtered_state_dict = {}

        for key, value in state_dict.items():
            if key in model_state_dict and model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                print(f"跳过键: {key}")

        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = model.to(config.device)
    model.eval()

    return model, vocab, idx_to_char, vocab_size, config


def nucleus_sampling(logits, top_p=0.9):
    """Top-p (nucleus) 采样"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # 移除累积概率超过top_p的token
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')

    return logits


def apply_repetition_penalty(logits, recent_tokens, penalty=1.2):
    """应用重复惩罚"""
    for token in set(recent_tokens):
        if token < len(logits):
            if logits[token] > 0:
                logits[token] /= penalty
            else:
                logits[token] *= penalty
    return logits


def get_optimized_parameters(exp_name, prompt):
    """根据实验名称和提示获取优化参数"""
    prompt_lower = prompt.lower()

    # 基础参数配置
    base_params = {
        'pos_1_res_1': {'temperature': 0.6, 'top_k': 25, 'top_p': 0.88, 'repetition_penalty': 1.3},
        'pos_0_res_1': {'temperature': 0.7, 'top_k': 40, 'top_p': 0.9, 'repetition_penalty': 1.2},
        'pos_1_res_0': {'temperature': 0.65, 'top_k': 30, 'top_p': 0.87, 'repetition_penalty': 1.25},
        'pos_0_res_0': {'temperature': 0.63, 'top_k': 28, 'top_p': 0.86, 'repetition_penalty': 1.3}
    }

    params = base_params.get(exp_name, {'temperature': 0.7, 'top_k': 40, 'top_p': 0.9, 'repetition_penalty': 1.2})

    # 根据提示类型调整参数
    if any(keyword in prompt_lower for keyword in ['king', 'queen', 'lord', 'lady']):
        params['max_length'] = 180
    elif any(keyword in prompt_lower for keyword in ['citizen', 'people', 'all']):
        params['max_length'] = 150
    elif '?' in prompt:
        params['max_length'] = 120
    elif any(keyword in prompt_lower for keyword in ['to be', 'romeo', 'juliet']):
        params['max_length'] = 200
    else:
        params['max_length'] = 160

    return params


def generate_text(model, vocab, idx_to_char, prompt, config, exp_name, params):
    """生成文本"""
    model.eval()
    device = config.device

    # 将提示转换为索引
    indices = []
    for char in prompt:
        indices.append(vocab.get(char, vocab.get('<unk>', 3)))

    # 记录最近生成的token
    recent_tokens = []
    max_recent_length = 15

    with torch.no_grad():
        for _ in range(params['max_length']):
            # 准备输入序列
            if len(indices) >= config.seq_length:
                input_seq = torch.tensor(indices[-config.seq_length:], dtype=torch.long)
            else:
                input_seq = torch.zeros(config.seq_length, dtype=torch.long)
                input_seq[-len(indices):] = torch.tensor(indices, dtype=torch.long)

            input_seq = input_seq.unsqueeze(0).to(device)

            try:
                # 前向传播
                encoder_output = model.encoder(input_seq, None)
                output = model.lm_head(encoder_output)

                # 获取最后一个token的logits
                logits = output[0, -1, :] / params['temperature']

                # 应用重复惩罚
                logits = apply_repetition_penalty(logits, recent_tokens, params['repetition_penalty'])

                # Top-k 采样
                if params['top_k'] > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(params['top_k'], logits.size(-1)))
                    logits = torch.full_like(logits, float('-inf'))
                    logits[top_k_indices] = top_k_logits

                # Top-p 采样
                logits = nucleus_sampling(logits, params['top_p'])

                # 从分布中采样
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                # 更新索引
                indices.append(next_token)

                # 更新最近token列表
                recent_tokens.append(next_token)
                if len(recent_tokens) > max_recent_length:
                    recent_tokens.pop(0)

                # 停止条件
                if (next_token == vocab.get('<eos>', -1) or
                        len(indices) >= params['max_length'] or
                        (idx_to_char.get(next_token, '') in '.!?' and len(indices) > len(prompt) + 20)):
                    break

            except Exception as e:
                print(f"生成过程中出错: {e}")
                break

    # 将索引转换回文本
    generated_text = ''.join([idx_to_char.get(idx, '?') for idx in indices])
    return generated_text


def main():
    """主函数"""
    # 实验配置
    experiments = [
        {
            'name': 'pos_1_res_1',
            'use_pos': True,
            'use_res': True,
            'description': '完整模型 (位置编码 + 残差连接)'
        },
        {
            'name': 'pos_0_res_1',
            'use_pos': False,
            'use_res': True,
            'description': '无位置编码 + 残差连接'
        },
        {
            'name': 'pos_1_res_0',
            'use_pos': True,
            'use_res': False,
            'description': '位置编码 + 无残差连接'
        },
        {
            'name': 'pos_0_res_0',
            'use_pos': False,
            'use_res': False,
            'description': '无位置编码 + 无残差连接'
        }
    ]

    # 提示文本
    prompts = [
        "First Citizen:",
        "To be or not to be",
        "KING:",
        "ROMEO:",
        "What light through yonder window breaks?"
    ]

    num_samples = 3

    all_results = {}

    for exp_config in experiments:
        exp_name = exp_config['name']
        print(f"\n处理实验: {exp_name} - {exp_config['description']}")

        try:
            # 加载模型
            model, vocab, idx_to_char, vocab_size, config = load_specific_model(
                exp_name, exp_config['use_pos'], exp_config['use_res']
            )
            print(f"模型加载成功，词汇表大小: {vocab_size}")

            results = {}
            for prompt in prompts:
                print(f"\n生成提示: '{prompt}'")

                samples = []
                for i in range(num_samples):
                    # 获取优化参数
                    params = get_optimized_parameters(exp_name, prompt)

                    # 生成文本
                    generated = generate_text(model, vocab, idx_to_char, prompt, config, exp_name, params)
                    samples.append(generated)

                    # 显示结果
                    preview = generated[:80] + "..." if len(generated) > 80 else generated
                    print(f"样本 {i + 1}: {preview}")

                results[prompt] = samples

            # 保存结果
            output_dir = os.path.join(config.data_dir, "generate")
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, f"{exp_name}_generations.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"实验: {exp_name} - {exp_config['description']}\n")
                f.write("=" * 60 + "\n\n")

                for prompt, samples in results.items():
                    f.write(f"提示: '{prompt}'\n")
                    f.write("-" * 40 + "\n")

                    for i, sample in enumerate(samples, 1):
                        f.write(f"样本 {i}:\n")
                        f.write(f"{sample}\n\n")

                    f.write("=" * 60 + "\n\n")

            print(f"结果已保存: {output_file}")
            all_results[exp_name] = results

        except FileNotFoundError as e:
            print(f"错误: {e}")
            continue
        except Exception as e:
            print(f"处理实验时出错: {e}")
            continue

    # 生成比较报告
    if all_results:
        output_dir = os.path.join("datasets", "generate")
        os.makedirs(output_dir, exist_ok=True)

        report_file = os.path.join(output_dir, "comparison_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("四个模型文本生成比较报告\n")
            f.write("=" * 80 + "\n\n")

            for prompt in prompts:
                f.write(f"\n提示: '{prompt}'\n")
                f.write("=" * 60 + "\n")

                for exp_config in experiments:
                    exp_name = exp_config['name']
                    if exp_name in all_results and prompt in all_results[exp_name]:
                        f.write(f"\n{exp_config['description']}:\n")
                        f.write("-" * 40 + "\n")

                        for i, sample in enumerate(all_results[exp_name][prompt][:2], 1):
                            f.write(f"样本 {i}: {sample[:100]}...\n")
                        f.write("\n")

        print(f"\n比较报告已保存: {report_file}")

    print("\n所有实验完成!")
    print("生成结果保存在: datasets/generate/ 目录下")


if __name__ == "__main__":
    main()
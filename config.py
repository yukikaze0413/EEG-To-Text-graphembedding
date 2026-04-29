import argparse

# 将字符串解析为布尔值，便于在命令行中传入真假开关
def str2bool(v):
    # 如果本身就是布尔类型，直接返回
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config(case):
    # 统一管理训练/评估脚本的命令行参数
    if case == 'train_decoding': 
        # EEG-To-Text 解码训练参数
        parser = argparse.ArgumentParser(description='指定 EEG-To-Text 解码训练参数')
        
        # 基础任务与模型配置
        parser.add_argument('-m', '--model_name', help='从 {BrainTranslator, BrainTranslatorNaive} 中选择', default = "BrainTranslator" ,required=True)
        parser.add_argument('-t', '--task_name', help='从 {task1,task1_task2, task1_task2_task3,task1_task2_taskNRv2} 中选择', default = "task1", required=True)
        
        # 两阶段训练开关：one_step 表示跳过第一阶段
        parser.add_argument('-1step', '--one_step', dest='skip_step_one', action='store_true')
        parser.add_argument('-2step', '--two_step', dest='skip_step_one', action='store_false')

        # 预训练权重 / 随机初始化
        parser.add_argument('-pre', '--pretrained', dest='use_random_init', action='store_false')
        parser.add_argument('-rand', '--rand_init', dest='use_random_init', action='store_true')
        
        parser.add_argument('-load1', '--load_step1_checkpoint', dest='load_step1_checkpoint', action='store_true')
        parser.add_argument('-no-load1', '--not_load_step1_checkpoint', dest='load_step1_checkpoint', action='store_false')

        # 训练轮次、学习率、批大小
        parser.add_argument('-ne1', '--num_epoch_step1', type = int, help='num_epoch_step1', default = 20, required=True)
        parser.add_argument('-ne2', '--num_epoch_step2', type = int, help='num_epoch_step2', default = 30, required=True)
        parser.add_argument('-lr1', '--learning_rate_step1', type = float, help='learning_rate_step1', default = 0.00005, required=True)
        parser.add_argument('-lr2', '--learning_rate_step2', type = float, help='learning_rate_step2', default = 0.0000005, required=True)
        parser.add_argument('-b', '--batch_size', type = int, help='batch_size', default = 32, required=True)
        
        # 数据与设备配置
        parser.add_argument('-s', '--save_path', help='checkpoint 保存路径', default = './checkpoints/decoding', required=True)
        parser.add_argument('-subj', '--subjects', help='使用全部被试或指定单个被试', default = 'ALL', required=False)
        parser.add_argument('-eeg', '--eeg_type', help='从 {GD, FFD, TRT} 中选择', default = 'GD', required=False)
        parser.add_argument('-band', '--eeg_bands', nargs='+', help='指定 EEG 频带', default = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] , required=False)
        parser.add_argument('-cuda', '--cuda', help='指定 cuda 设备，如 cuda:0, cuda:1', default = 'cuda:0')
        
        parser.add_argument('-train_input', '--train_input', help='训练输入类型（如 EEG/noise）' ,required=True)

        # EEG 编码器配置：支持 Transformer, 双向 Mamba 与 GAT 图网络
        parser.add_argument('--eeg_encoder_type', type=str, default='transformer', choices=['transformer', 'bimamba', 'gat'])
        parser.add_argument('--mamba_num_layers', type=int, default=6)
        parser.add_argument('--mamba_d_state', type=int, default=16)
        parser.add_argument('--mamba_d_conv', type=int, default=4)
        parser.add_argument('--mamba_expand', type=int, default=2)
        parser.add_argument('--mamba_dropout', type=float, default=0.1)
        parser.add_argument('--bimamba_fusion', type=str, default='concat_linear', choices=['concat_linear', 'gated_sum'])
        
        # GAT 图网络特定配置
        parser.add_argument('--gat_num_layers', type=int, default=2)
        parser.add_argument('--gat_dropout', type=float, default=0.1)

        # 对比对齐配置：用于增强 EEG 与文本语义一致性
        parser.add_argument('--use_contrastive_align', type=str2bool, default=False)
        parser.add_argument('--contrastive_weight', type=float, default=0.05)
        parser.add_argument('--contrastive_warmup_epochs', type=int, default=0)
        parser.add_argument('--contrastive_proj_dim', type=int, default=768)
        parser.add_argument('--contrastive_temperature', type=float, default=0.07)
        parser.add_argument('--text_embed_model', type=str, default='sentence-transformers/all-mpnet-base-v2')

        args = vars(parser.parse_args())

    elif case == 'eval_decoding':
        # EEG-To-Text 解码评估参数
        parser = argparse.ArgumentParser(description='指定 EEG-To-Text 解码评估参数')
        parser.add_argument('-checkpoint', '--checkpoint_path', help='模型 checkpoint 路径' ,required=True)
        parser.add_argument('-conf', '--config_path', help='训练配置 JSON 路径' ,required=True)
        parser.add_argument('-test_input', '--test_input', help='测试输入类型（如 EEG/noise）' ,required=True)
        parser.add_argument('-train_input', '--train_input', help='训练输入类型（如 EEG/noise）' ,required=True)
        parser.add_argument('-cuda', '--cuda', help='指定 cuda 设备，如 cuda:0, cuda:1', default = 'cuda:0')
        args = vars(parser.parse_args())

    # 返回字典格式参数，便于训练/评估脚本统一读取
    return args
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm
from transformers import BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BartConfig, BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, PegasusForConditionalGeneration, PegasusTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderConfig, EncoderDecoderModel
from data import ZuCo_dataset
from model_decoding import BrainTranslator, BrainTranslatorNaive, T5Translator
from config import get_config

def train_model(
    dataloaders,
    device,
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs=25,
    checkpoint_path_best='./checkpoints/decoding/best/temp_decoding.pt',
    checkpoint_path_last='./checkpoints/decoding/last/temp_decoding.pt',
    use_contrastive_align=False,
    contrastive_weight=0.0,
    contrastive_warmup_epochs=0,
):
    # 训练主循环（参考 PyTorch 迁移学习教程）
    since = time.time()
      
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每个 epoch 包含训练和验证两个阶段
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 评估模式

            # 累计总损失与对比损失（用于 epoch 级日志）
            running_loss = 0.0
            running_contrastive_loss = 0.0

            # 遍历当前阶段的数据
            first_batch_logged = False
            for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask in tqdm(dataloaders[phase]):
                
                # 将一个 batch 数据搬运到目标设备
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
                # 将目标 token 解码成文本，用于对比学习文本侧输入
                target_text_batch = tokenizer.batch_decode(target_ids, skip_special_tokens=True)
                """将 target_ids 中的 padding id 替换为 -100"""
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                # 梯度清零
                optimizer.zero_grad()

                # 前向计算；仅训练阶段保留梯度图
                # 仅在训练阶段跟踪梯度历史
                with torch.set_grad_enabled(phase == 'train'):
                    # warmup 之前不启用对比损失，避免初期训练不稳定
                    current_contrastive_weight = 0.0
                    if use_contrastive_align and (epoch >= contrastive_warmup_epochs):
                        current_contrastive_weight = contrastive_weight

                    # 启用对比学习时，向模型传入文本与权重
                    if use_contrastive_align:
                        seq2seqLMoutput = model(
                            input_embeddings_batch,
                            input_masks_batch,
                            input_mask_invert_batch,
                            target_ids_batch,
                            text_list=target_text_batch,
                            contrastive_weight=current_contrastive_weight,
                        )
                    else:
                        seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)

                    # 诊断日志：仅在每个 epoch 的首个 batch 打印一次
                    if not first_batch_logged:
                        # [问题源1] 诊断：冻结文本编码器是否被意外切回 train 模式
                        text_encoder_training = None
                        if use_contrastive_align:
                            root_model = model.module if hasattr(model, 'module') else model
                            if hasattr(root_model, 'contrast_head') and hasattr(root_model.contrast_head, 'text_encoder'):
                                text_encoder_training = root_model.contrast_head.text_encoder.training

                        # [问题源2] 诊断：contrastive_loss 是否稳定挂载到输出对象
                        contrastive_attr_exists = hasattr(seq2seqLMoutput, 'contrastive_loss')
                        contrastive_is_none = (getattr(seq2seqLMoutput, 'contrastive_loss', None) is None)

                        print(
                            f"[DIAG][{phase}] epoch={epoch} "
                            f"text_encoder.training={text_encoder_training} "
                            f"contrastive_attr_exists={contrastive_attr_exists} "
                            f"contrastive_is_none={contrastive_is_none}"
                        )
                        first_batch_logged = True

                    """计算损失"""
                    # logits = seq2seqLMoutput.logits # 8*48*50265
                    # logits = logits.permute(0,2,1) # 8*50265*48

                    # loss = criterion(logits, target_ids_batch_label) # 仅在有效目标位置上计算交叉熵
                    # 注：当前未使用自定义 criterion
                    loss = seq2seqLMoutput.loss # 使用模型自带语言建模损失

                    # """调试：检查每个 batch 第 0 个样本预测"""
                    # print('target size:', target_ids_batch.size(), ',original logits size:', logits.size(), ',target_mask size', target_mask_batch.size())
                    # logits = logits.permute(0,2,1)
                    # for idx in [0]:
                    #     print(f'-- instance {idx} --')
                    #     # print('permuted logits size:', logits.size())
                    #     probs = logits[idx].softmax(dim = 1)
                    #     # print('probs size:', probs.size())
                    #     values, predictions = probs.topk(1)
                    #     # print('predictions before squeeze:',predictions.size())
                    #     predictions = torch.squeeze(predictions)
                    #     # print('predictions:',predictions)
                    #     # print('target mask:', target_mask_batch[idx])
                    #     # print('[DEBUG]target tokens:',tokenizer.decode(target_ids_batch_copy[idx]))
                    #     print('[DEBUG]predicted tokens:',tokenizer.decode(predictions))
                
                    # 仅训练阶段反向传播并更新参数
                    if phase == 'train':
                        # with torch.autograd.detect_anomaly():
                        loss.sum().backward()
                        optimizer.step()

                # 统计当前 batch 损失
                running_loss += loss.sum().item() * input_embeddings_batch.size()[0] # 当前 batch 的损失
                contrastive_loss = getattr(seq2seqLMoutput, 'contrastive_loss', None)
                if contrastive_loss is not None:
                    running_contrastive_loss += contrastive_loss.detach().item() * input_embeddings_batch.size()[0]
                # print('[DEBUG]loss:',loss.item())
                # print('#################################')
                

            # 仅训练阶段更新学习率调度器
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            # 打印 epoch 级损失日志
            if use_contrastive_align:
                epoch_contrastive_loss = running_contrastive_loss / dataset_sizes[phase]
                print('{} Loss: {:.4f}, Contrastive: {:.4f}'.format(phase, epoch_loss, epoch_contrastive_loss))
            else:
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # 在验证集上更新最优模型
            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                '''保存最优 checkpoint'''
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'在 dev 集更新最优 checkpoint: {checkpoint_path_best}')
                # with torch.set_grad_enabled(False):
                #     traced_model_1 = torch.jit.trace(model, (torch.rand(1, 56, 840).to(device), torch.randint(1, 56).to(device), torch.rand(1, 56).to(device), torch.rand(1, 56).to(device)))
                #     traced_model_32 = torch.jit.trace(model, (torch.rand(32, 56, 840).to(device), torch.randint(32, 56).to(device), torch.rand(32, 56).to(device), torch.rand(32, 56).to(device)))
                # torch.jit.save(traced_model_1, checkpoint_path_best[:-3]+'_1_jit.pt')
                # torch.jit.save(traced_model_32, checkpoint_path_best[:-3]+'_32_jit.pt')
        print()

    time_elapsed = time.time() - since
    print('训练完成，用时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('最佳验证损失: {:4f}'.format(best_loss))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'已保存最后一个 checkpoint: {checkpoint_path_last}')

    # 回载验证集最优模型参数
    model.load_state_dict(best_model_wts)
    return model

def show_require_grad_layers(model):
    # 打印当前参与训练的参数层，便于检查冻结策略
    print()
    print(' 可训练层如下:')
    # 一致性检查
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)

if __name__ == '__main__':
    args = get_config('train_decoding')

    ''' 读取训练配置参数 '''
    dataset_setting = 'unique_sent'
    
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    
    batch_size = args['batch_size']
    
    model_name = args['model_name']
    # model_name = 'BrainTranslatorNaive' # 不使用额外 Transformer 编码器
    # model_name = 'BrainTranslator'
    
    # task_name = 'task1'
    # task_name = 'task1_task2'
    # task_name = 'task1_task2_task3'
    # task_name = 'task1_task2_taskNRv2'
    task_name = args['task_name']
    train_input = args['train_input']
    print("train_input 为:", train_input)
    save_path = args['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    skip_step_one = args['skip_step_one']
    load_step1_checkpoint = args['load_step1_checkpoint']
    use_random_init = args['use_random_init']
    device_ids = [0] # 设备编号配置

    # 双向 Mamba 编码器相关参数
    eeg_encoder_type = args['eeg_encoder_type']
    mamba_num_layers = args['mamba_num_layers']
    mamba_d_state = args['mamba_d_state']
    mamba_d_conv = args['mamba_d_conv']
    mamba_expand = args['mamba_expand']
    mamba_dropout = args['mamba_dropout']
    bimamba_fusion = args['bimamba_fusion']

    gat_num_layers = args['gat_num_layers']
    gat_dropout = args['gat_dropout']

    # 对比对齐相关参数
    use_contrastive_align = args['use_contrastive_align']
    contrastive_weight = args['contrastive_weight']
    contrastive_warmup_epochs = args['contrastive_warmup_epochs']
    contrastive_proj_dim = args['contrastive_proj_dim']
    contrastive_temperature = args['contrastive_temperature']
    text_embed_model = args['text_embed_model']

    if use_random_init and skip_step_one:
        step2_lr = 5*1e-4
        
    print(f'[INFO]使用模型: {model_name}')
    
    if skip_step_one:
        save_name = f'{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}_{train_input}'
    else:
        save_name = f'{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}_{train_input}'
    
    if use_random_init:
        save_name = 'randinit_' + save_name

    save_path_best = os.path.join(save_path, 'best')
    if not os.path.exists(save_path_best):
        os.makedirs(save_path_best)

    output_checkpoint_name_best = os.path.join(save_path_best, f'{save_name}.pt')

    save_path_last = os.path.join(save_path, 'last')
    if not os.path.exists(save_path_last):
        os.makedirs(save_path_last)

    output_checkpoint_name_last = os.path.join(save_path_last, f'{save_name}.pt')

    # subject_choice = 'ALL
    subject_choice = args['subjects']
    print(f'![Debug]使用被试: {subject_choice}')
    # eeg_type_choice = 'GD
    eeg_type_choice = args['eeg_type']
    print(f'[INFO]EEG 类型: {eeg_type_choice}')
    # bands_choice = ['_t1'] 
    # bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    bands_choice = args['eeg_bands']
    print(f'[INFO]使用频带: {bands_choice}')


    
    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


    ''' 配置设备（GPU/CPU） '''
    # 优先使用 CUDA
    if torch.cuda.is_available():  
        # dev = "cuda:3" 
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3
    device = torch.device(dev)
    print(f'[INFO]使用设备: {dev}')
    print()

    ''' 构建数据集与 DataLoader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = '/data/johj/ZuCo_data/task1-SR/task1_source.pkl'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = '/data/johj/ZuCo_data/task2-NR/task2_source.pkl' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = '/data/johj/ZuCo_data/task3-TSR/task3_source.pkl' 
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = '/data/johj/ZuCo_data/task2-NR-2.0/taskNRv2_source.pkl' 
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    print()

    """保存训练配置到 JSON"""
    cfg_dir = './config/decoding/'

    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)

    with open(os.path.join(cfg_dir,f'{save_name}.json'), 'w') as out_config:
        json.dump(args, out_config, indent = 4)

    if model_name in ['BrainTranslator','BrainTranslatorNaive']:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    elif model_name == 'PegasusTranslator':
        tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
    
    elif model_name == 'T5Translator':
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
        #tokenizer.set_prefix_tokens(language='english')

    # 训练集
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    # 验证集
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    # 测试集（此处未启用）
    # test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)

    
    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO]训练集大小: ', len(train_set))
    print('[INFO]验证集大小: ', len(dev_set))
    # print('[INFO]test_set size: ', len(test_set))
    
    # 训练 dataloader
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=4)
    # 验证 dataloader
    val_dataloader = DataLoader(dev_set, batch_size = 1, shuffle=False, num_workers=4)
    # dataloaders 字典
    dataloaders = {'train':train_dataloader, 'dev':val_dataloader}

    ''' 构建模型 '''
    if model_name == 'BrainTranslator':
        if use_random_init:
            config = BartConfig.from_pretrained('facebook/bart-large')
            pretrained = BartForConditionalGeneration(config)
        else:
            pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    
        model = BrainTranslator(
            pretrained,
            in_feature=105*len(bands_choice),
            decoder_embedding_size=1024,
            additional_encoder_nhead=8,
            additional_encoder_dim_feedforward=2048,
            eeg_encoder_type=eeg_encoder_type,
            mamba_num_layers=mamba_num_layers,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_dropout=mamba_dropout,
            bimamba_fusion=bimamba_fusion,
            gat_num_layers=gat_num_layers,
            gat_dropout=gat_dropout,
            use_contrastive_align=use_contrastive_align,
            contrastive_proj_dim=contrastive_proj_dim,
            contrastive_temperature=contrastive_temperature,
            text_embed_model=text_embed_model,
        )
    
    elif model_name == 'BrainTranslatorNaive':
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        model = BrainTranslatorNaive(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

    elif model_name == 'PegasusTranslator':
        pretrained = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
        model = BrainTranslator(
            pretrained,
            in_feature=105*len(bands_choice),
            decoder_embedding_size=1024,
            additional_encoder_nhead=8,
            additional_encoder_dim_feedforward=2048,
            eeg_encoder_type=eeg_encoder_type,
            mamba_num_layers=mamba_num_layers,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_dropout=mamba_dropout,
            bimamba_fusion=bimamba_fusion,
            gat_num_layers=gat_num_layers,
            gat_dropout=gat_dropout,
            use_contrastive_align=use_contrastive_align,
            contrastive_proj_dim=contrastive_proj_dim,
            contrastive_temperature=contrastive_temperature,
            text_embed_model=text_embed_model,
        )
    
    elif model_name == 'T5Translator':
        pretrained = T5ForConditionalGeneration.from_pretrained("t5-large")
        model = T5Translator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    ''' 两阶段训练循环 '''

    ######################################################
    '''第一阶段训练：冻结大部分预训练参数'''
    ######################################################

    # 基本沿用 BART 论文的参数冻结策略
    if model_name in ['BrainTranslator','BrainTranslatorNaive', 'PegasusTranslator', 'T5Translator']:
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                    continue
                else:
                    param.requires_grad = False

    elif model_name == 'BertGeneration':
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if ('embeddings' in name) or ('encoder.layer.0' in name):
                    continue
                else:
                    param.requires_grad = False
 

    if skip_step_one:
        if load_step1_checkpoint:
            stepone_checkpoint = 'path_to_step_1_checkpoint.pt'
            print(f'skip step one, load checkpoint: {stepone_checkpoint}')
            model.load_state_dict(torch.load(stepone_checkpoint))
        else:
            print('跳过 step1，从 step2 开始训练')
    else:

        ''' 配置第一阶段优化器与学习率调度器 '''
        optimizer_step1 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step1_lr, momentum=0.9)

        exp_lr_scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=20, gamma=0.1)

        ''' 配置损失函数 '''
        criterion = nn.CrossEntropyLoss()

        print('=== 开始 Step1 训练 ... ===')
        # 打印参与训练的层
        show_require_grad_layers(model)
        # 返回第一阶段在验证集上最优的模型
        model = train_model(
            dataloaders,
            device,
            model,
            criterion,
            optimizer_step1,
            exp_lr_scheduler_step1,
            num_epochs=num_epochs_step1,
            checkpoint_path_best=output_checkpoint_name_best,
            checkpoint_path_last=output_checkpoint_name_last,
            use_contrastive_align=False,
            contrastive_weight=0.0,
            contrastive_warmup_epochs=0,
        )

    ######################################################
    '''第二阶段训练：解冻全模型后继续优化若干轮'''
    ######################################################
    for name, param in model.named_parameters():
        param.requires_grad = True

    ''' 配置第二阶段优化器与学习率调度器 '''
    optimizer_step2 = optim.SGD(model.parameters(), lr=step2_lr, momentum=0.9)

    exp_lr_scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=30, gamma=0.1)

    ''' 配置损失函数 '''
    criterion = nn.CrossEntropyLoss()
    
    print()
    print('=== 开始 Step2 训练 ... ===')
    # 打印训练层
    show_require_grad_layers(model)
    
    '''主训练循环'''
    trained_model = train_model(
        dataloaders,
        device,
        model,
        criterion,
        optimizer_step2,
        exp_lr_scheduler_step2,
        num_epochs=num_epochs_step2,
        checkpoint_path_best=output_checkpoint_name_best,
        checkpoint_path_last=output_checkpoint_name_last,
        use_contrastive_align=use_contrastive_align,
        contrastive_weight=contrastive_weight,
        contrastive_warmup_epochs=contrastive_warmup_epochs,
    )

    # '''保存 checkpoint'''
    # torch.save(trained_model.state_dict(), os.path.join(save_path,output_checkpoint_name))

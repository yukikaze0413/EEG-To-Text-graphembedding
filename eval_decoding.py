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
import torch.nn.functional as F
import time
from transformers import BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BartConfig, BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, PegasusForConditionalGeneration, PegasusTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertGenerationDecoder
from data import ZuCo_dataset
from model_decoding import BrainTranslator, BrainTranslatorNaive, T5Translator
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import Rouge
from config import get_config
import evaluate
from evaluate import load

metric = evaluate.load("sacrebleu")
cer_metric = load("cer")
wer_metric = load("wer")

def remove_text_after_token(text, token='</s>'):
    # 截断指定 token 之后的文本
    token_index = text.find(token)
    if token_index != -1:  # 找到 token
        return text[:token_index]  # 返回 token 之前的文本
    return text  # 未找到 token，返回原文本

def eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path = './results/temp.txt' , score_results='./score_results/task.txt'):
    # 参考 PyTorch 教程改写
    start_time = time.time()
    model.eval()   # 切换到评估模式
    
    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    pred_tokens_list_previous = []
    pred_string_list_previous = []


    with open(output_all_results_path,'w') as f:
        for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask in tqdm(dataloaders['test']):
            # 加载并搬运当前 batch
            input_embeddings_batch = input_embeddings.to(device).float() # B, 56, 840
            input_masks_batch = input_masks.to(device) # B, 56
            target_ids_batch = target_ids.to(device) # B, 56
            input_mask_invert_batch = input_mask_invert.to(device) # B, 56
            
            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens = True)
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens = True)
            # print('target ids tensor:',target_ids_batch[0])
            # print('target ids:',target_ids_batch[0].tolist())
            # print('target tokens:',target_tokens)
            # print('target string:',target_strininvert.to(device) # B, 56
            
            f.write(f'target string: {target_string}\n')

            # 保存目标文本，供后续计算 BLEU 等指标
            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)
            
            """将 target 中的 padding id 替换为 -100（忽略损失）"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 

            # target_ids_batch_label = target_ids_batch.clone().detach()
            # target_ids_batch_label[target_ids_batch_label == tokenizer.pad_token_id] = -100

            # 教师强制（teacher forcing）路径
            seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch) # (batch, time, n_class)
            logits_previous = seq2seqLMoutput.logits
            probs_previous = logits_previous[0].softmax(dim = 1)
            values_previous, predictions_previous = probs_previous.topk(1)
            predictions_previous = torch.squeeze(predictions_previous)
            predicted_string_previous = remove_text_after_token(tokenizer.decode(predictions_previous).split('</s></s>')[0].replace('<s>',''))
            f.write(f'predicted string with tf: {predicted_string_previous}\n')
            predictions_previous = predictions_previous.tolist()
            truncated_prediction_previous = []
            for t in predictions_previous:
                if t != tokenizer.eos_token_id:
                    truncated_prediction_previous.append(t)
                else:
                    break
            pred_tokens_previous = tokenizer.convert_ids_to_tokens(truncated_prediction_previous, skip_special_tokens = True)
            pred_tokens_list_previous.append(pred_tokens_previous)
            pred_string_list_previous.append(predicted_string_previous)
            

            # 自回归生成路径
            predictions=model.generate(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch,
                                       max_length=56,
                                       num_beams=5,
                                       do_sample=True,
                                       repetition_penalty= 5.0,
                                       no_repeat_ngram_size = 2
                                       # 备选参数示例：
                                       # num_beams=5, encoder_no_repeat_ngram_size=1
                                       # do_sample=True, top_k=15, temperature=0.5, num_return_sequences=5
                                       # early_stopping=True
                                       )
            
            predicted_string=tokenizer.batch_decode(predictions, skip_special_tokens=True)[0]
            # predicted_string=predicted_string.squeeze()
            
            predictions=tokenizer.encode(predicted_string)
            # print('predicted string:',predicted_string)
            f.write(f'predicted string: {predicted_string}\n')
            f.write(f'################################################\n\n\n')

            # 转为 int 列表（当前 predictions 已经是 list）
            # predictions = predictions.tolist()
            truncated_prediction = []
            for t in predictions:
                if t != tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens = True)
            # print('predicted tokens:',pred_tokens)
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)
            # pred_tokens_list.extend(pred_tokens)
            # pred_string_list.extend(predicted_string)
            # print('################################################')
            # print()
    # print(f"pred_string_list : {pred_string_list}")
    
    """计算 corpus BLEU 分数"""
    weights_list = [(1.0,),(0.5,0.5),(1./3.,1./3.,1./3.),(0.25,0.25,0.25,0.25)]
    corpus_bleu_scores = []
    corpus_bleu_scores_previous = []
    for weight in weights_list:
        # print('weight:',weight)
        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights = weight)
        corpus_bleu_score_previous = corpus_bleu(target_tokens_list, pred_tokens_list_previous, weights = weight)
        corpus_bleu_scores.append(corpus_bleu_score)
        corpus_bleu_scores_previous.append(corpus_bleu_score_previous)
        print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)
        print(f'corpus BLEU-{len(list(weight))} score with tf:', corpus_bleu_score_previous)


    """计算 sacreBLEU 分数"""
    
    reference_list = [[item] for item in target_string_list]

    # print(f'ref: {reference_list}')
    # print(f'pred: {prediction_list}')
    sacre_blue = metric.compute(predictions=pred_string_list, references=reference_list)
    sacre_blue_previous = metric.compute(predictions=pred_string_list_previous, references=reference_list)
    print("sacreblue score: ", sacre_blue, '\n')
    print("sacreblue score with tf: ", sacre_blue_previous)


    print()
    """计算 ROUGE 分数"""
    rouge = Rouge()
    
    # pred_string_list = [item for sublist in pred_string_list for item in sublist]
    # pred_string_list = [item for sublist in pred_string_list for item in sublist]
    # pred_string_list_previous = [item for sublist in pred_string_list_previous for item in sublist]
    # rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg = True, ignore_empty=True)
    # rouge_scores_previous = rouge.get_scores(pred_string_list_previous, target_string_list, avg = True, ignore_empty=True)
    # print('rouge_scores: ', rouge_scores)
    # print('rouge_scores with tf:', rouge_scores_previous)

    # rouge_scores_previous = rouge.get_scores(pred_string_list_previous, target_string_list, avg = True, ignore_empty=True)
    # print('rouge_scores', rouge_scores)
    # print('previous rouge_scores', rouge_scores_previous)

    try:
        rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg = True, ignore_empty=True)
    except ValueError as e:
        rouge_scores = '预测文本为空'

    try:
        rouge_scores_previous = rouge.get_scores(pred_string_list_previous, target_string_list, avg = True, ignore_empty=True)
    except ValueError as e:
        rouge_scores_previous = '预测文本为空'
    print()


    print()
    """计算 WER 分数"""
    # wer = WordErrorRate()
    wer_scores = wer_metric.compute(predictions=pred_string_list, references=target_string_list)
    wer_scores_previous = wer_metric.compute(predictions=pred_string_list_previous, references=target_string_list)
    print("WER score:", wer_scores)
    print("WER score with tf:", wer_scores_previous)
    

    """计算 CER 分数"""
    cer_scores = cer_metric.compute(predictions=pred_string_list, references=target_string_list)
    cer_scores_previous = cer_metric.compute(predictions=pred_string_list_previous, references=target_string_list)
    print("CER score:", cer_scores)
    print("CER score with tf:", cer_scores_previous)


    end_time = time.time()
    print(f"评估耗时 {(end_time-start_time)/60} 分钟。")

     # 将评估结果写入文件（含 teacher-forcing 对照）
    file_content = [
    f'corpus_bleu_score = {corpus_bleu_scores}',
    f'sacre_blue_score = {sacre_blue}',
    f'rouge_scores = {rouge_scores}',
    f'wer_scores = {wer_scores}',
    f'cer_scores = {cer_scores}',
    f'corpus_bleu_score_with_tf = {corpus_bleu_scores_previous}',
    f'sacre_blue_score_with_tf = {sacre_blue_previous}',
    f'rouge_scores_with_tf = {rouge_scores_previous}',
    f'wer_scores_with_tf = {wer_scores_previous}',
    f'cer_scores_with_tf = {cer_scores_previous}',
    ]
    
    with open(score_results, "a") as file_results:
        for line in file_content:
            if isinstance(line, list):
                for item in line:
                    file_results.write(str(item) + "\n")
            else:
                file_results.write(str(line) + "\n")



if __name__ == '__main__':
    # 评估阶段固定 batch_size=1，逐句生成便于统计
    batch_size = 1
    ''' 读取评估参数 '''
    args = get_config('eval_decoding')
    test_input = args['test_input']
    print("test_input is:", test_input)
    train_input = args['train_input']
    print("train_input is:", train_input)
    ''' 读取训练阶段保存的配置文件 '''
    training_config = json.load(open(args['config_path']))


    subject_choice = training_config['subjects']
    print(f'[INFO]subjects: {subject_choice}')
    eeg_type_choice = training_config['eeg_type']
    print(f'[INFO]eeg type: {eeg_type_choice}')
    bands_choice = training_config['eeg_bands']
    print(f'[INFO]using bands: {bands_choice}')
    
    dataset_setting = 'unique_sent'

    task_name = training_config['task_name']
    model_name = training_config['model_name']

    # 兼容旧配置：若不存在新字段则回退到默认值
    eeg_encoder_type = training_config.get('eeg_encoder_type', 'transformer')
    mamba_num_layers = training_config.get('mamba_num_layers', 6)
    mamba_d_state = training_config.get('mamba_d_state', 16)
    mamba_d_conv = training_config.get('mamba_d_conv', 4)
    mamba_expand = training_config.get('mamba_expand', 2)
    mamba_dropout = training_config.get('mamba_dropout', 0.1)
    bimamba_fusion = training_config.get('bimamba_fusion', 'concat_linear')

    gat_num_layers = training_config.get('gat_num_layers', 2)
    gat_dropout = training_config.get('gat_dropout', 0.1)

    use_contrastive_align = training_config.get('use_contrastive_align', False)
    contrastive_proj_dim = training_config.get('contrastive_proj_dim', 768)
    contrastive_temperature = training_config.get('contrastive_temperature', 0.07)
    text_embed_model = training_config.get('text_embed_model', 'sentence-transformers/all-mpnet-base-v2')
    

    # 根据 train/test 输入类型决定结果文件命名
    if test_input == 'EEG' and train_input=='EEG':
        print("EEG and EEG")
        output_all_results_path = f'./results/{task_name}-{model_name}-all_decoding_results.txt'
        score_results = f'./score_results/{task_name}-{model_name}.txt'
    else:
        output_all_results_path = f'./results/{task_name}-{model_name}-{train_input}_{test_input}-all_decoding_results.txt'
        score_results = f'./score_results/{task_name}-{model_name}-{train_input}_{test_input}.txt'


    ''' 设置随机种子，保证评估可复现 '''
    seed_val = 20 #500
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' 配置运行设备 '''
    # use cuda
    if torch.cuda.is_available():  
        dev = 0
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')

    # task_name = 'task1_task2_task3'

    ''' 构建测试集与 DataLoader '''
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
    
    if model_name in ['BrainTranslator','BrainTranslatorNaive']:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    elif model_name == 'PegasusTranslator':
        tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
    
    elif model_name == 'T5Translator':
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
        # tokenizer.set_prefix_tokens(language='english')

    # 构建测试集
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=test_input)

    dataset_sizes = {"test_set":len(test_set)}
    print('[INFO]test_set size: ', len(test_set))
    
    # 构建 dataloader
    test_dataloader = DataLoader(test_set, batch_size = batch_size, shuffle=False, num_workers=4)

    dataloaders = {'test':test_dataloader}

    ''' 构建模型并加载 checkpoint '''
    checkpoint_path = args['checkpoint_path']
    
    # 根据模型名称构建对应结构；BrainTranslator 分支注入 bimamba/对比参数
    if model_name == 'BrainTranslator':
        pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        model = BrainTranslator(
            pretrained_bart,
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
        pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        # Naive 版本不包含额外 Mamba/对比模块
        model = BrainTranslatorNaive(pretrained_bart, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

    elif model_name == 'BertGeneration':
        pretrained = BertGenerationDecoder.from_pretrained('google-bert/bert-large-uncased', is_decoder = True)
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
    

    # 兼容 DataParallel 保存参数时的 module. 前缀
    state_dict = torch.load(checkpoint_path)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    '''
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(checkpoint_path))
    else:
        model.load_state_dict(torch.load(checkpoint_path))
    '''

    # model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    ''' 执行评估 '''
    eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path = output_all_results_path, score_results=score_results)

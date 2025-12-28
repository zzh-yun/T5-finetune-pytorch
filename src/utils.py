"""
Utility functions for data loading and text preprocessing
Reused from Transformer_NMT project
"""
import json
import nltk
import jieba
from nltk import word_tokenize
from pathlib import Path
import re
import sentencepiece as spm

# load tokenizer from local path
nltk.data.path.append('/data/250010009/course/nlpAllms/data/')

def read_corpus(file_path, source, zh_spm_model_path='save_dir/zh_spm_model.model', 
                en_spm_model_path='save_dir/en_spm_model.model'):
    """Read file, where each sentence is delineated by a "\n".

    Args:
        file_path (str): Path to file containing corpus.
        source (str): "src" or "tgt" indicating whether text is of the source language or
            target language.
        zh_spm_model_path: Path to Chinese SPM model
        en_spm_model_path: Path to English SPM model

    Returns:
        data (List[List(str)]): Sentences as a list of list of SPM tokens.
    """
    assert source=='en' or source=='zh', f"source is incorrect, only support en / zh"
    
    # 加载对应的SPM分词器
    sp = spm.SentencePieceProcessor()
    if source == "en":
        sp.load(en_spm_model_path)
    else:
        sp.load(zh_spm_model_path)
    
    data = []    
    with open(file_path, 'rb') as file:
        
        for line_num, line in enumerate(file, 1):
            try:
                if source == "en":
                    line_data = json.loads(line.strip())
                    line_data = line_data[source]
                    clean_data = clean_text_en(line_data)
                    # 使用英文SPM编码
                    sent = sp.encode(clean_data, out_type=str)
                    data.append(sent)
                
                else:
                    line_data = json.loads(line.strip())
                    line_data = line_data[source]
                    clean_data = clean_text_zh(line_data)
                    # 使用jieba分词
                    zh_tokens = list(jieba.cut(clean_data, cut_all=False))
                    # 用空格连接，然后使用中文SPM编码
                    zh_text = ' '.join(zh_tokens)
                    sent = sp.encode(zh_text, out_type=str)
                    data.append(sent)

            
            except json.JSONDecodeError as e:
                print(f"the {line_num} th row decode error")
                continue

    return data


def get_data(root_path, src='zh', tgt='en', zh_spm_model_path='save_dir/zh_spm_model.model',
             en_spm_model_path='save_dir/en_spm_model.model'):
    """
    get all of data
    params:
    - src | str : control the source language
    - tgt | str : control the target language
    - zh_spm_model_path: Path to Chinese SPM model
    - en_spm_model_path: Path to English SPM model
    """
    train_path = Path(root_path, 'train_100k.jsonl')   # 修改训练集
    valid_path = Path(root_path, 'valid.jsonl')
    test_path = Path(root_path, 'test.jsonl')
    # train data
    src_train_sents = read_corpus(train_path, src, zh_spm_model_path=zh_spm_model_path,
                                   en_spm_model_path=en_spm_model_path)
    tgt_train_sents = read_corpus(train_path, tgt, zh_spm_model_path=zh_spm_model_path,
                                  en_spm_model_path=en_spm_model_path)
    # valid data
    src_valid_sents = read_corpus(valid_path, src, zh_spm_model_path=zh_spm_model_path,
                                  en_spm_model_path=en_spm_model_path)
    tgt_valid_sents = read_corpus(valid_path, tgt, zh_spm_model_path=zh_spm_model_path,
                                   en_spm_model_path=en_spm_model_path)
    # test data
    src_test_sents = read_corpus(test_path, src, zh_spm_model_path=zh_spm_model_path,
                                  en_spm_model_path=en_spm_model_path)
    tgt_test_sents = read_corpus(test_path, tgt, zh_spm_model_path=zh_spm_model_path,
                                 en_spm_model_path=en_spm_model_path)

    return (src_train_sents, tgt_train_sents,
            src_valid_sents, tgt_valid_sents,
            src_test_sents, tgt_test_sents)


def clean_text_en(text):
    """
    Preprocess the text, including:
      Convert to lowercase letters
      Remove punctuation marks
      Remove numbers
      Remove redundant spaces
    """
    # text = text.lower()  
    text = re.sub(r'[^\w\s]', '', text)    #  不移除标点
    # text = re.sub(r'\d+', '', text)  # delete number, not adaptive for this data
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

def clean_text_zh(text):
    text = re.sub(r'[^\w\s]', '', text)   # Remove punctuation marks
    text = re.sub(r'\s+', ' ', text).strip()   # Remove redundant spaces
    return text


def extract_en_corpus(jsonl_path, output_corpus_path):
    """
    从JSONL文件中提取en字段，保存为纯文本文件（每行一个英文句子）
    """
    with open(jsonl_path, 'r', encoding='utf-8') as in_f, \
         open(output_corpus_path, 'w', encoding='utf-8') as out_f:
        
        for line_num, line in enumerate(in_f, 1):
            try:
                # 解析每行的JSON字典
                line_data = json.loads(line.strip())
                # 提取en字段的文本（确保字段存在）
                en_text = line_data.get('en', '').strip()
                # 跳过空文本
                if not en_text:
                    print(f"第{line_num}行：en字段为空，跳过")
                    continue
                # 写入纯文本文件（每行一个英文句子）
                out_f.write(en_text + '\n')
            except json.JSONDecodeError:
                print(f"第{line_num}行：JSON解析失败，跳过")
            except Exception as e:
                print(f"第{line_num}行：处理失败 - {e}，跳过")
    
    print(f"英文语料提取完成，保存至：{output_corpus_path}")


def extract_zh_corpus(jsonl_path, output_corpus_path):
    """
    从JSONL文件中提取zh字段，保存为纯文本文件（每行一个中文句子）
    使用jieba分词后，用空格连接，以便训练SentencePiece模型
    """
    with open(jsonl_path, 'r', encoding='utf-8') as in_f, \
         open(output_corpus_path, 'w', encoding='utf-8') as out_f:
        
        for line_num, line in enumerate(in_f, 1):
            try:
                # 解析每行的JSON字典
                line_data = json.loads(line.strip())
                # 提取zh字段的文本（确保字段存在）
                zh_text = line_data.get('zh', '').strip()
                # 跳过空文本
                if not zh_text:
                    print(f"第{line_num}行：zh字段为空，跳过")
                    continue
                # 清理文本并使用jieba分词
                clean_data = clean_text_zh(zh_text)
                zh_tokens = list(jieba.cut(clean_data, cut_all=False))
                # 用空格连接分词结果
                zh_segmented = ' '.join(zh_tokens)
                # 写入纯文本文件（每行一个中文句子，已分词）
                out_f.write(zh_segmented + '\n')
            except json.JSONDecodeError:
                print(f"第{line_num}行：JSON解析失败，跳过")
            except Exception as e:
                print(f"第{line_num}行：处理失败 - {e}，跳过")
    
    print(f"中文语料提取完成，保存至：{output_corpus_path}")

def train_zh_spm_model(corpus_path, output_dir='save_dir', vocab_size=32000):
    """
    训练中文SentencePiece模型
    
    Args:
        corpus_path: 中文语料文件路径（已分词，每行一个句子）
        output_dir: 输出目录
        vocab_size: 词汇表大小
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, 'zh_spm_model')
    
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
        bos_piece="<s>", eos_piece="</s>", pad_piece="<pad>", unk_piece="<unk>"
    )
    
    print(f"中文SPM模型训练完成，保存至：{model_prefix}.model")


def train_en_spm_model(corpus_path, output_dir='save_dir', vocab_size=32000):
    """
    训练英文SentencePiece模型
    
    Args:
        corpus_path: 英文语料文件路径
        output_dir: 输出目录
        vocab_size: 词汇表大小
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, 'en_spm_model')
    
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
        bos_piece="<s>", eos_piece="</s>", pad_piece="<pad>", unk_piece="<unk>"
    )
    
    print(f"英文SPM模型训练完成，保存至：{model_prefix}.model")


if __name__ == "__main__":
    root_path = "/data/250010009/course/nlpAllms/data/translation_dataset_zh_en"
    train_path = Path(root_path, 'train_100k.jsonl')
    
    # 提取英文语料
    en_corpus_path = Path(root_path, 'english_corpus.txt')
    extract_en_corpus(train_path, en_corpus_path)
    
    # 提取中文语料
    zh_corpus_path = Path(root_path, 'chinese_corpus.txt')
    extract_zh_corpus(train_path, zh_corpus_path)
    
    # 训练英文SPM模型
    train_en_spm_model(en_corpus_path, output_dir='save_dir', vocab_size=32000)
    
    # 训练中文SPM模型
    train_zh_spm_model(zh_corpus_path, output_dir='save_dir', vocab_size=32000)
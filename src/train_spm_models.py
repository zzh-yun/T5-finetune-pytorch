#!/usr/bin/env python
"""
训练中文和英文SentencePiece模型
"""
import os
from pathlib import Path
from src.utils import extract_zh_corpus, extract_en_corpus, train_zh_spm_model, train_en_spm_model
import hydra
from omegaconf import DictConfig


@hydra.main(version_base='1.3', config_path='./configs', config_name='train.yaml')
def main(cfgs: DictConfig):
    """Train Chinese and English SentencePiece models"""
    print("=" * 60)
    print("训练中文和英文SentencePiece模型")
    print("=" * 60)
    
    # Get paths
    data_dir = cfgs.DATA_DIR
    save_dir = getattr(cfgs, 'SAVE_DIR', 'save_dir')
    train_path = Path(data_dir, 'train_100k.jsonl')
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract corpora
    print("\n1. 提取语料...")
    en_corpus_path = Path(data_dir, 'english_corpus.txt')
    zh_corpus_path = Path(data_dir, 'chinese_corpus.txt')
    
    if not en_corpus_path.exists() or not zh_corpus_path.exists():
        print(f"提取英文语料...")
        extract_en_corpus(train_path, en_corpus_path)
        
        print(f"提取中文语料...")
        extract_zh_corpus(train_path, zh_corpus_path)
    else:
        print("语料文件已存在，跳过提取步骤")
    
    # Train English SPM model
    print("\n2. 训练英文SPM模型...")
    en_spm_path = Path(save_dir, 'en_spm_model.model')
    if not en_spm_path.exists():
        train_en_spm_model(str(en_corpus_path), output_dir=save_dir, vocab_size=32000)
    else:
        print(f"英文SPM模型已存在: {en_spm_path}")
    
    # Train Chinese SPM model
    print("\n3. 训练中文SPM模型...")
    zh_spm_path = Path(save_dir, 'zh_spm_model.model')
    if not zh_spm_path.exists():
        train_zh_spm_model(str(zh_corpus_path), output_dir=save_dir, vocab_size=32000)
    else:
        print(f"中文SPM模型已存在: {zh_spm_path}")
    
    print("\n" + "=" * 60)
    print("SPM模型训练完成！")
    print("=" * 60)
    print(f"英文SPM模型: {en_spm_path}")
    print(f"中文SPM模型: {zh_spm_path}")
    print("\n现在可以开始训练T5模型了！")


if __name__ == "__main__":
    main()


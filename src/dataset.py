"""
Dataset classes for T5 Chinese-English translation
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import T5Tokenizer
from .utils import get_data
from .spm_tokenizer import SPMTokenizer
import os


class T5NMTDataset(Dataset):
    """Dataset for T5 Neural Machine Translation with SPM tokenization"""
    
    def __init__(self, zh_token_lists, en_token_lists, tokenizer, max_input_length=512, max_target_length=512,
                 zh_spm_model_path=None, en_spm_model_path=None, use_spm=True):
        """
        Initialize dataset
        
        Args:
            zh_token_lists: List of Chinese SPM token lists (from read_corpus)
            en_token_lists: List of English SPM token lists (from read_corpus)
            tokenizer: T5Tokenizer instance
            max_input_length: Maximum input sequence length
            max_target_length: Maximum target sequence length
            zh_spm_model_path: Path to Chinese SPM model (optional, for decoding)
            en_spm_model_path: Path to English SPM model (optional, for decoding)
            use_spm: Whether to use SPM tokens (if False, fallback to original method)
        """
        self.zh_token_lists = zh_token_lists
        self.en_token_lists = en_token_lists
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.use_spm = use_spm
        
        # Load SPM tokenizers if provided
        if use_spm:
            if zh_spm_model_path and os.path.exists(zh_spm_model_path):
                self.zh_spm = SPMTokenizer(zh_spm_model_path)
            else:
                self.zh_spm = None
                print("Warning: Chinese SPM model not found, falling back to T5 tokenizer")
            
            if en_spm_model_path and os.path.exists(en_spm_model_path):
                self.en_spm = SPMTokenizer(en_spm_model_path)
            else:
                self.en_spm = None
                print("Warning: English SPM model not found, falling back to T5 tokenizer")
    
    def __len__(self):
        return len(self.zh_token_lists)
    
    def __getitem__(self, idx):
        # zh_token_lists and en_token_lists are already SPM token lists (from read_corpus)
        # Join SPM tokens with space to form text
        zh_spm_tokens = self.zh_token_lists[idx]
        en_spm_tokens = self.en_token_lists[idx]
        
        # Convert SPM token lists to strings
        zh_text = " ".join(zh_spm_tokens)
        en_text = " ".join(en_spm_tokens)
        
        # Build input text with T5 translation prefix
        # Note: SPM tokens may contain special characters like "▁" which T5 tokenizer might not recognize
        # But at least the subword structure is preserved
        input_text = f"translate Chinese to English: {zh_text}"
        
        # Encode input using T5 tokenizer
        # T5 tokenizer will handle the SPM tokens (may encode some as <unk> if not in vocab)
        input_encoded = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Encode target using T5 tokenizer
        target_encoded = self.tokenizer(
            en_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoded['input_ids'].squeeze(0),
            'attention_mask': input_encoded['attention_mask'].squeeze(0),
            'labels': target_encoded['input_ids'].squeeze(0),
            'target_attention_mask': target_encoded['attention_mask'].squeeze(0)
        }


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32, 
                        num_workers=4, distributed=False):
    """
    Create data loaders for train/val/test
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        distributed: Whether to use distributed training
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    def collate_fn(batch):
        """Collate function to pad sequences"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        target_attention_mask = torch.stack([item['target_attention_mask'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'target_attention_mask': target_attention_mask
        }
    
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class DataPreprocessor:
    """Data preprocessing class for T5 NMT"""
    
    def __init__(self, config, tokenizer):
        """
        Initialize data preprocessor
        
        Args:
            config: Configuration object
            tokenizer: T5Tokenizer instance
        """
        self.config = config
        self.data_dir = config.DATA_DIR
        self.tokenizer = tokenizer
        
        # Get SPM model paths from config
        save_dir = getattr(config, 'SAVE_DIR', 'save_dir')
        self.zh_spm_path = os.path.join(save_dir, 'zh_spm_model.model')
        self.en_spm_path = os.path.join(save_dir, 'en_spm_model.model')
    
    def create_datasets(self):
        """Create train, validation, and test datasets"""
        # Load data (now returns SPM token lists)
        (train_zh_tokens, train_en_tokens,
         valid_zh_tokens, valid_en_tokens,
         test_zh_tokens, test_en_tokens) = get_data(
            self.data_dir,
            zh_spm_model_path=self.zh_spm_path,
            en_spm_model_path=self.en_spm_path
        )
        
        # Check if SPM models exist
        use_spm = os.path.exists(self.zh_spm_path) and os.path.exists(self.en_spm_path)
        if use_spm:
            print(f"Using SPM models: {self.zh_spm_path}, {self.en_spm_path}")
        else:
            print("Warning: SPM models not found, using T5 tokenizer directly")
        
        # Create datasets
        train_dataset = T5NMTDataset(
            train_zh_tokens,
            train_en_tokens,
            self.tokenizer,
            max_input_length=getattr(self.config, 'MAX_INPUT_LENGTH', 512),
            max_target_length=getattr(self.config, 'MAX_TARGET_LENGTH', 512),
            zh_spm_model_path=self.zh_spm_path if use_spm else None,
            en_spm_model_path=self.en_spm_path if use_spm else None,
            use_spm=use_spm
        )
        
        val_dataset = T5NMTDataset(
            valid_zh_tokens,
            valid_en_tokens,
            self.tokenizer,
            max_input_length=getattr(self.config, 'MAX_INPUT_LENGTH', 512),
            max_target_length=getattr(self.config, 'MAX_TARGET_LENGTH', 512),
            zh_spm_model_path=self.zh_spm_path if use_spm else None,
            en_spm_model_path=self.en_spm_path if use_spm else None,
            use_spm=use_spm
        )
        
        test_dataset = T5NMTDataset(
            test_zh_tokens,
            test_en_tokens,
            self.tokenizer,
            max_input_length=getattr(self.config, 'MAX_INPUT_LENGTH', 512),
            max_target_length=getattr(self.config, 'MAX_TARGET_LENGTH', 512),
            zh_spm_model_path=self.zh_spm_path if use_spm else None,
            en_spm_model_path=self.en_spm_path if use_spm else None,
            use_spm=use_spm
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset

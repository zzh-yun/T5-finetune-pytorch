"""
T5 Model wrapper for Chinese-English translation
"""
import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5NMTModel:
    """T5 Model wrapper for Neural Machine Translation"""
    
    def __init__(self, pretrained_model_path, device=None):
        """
        Initialize T5 model and tokenizer
        
        Args:
            pretrained_model_path: Path to pretrained T5 model directory
            device: torch.device or None (auto-detect)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_path)
        
        # Load model
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"T5 model loaded from {pretrained_model_path}")
        print(f"Model device: {self.device}")
        print(f"Vocab size: {self.tokenizer.vocab_size}")
    
    def encode_input(self, chinese_text, max_length=512):
        """
        Encode Chinese text with T5 format: "translate Chinese to English: <text>"
        
        Args:
            chinese_text: Chinese text (already tokenized with jieba, space-separated)
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids, attention_mask
        """
        # Build input text with T5 translation prefix
        input_text = f"translate Chinese to English: {chinese_text}"
        
        # Tokenize
        encoded = self.tokenizer(
            input_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device)
        }
    
    def encode_target(self, english_text, max_length=512):
        """
        Encode English target text
        
        Args:
            english_text: English text (space-separated tokens)
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids, attention_mask
        """
        encoded = self.tokenizer(
            english_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device)
        }
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token ids to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def generate(self, input_ids, attention_mask=None, max_length=128, num_beams=4, 
                 early_stopping=True, no_repeat_ngram_size=2, **kwargs):
        """
        Generate translation using T5 model
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            max_length: Maximum generation length
            num_beams: Beam search size
            early_stopping: Whether to use early stopping
            no_repeat_ngram_size: N-gram size for repetition penalty
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token ids
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                no_repeat_ngram_size=no_repeat_ngram_size,
                **kwargs
            )
        
        return outputs
    
    def train_mode(self):
        """Set model to training mode"""
        self.model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode"""
        self.model.eval()
    
    def save_pretrained(self, save_directory):
        """Save model and tokenizer"""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
    
    def load_state_dict(self, state_dict):
        """Load model state dict"""
        self.model.load_state_dict(state_dict)

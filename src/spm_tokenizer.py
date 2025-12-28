"""
SentencePiece Tokenizer wrapper for T5 NMT
Supports both Chinese and English SPM models
"""
import sentencepiece as spm
import torch
from typing import List, Union
import os


class SPMTokenizer:
    """SentencePiece Tokenizer wrapper"""
    
    def __init__(self, model_path: str):
        """
        Initialize SPM tokenizer
        
        Args:
            model_path: Path to SentencePiece model file (.model)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SPM model not found: {model_path}")
        
        self.processor = spm.SentencePieceProcessor()
        self.processor.load(model_path)
        self.model_path = model_path
        
        # Get special token IDs
        self.pad_id = self.processor.pad_id()
        self.bos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()
        self.unk_id = self.processor.unk_id()
        
        # Get vocab size
        self.vocab_size = self.processor.get_piece_size()
    
    def encode(self, text: str, add_special_tokens: bool = True, 
               return_tensors: str = None, max_length: int = None,
               truncation: bool = False, padding: bool = False) -> Union[List[int], torch.Tensor]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens (BOS/EOS)
            return_tensors: If 'pt', return PyTorch tensor
            max_length: Maximum sequence length
            truncation: Whether to truncate if exceeds max_length
            padding: Whether to pad (not implemented for SPM, use in collate_fn)
            
        Returns:
            Token IDs as list or tensor
        """
        # Encode text
        if add_special_tokens:
            ids = self.processor.encode(text, out_type=int, add_bos=True, add_eos=True)
        else:
            ids = self.processor.encode(text, out_type=int, add_bos=False, add_eos=False)
        
        # Truncate if needed
        if max_length and truncation and len(ids) > max_length:
            if add_special_tokens:
                # Keep BOS and EOS, truncate middle
                ids = [ids[0]] + ids[1:max_length-1] + [ids[-1]]
            else:
                ids = ids[:max_length]
        
        # Convert to tensor if requested
        if return_tensors == 'pt':
            return torch.tensor(ids, dtype=torch.long)
        
        return ids
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], 
               skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: List of token IDs or tensor
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        # Convert tensor to list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Remove special tokens if requested
        if skip_special_tokens:
            token_ids = [id for id in token_ids 
                        if id not in [self.pad_id, self.bos_id, self.eos_id, self.unk_id]]
        
        # Decode
        text = self.processor.decode(token_ids)
        return text
    
    def __len__(self):
        """Return vocabulary size"""
        return self.vocab_size


class DualSPMTokenizer:
    """Dual SPM tokenizer for Chinese and English"""
    
    def __init__(self, zh_spm_path: str, en_spm_path: str):
        """
        Initialize dual SPM tokenizers
        
        Args:
            zh_spm_path: Path to Chinese SPM model
            en_spm_path: Path to English SPM model
        """
        self.zh_tokenizer = SPMTokenizer(zh_spm_path)
        self.en_tokenizer = SPMTokenizer(en_spm_path)
    
    def encode_zh(self, text: str, **kwargs) -> Union[List[int], torch.Tensor]:
        """Encode Chinese text"""
        return self.zh_tokenizer.encode(text, **kwargs)
    
    def encode_en(self, text: str, **kwargs) -> Union[List[int], torch.Tensor]:
        """Encode English text"""
        return self.en_tokenizer.encode(text, **kwargs)
    
    def decode_zh(self, token_ids: Union[List[int], torch.Tensor], **kwargs) -> str:
        """Decode Chinese token IDs"""
        return self.zh_tokenizer.decode(token_ids, **kwargs)
    
    def decode_en(self, token_ids: Union[List[int], torch.Tensor], **kwargs) -> str:
        """Decode English token IDs"""
        return self.en_tokenizer.decode(token_ids, **kwargs)


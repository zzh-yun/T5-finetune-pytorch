"""
Inference script for T5 Chinese-English translation
"""
import torch
import argparse
import os
import jieba
from typing import List, Optional
import time

import hydra
from omegaconf import DictConfig
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.utils import clean_text_zh, get_data
from src.spm_tokenizer import SPMTokenizer
from evaluation import BLEUEvaluator
import os


class T5NMTInference:
    """T5 Neural Machine Translation Inference Engine"""
    
    def __init__(self, model_path: str, config, device: str = 'auto'):
        """Initialize the inference engine"""
        self.device = self._setup_device(device)
        self.config = config
        
        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.PRETRAINED_MODEL_PATH)
        
        # Load SPM tokenizers if available
        # Try multiple possible locations for SPM models
        possible_dirs = [
            getattr(config, 'SAVE_DIR', 'save_dir'),
            'save_dir',
            os.path.dirname(model_path) if hasattr(config, 'model_path') and config.model_path else 'save_dir'
        ]
        
        zh_spm_path = None
        en_spm_path = None
        
        for save_dir in possible_dirs:
            zh_path = os.path.join(save_dir, 'zh_spm_model.model')
            en_path = os.path.join(save_dir, 'en_spm_model.model')
            if os.path.exists(zh_path) and zh_spm_path is None:
                zh_spm_path = zh_path
            if os.path.exists(en_path) and en_spm_path is None:
                en_spm_path = en_path
        
        if zh_spm_path and os.path.exists(zh_spm_path):
            try:
                self.zh_spm = SPMTokenizer(zh_spm_path)
                print(f"Chinese SPM model loaded: {zh_spm_path}")
            except Exception as e:
                print(f"Warning: Failed to load Chinese SPM model: {e}")
                self.zh_spm = None
        else:
            self.zh_spm = None
            print("Warning: Chinese SPM model not found, using T5 tokenizer directly")
        
        if en_spm_path and os.path.exists(en_spm_path):
            try:
                self.en_spm = SPMTokenizer(en_spm_path)
                print(f"English SPM model loaded: {en_spm_path}")
            except Exception as e:
                print(f"Warning: Failed to load English SPM model: {e}")
                self.en_spm = None
        else:
            self.en_spm = None
            print("Warning: English SPM model not found, using T5 tokenizer directly")
        
        # Load model
        self.model = self._load_model(model_path)
        
        print(f"Model loaded on {self.device}")
        print(f"Vocab size: {self.tokenizer.vocab_size}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                print("Using CPU")
        
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> T5ForConditionalGeneration:
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Check if it's a directory (pretrained model) or a checkpoint file
        if os.path.isdir(model_path):
            # Load from pretrained model directory
            print(f"Loading model from pretrained directory: {model_path}")
            print("=" * 60)
            print("WARNING: Using pretrained model without fine-tuning!")
            print("The pretrained T5 model is NOT trained for Chinese-English translation.")
            print("Translation results will be poor or incorrect.")
            print("Please train the model first using train.py, then use the trained checkpoint.")
            print("=" * 60)
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            model.to(self.device)
            model.eval()
            print("Model loaded successfully from pretrained directory")
            return model
        
        # Load from checkpoint file
        print(f"Loading model from checkpoint file: {model_path}")
        
        # Check if it's actually a pretrained model file (pytorch_model.bin)
        if 'pytorch_model.bin' in model_path or not os.path.exists(model_path):
            print("=" * 60)
            print("WARNING: This appears to be a pretrained model file, not a trained checkpoint!")
            print("The pretrained T5 model is NOT trained for Chinese-English translation.")
            print("Please train the model first using train.py, then use outputs/checkpoint_best.pth")
            print("=" * 60)
        
        # Load checkpoint with error handling for corrupted files
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            # Try alternative loading methods
            print(f"Warning: Failed to load checkpoint with default method: {e}")
            try:
                # Try loading with pickle
                import pickle
                with open(model_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                print("Checkpoint loaded using pickle")
            except Exception as e2:
                raise ValueError(
                    f"Failed to load checkpoint from {model_path}.\n"
                    f"Error 1: {e}\nError 2: {e2}\n"
                    f"The checkpoint file may be corrupted. File size: {os.path.getsize(model_path) / 1e9:.2f} GB\n"
                    f"Try using checkpoint_best.pth instead, or retrain the model."
                )
        
        # Load model from pretrained path first
        model = T5ForConditionalGeneration.from_pretrained(self.config.PRETRAINED_MODEL_PATH)
        
        # Get state dict from checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # If checkpoint is just the state dict itself
            state_dict = checkpoint
        
        # Handle T5's shared weights: embed_tokens and lm_head share the same weight
        # If checkpoint doesn't have these keys, copy from model's current state
        model_state_dict = model.state_dict()
        if 'encoder.embed_tokens.weight' not in state_dict and 'encoder.embed_tokens.weight' in model_state_dict:
            state_dict['encoder.embed_tokens.weight'] = model_state_dict['encoder.embed_tokens.weight']
        if 'decoder.embed_tokens.weight' not in state_dict and 'decoder.embed_tokens.weight' in model_state_dict:
            state_dict['decoder.embed_tokens.weight'] = model_state_dict['decoder.embed_tokens.weight']
        if 'lm_head.weight' not in state_dict and 'lm_head.weight' in model_state_dict:
            state_dict['lm_head.weight'] = model_state_dict['lm_head.weight']
        
        # Remove unexpected keys that don't exist in the model
        model_keys = set(model_state_dict.keys())
        state_dict_keys = set(state_dict.keys())
        unexpected_keys = state_dict_keys - model_keys
        
        if unexpected_keys:
            print(f"Warning: Removing {len(unexpected_keys)} unexpected keys from checkpoint:")
            for key in list(unexpected_keys)[:5]:  # Show first 5
                print(f"  - {key}")
            if len(unexpected_keys) > 5:
                print(f"  ... and {len(unexpected_keys) - 5} more")
            state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        
        # Load state dict with strict=False to allow missing keys
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Warning: {len(missing_keys)} keys were missing and will use default values:")
                for key in list(missing_keys)[:5]:  # Show first 5
                    print(f"  - {key}")
                if len(missing_keys) > 5:
                    print(f"  ... and {len(missing_keys) - 5} more")
        except Exception as e:
            raise ValueError(
                f"Could not load model state dict from checkpoint. "
                f"Error: {e}"
            )
        
        model.to(self.device)
        model.eval()
        
        # Print checkpoint info if available
        if isinstance(checkpoint, dict):
            print(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")
            print(f"Best BLEU score: {checkpoint.get('bleu_score', 'N/A')}")
        else:
            print("Model loaded from checkpoint (no metadata available)")
        
        return model
    
    def translate_sentence(self, chinese_text: str, max_length: int = 128, num_beams: int = 4) -> str:
        """
        Translate a single Chinese sentence to English
        
        Args:
            chinese_text: Chinese text to translate
            max_length: Maximum generation length
            num_beams: Beam search size
            
        Returns:
            Translated English text
        """
        # Preprocess Chinese text
        clean_data = clean_text_zh(chinese_text)
        
        # Use SPM if available, otherwise use jieba
        if self.zh_spm is not None:
            # Use jieba for word segmentation, then SPM for subword tokenization
            zh_tokens = list(jieba.cut(clean_data, cut_all=False))
            zh_text = " ".join(zh_tokens)
            # Encode with SPM to get subword tokens (as strings)
            zh_spm_tokens = self.zh_spm.processor.encode(zh_text, out_type=str)
            zh_text = " ".join(zh_spm_tokens)
        else:
            # Fallback to original method
            zh_tokens = list(jieba.cut(clean_data, cut_all=False))
            zh_text = " ".join(zh_tokens)
        
        # Build input text with T5 translation prefix
        input_text = f"translate Chinese to English: {zh_text}"
        
        # Debug: print input text (only in verbose mode)
        if getattr(self.config, 'verbose', False):
            print(f"Debug - Input text: {input_text}")
        
        # Encode input
        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                do_sample=False,  # Use greedy/beam search, not sampling
                temperature=1.0
            )
        
        # Decode output
        translated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Debug: print raw output (only in verbose mode)
        if getattr(self.config, 'verbose', False):
            print(f"Debug - Raw output tokens: {generated_ids[0].cpu().tolist()[:20]}")
            print(f"Debug - Translated text: {translated_text}")
        
        return translated_text
    
    def translate_batch(self, sentences: List[str], max_length: int = 128, num_beams: int = 4) -> List[str]:
        """Translate a batch of sentences"""
        results = []
        
        for sentence in sentences:
            translated = self.translate_sentence(sentence, max_length, num_beams)
            results.append(translated)
        
        return results
    
    def evaluate_on_test_set(self, test_file: str, reference_file: str = None) -> dict:
        """
        Evaluate model on test set
        
        Args:
            test_file: Path to test file (Chinese sentences, one per line)
            reference_file: Path to reference file (English sentences, one per line)
            
        Returns:
            Dictionary with evaluation results
        """
        # Load test sentences
        with open(test_file, 'r', encoding='utf-8') as f:
            test_sentences = [line.strip() for line in f.readlines()]
        
        print(f"Translating {len(test_sentences)} test sentences...")
        
        # Translate
        start_time = time.time()
        translations = []
        
        for i, sentence in enumerate(test_sentences):
            if i % 100 == 0:
                print(f"Translated {i}/{len(test_sentences)} sentences")
            
            translated = self.translate_sentence(sentence)
            translations.append(translated)
        
        translation_time = time.time() - start_time
        
        print(f"Translation completed in {translation_time:.2f} seconds")
        print(f"Average time per sentence: {translation_time/len(test_sentences):.3f} seconds")
        
        # Evaluate if reference is provided
        results = {
            'num_sentences': len(test_sentences),
            'translation_time': translation_time,
            'avg_time_per_sentence': translation_time / len(test_sentences)
        }
        
        if reference_file and os.path.exists(reference_file):
            with open(reference_file, 'r', encoding='utf-8') as f:
                references = [line.strip() for line in f.readlines()]
            
            if len(references) == len(translations):
                evaluator = BLEUEvaluator()
                
                # Calculate BLEU scores
                references_list = [[r] for r in references]
                bleu_1 = evaluator.corpus_bleu(references_list, translations, n_gram=1)
                bleu_2 = evaluator.corpus_bleu(references_list, translations, n_gram=2)
                bleu_3 = evaluator.corpus_bleu(references_list, translations, n_gram=3)
                bleu_4 = evaluator.corpus_bleu(references_list, translations, n_gram=4)
                
                results.update({
                    'bleu_1': bleu_1,
                    'bleu_2': bleu_2,
                    'bleu_3': bleu_3,
                    'bleu_4': bleu_4
                })
                
                print(f"\nBLEU Scores:")
                print(f"  BLEU-1: {bleu_1:.4f}")
                print(f"  BLEU-2: {bleu_2:.4f}")
                print(f"  BLEU-3: {bleu_3:.4f}")
                print(f"  BLEU-4: {bleu_4:.4f}")
            else:
                print(f"Warning: Number of references ({len(references)}) != translations ({len(translations)})")
        
        return results, translations
    
    def interactive_translation(self):
        """Interactive translation mode"""
        print("=" * 60)
        print("INTERACTIVE TRANSLATION MODE")
        print("Enter Chinese sentences to translate to English")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 60)
        
        while True:
            try:
                # Get input
                chinese_text = input("\n中文输入: ").strip()
                
                if chinese_text.lower() in ['quit', 'exit', '退出']:
                    break
                
                if not chinese_text:
                    continue
                
                # Translate
                start_time = time.time()
                english_text = self.translate_sentence(chinese_text)
                translation_time = time.time() - start_time
                
                # Display result
                print(f"English: {english_text}")
                print(f"Time: {translation_time:.3f}s")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Translation error: {e}")
        
        print("\nGoodbye!")


@hydra.main(version_base='1.3', config_path='./configs', config_name='inference.yaml')
def main(cfgs: DictConfig) -> Optional[float]:
    
    try:
        inference_engine = T5NMTInference(cfgs.model_path, cfgs, cfgs.device)
        
        if cfgs.interactive:
            inference_engine.interactive_translation()
        elif cfgs.input_text:
            translation = inference_engine.translate_sentence(cfgs.input_text)
            print(f"Input:    {cfgs.input_text}")
            print(f"Translated: {translation}")
        elif cfgs.input_file:
            with open(cfgs.input_file, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f]
            
            translations = inference_engine.translate_batch(sentences)
            
            with open(cfgs.output_file, 'w', encoding='utf-8') as f:
                for t in translations:
                    f.write(t + '\n')
            
            print(f"Translations saved to {cfgs.output_file}")
            
            if cfgs.reference_file:
                results, _ = inference_engine.evaluate_on_test_set(cfgs.input_file, cfgs.reference_file)
                print("\nEvaluation Results:")
                for key, value in results.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
        
        else:
            print("No action specified. Use --interactive, --input_text, or --input_file.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the model path and vocabulary files are correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()

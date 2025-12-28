import nltk
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import os
import jieba
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.utils import get_data, clean_text_zh
from src.spm_tokenizer import SPMTokenizer
import jieba


class BLEUEvaluator:
    """BLEU score evaluator for machine translation"""
    
    def __init__(self):
        # Download NLTK data if needed
        try:
            nltk.data.path.append('/data/250010009/course/nlpAllms/data/')
        except LookupError:
            nltk.download('punkt')
    
    def sentence_bleu(self, reference: List[str], candidate: str, n_gram=4) -> float:
        """Calculate BLEU score for a single sentence"""
        reference_tokens = [ref.split() for ref in reference]
        candidate_tokens = candidate.split()
        
        return self._compute_bleu(reference_tokens, candidate_tokens, n_gram)
    
    def corpus_bleu(self, references: List[List[str]], candidates: List[str], n_gram=4) -> float:
        """Calculate BLEU score for a corpus (standard corpus-level BLEU-4)"""
        if len(references) != len(candidates):
            raise ValueError("Number of references and candidates must match")
        
        # Standard BLEU-4: calculate precision at corpus level
        # Aggregate all n-grams across the entire corpus
        total_clipped_matches = [0] * n_gram  # For each n-gram order
        total_candidate_ngrams = [0] * n_gram
        
        total_candidate_length = 0
        total_reference_length = 0
        
        for ref_list, cand in zip(references, candidates):
            if not cand.strip():  # Skip empty candidates
                continue
                
            candidate_tokens = cand.split()
            reference_tokens_list = [ref.split() for ref in ref_list]
            
            # Update length statistics
            total_candidate_length += len(candidate_tokens)
            # Use closest reference length for this sentence
            closest_ref_len = min([len(ref_tokens) for ref_tokens in reference_tokens_list],
                                 key=lambda x: abs(x - len(candidate_tokens)))
            total_reference_length += closest_ref_len
            
            # Calculate n-gram matches for each order
            for n in range(1, n_gram + 1):
                candidate_ngrams = self._get_ngrams(candidate_tokens, n)
                if not candidate_ngrams:
                    continue
                
                reference_ngrams_list = [self._get_ngrams(ref_tokens, n) for ref_tokens in reference_tokens_list]
                
                # Count clipped matches
                matches = 0
                total_ngrams_in_candidate = sum(candidate_ngrams.values())  # Total count, not unique count
                for ngram, count in candidate_ngrams.items():
                    max_ref_count = max([ref_ngrams.get(ngram, 0) for ref_ngrams in reference_ngrams_list])
                    matches += min(count, max_ref_count)
                
                total_clipped_matches[n-1] += matches
                total_candidate_ngrams[n-1] += total_ngrams_in_candidate
        
        # Calculate precision for each n-gram order
        precisions = []
        for n in range(n_gram):
            if total_candidate_ngrams[n] > 0:
                precision = total_clipped_matches[n] / total_candidate_ngrams[n]
                precisions.append(precision)
            else:
                precisions.append(0.0)
        
        # Calculate geometric mean of all precisions (standard BLEU-4)
        # If any precision is 0, the geometric mean is 0
        if any(p == 0 for p in precisions):
            return 0.0
        
        geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
        
        # Calculate brevity penalty at corpus level
        # Standard formula: BP = exp(1 - r/c) if c < r, else 1.0
        # where r = reference_length, c = candidate_length
        if total_candidate_length == 0:
            return 0.0
        if total_candidate_length >= total_reference_length:
            brevity_penalty = 1.0
        else:
            brevity_penalty = np.exp(1 - total_reference_length / total_candidate_length) if total_reference_length > 0 else 0.0
        
        # Return as percentage (0-100 scale) to match standard BLEU reporting
        return (brevity_penalty * geo_mean) * 100
    
    def _compute_bleu(self, references: List[List[str]], candidate: List[str], n_gram: int) -> float:
        """Compute BLEU score using modified precision and brevity penalty"""
        if not candidate:
            return 0.0
        
        # Calculate precision for each n-gram
        precisions = []
        
        for n in range(1, n_gram + 1):
            # Get n-grams from candidate
            candidate_ngrams = self._get_ngrams(candidate, n)
            if not candidate_ngrams:
                precisions.append(0.0)
                continue
            
            # Get n-grams from all references
            reference_ngrams_list = [self._get_ngrams(ref, n) for ref in references]
            
            # Calculate modified precision
            matches = 0
            total_candidate_ngrams = len(candidate_ngrams)
            
            for ngram in candidate_ngrams:
                # Find maximum count of this n-gram in any reference
                max_ref_count = max([ref_ngrams.get(ngram, 0) for ref_ngrams in reference_ngrams_list])
                # Count in candidate
                candidate_count = candidate_ngrams[ngram]
                # Clipped count
                matches += min(candidate_count, max_ref_count)
            
            precision = matches / total_candidate_ngrams if total_candidate_ngrams > 0 else 0.0
            precisions.append(precision)
        
        # Calculate geometric mean of precisions
        # Only use non-zero precisions for geometric mean (standard BLEU calculation)
        non_zero_precisions = [p for p in precisions if p > 0]
        if not non_zero_precisions:
            return 0.0
        
        geo_mean = np.exp(np.mean([np.log(p) for p in non_zero_precisions]))
        
        # Calculate brevity penalty
        candidate_length = len(candidate)
        closest_ref_length = min([len(ref) for ref in references], 
                                key=lambda x: abs(x - candidate_length))
        
        if candidate_length > closest_ref_length:
            brevity_penalty = 1.0
        else:
            brevity_penalty = np.exp(1 - closest_ref_length / candidate_length)
        
        return (brevity_penalty * geo_mean) * 100   # 改成100分制
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Extract n-grams from token list"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        return Counter(ngrams)


def compute_bleu_score(references: List[str], hypotheses: List[str]) -> float:
    """Convenient function to compute BLEU-4 score"""
    evaluator = BLEUEvaluator()
    
    # Convert to required format
    ref_lists = [[ref] for ref in references]
    
    return evaluator.corpus_bleu(ref_lists, hypotheses, n_gram=4)


class TranslationEvaluator:
    """Comprehensive translation evaluation including BLEU and other metrics"""
    
    def __init__(self):
        self.bleu_evaluator = BLEUEvaluator()
    
    def evaluate(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """Evaluate translations using multiple metrics"""
        results = {}
        
        # BLEU scores
        results['bleu_1'] = self.bleu_evaluator.corpus_bleu([[ref] for ref in references], hypotheses, 1)
        results['bleu_2'] = self.bleu_evaluator.corpus_bleu([[ref] for ref in references], hypotheses, 2)
        results['bleu_3'] = self.bleu_evaluator.corpus_bleu([[ref] for ref in references], hypotheses, 3)
        results['bleu_4'] = self.bleu_evaluator.corpus_bleu([[ref] for ref in references], hypotheses, 4)
        
        # Additional metrics
        results['exact_match'] = self._exact_match_score(references, hypotheses)
        results['avg_length_ratio'] = self._avg_length_ratio(references, hypotheses)
        
        return results
    
    def _exact_match_score(self, references: List[str], hypotheses: List[str]) -> float:
        """Calculate exact match score"""
        matches = sum(1 for ref, hyp in zip(references, hypotheses) if ref.strip() == hyp.strip())
        return matches / len(references) if references else 0.0
    
    def _avg_length_ratio(self, references: List[str], hypotheses: List[str]) -> float:
        """Calculate average length ratio (hypothesis/reference)"""
        ratios = []
        for ref, hyp in zip(references, hypotheses):
            ref_len = len(ref.split())
            hyp_len = len(hyp.split())
            if ref_len > 0:
                ratios.append(hyp_len / ref_len)
        
        return np.mean(ratios) if ratios else 0.0


def evaluate_model_predictions(model, data_loader, src_vocab, tgt_vocab, device, max_samples=None):
    """Evaluate model predictions on a dataset"""
    model.eval()
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_samples and batch_idx * data_loader.batch_size >= max_samples:
                break
            
            src = batch['src'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            
            # Generate predictions (simple greedy decoding)
            batch_size = src.size(1)
            max_len = 100
            
            # Initialize decoder input
            decoder_input = torch.full((1, batch_size), tgt_vocab.SOS_IDX, dtype=torch.long, device=device)
            
            for step in range(max_len):
                # Create masks
                tgt_mask = model.generate_square_subsequent_mask(decoder_input.size(0)).to(device)
                src_key_padding_mask = model.create_padding_mask(src, src_vocab.PAD_IDX)
                tgt_key_padding_mask = model.create_padding_mask(decoder_input, tgt_vocab.PAD_IDX)
                
                # Forward pass
                output = model(
                    src=src,
                    tgt=decoder_input,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
                
                # Get next token
                next_token = torch.argmax(output[-1, :, :], dim=-1, keepdim=True)
                decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=0)
                
                # Check for EOS
                if torch.all(next_token == tgt_vocab.EOS_IDX):
                    break
            
            # Convert to text
            for i in range(batch_size):
                # Hypothesis
                hyp_tokens = decoder_input[1:, i].cpu().tolist()  # Skip SOS token
                hyp_text = tgt_vocab.indices_to_sentence(hyp_tokens)
                hypotheses.append(hyp_text)
                
                # Reference
                ref_tokens = tgt_output[:, i].cpu().tolist()
                ref_text = tgt_vocab.indices_to_sentence(ref_tokens)
                references.append(ref_text)
    
    # Evaluate
    evaluator = TranslationEvaluator()
    results = evaluator.evaluate(references, hypotheses)
    
    return results, references, hypotheses


class T5NMTEvaluator:
    """T5 Neural Machine Translation Evaluator for test dataset evaluation"""
    
    def __init__(self, model_path: str, config, device: str = 'auto'):
        """Initialize the evaluator"""
        self.device = self._setup_device(device)
        self.config = config
        
        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.PRETRAINED_MODEL_PATH)
        
        # Load SPM tokenizers if available
        save_dir = getattr(config, 'SAVE_DIR', 'save_dir')
        zh_spm_path = os.path.join(save_dir, 'zh_spm_model.model')
        en_spm_path = os.path.join(save_dir, 'en_spm_model.model')
        
        if os.path.exists(zh_spm_path):
            self.zh_spm = SPMTokenizer(zh_spm_path)
            print(f"Chinese SPM model loaded: {zh_spm_path}")
        else:
            self.zh_spm = None
            print("Warning: Chinese SPM model not found, using T5 tokenizer directly")
        
        if os.path.exists(en_spm_path):
            self.en_spm = SPMTokenizer(en_spm_path)
            print(f"English SPM model loaded: {en_spm_path}")
        else:
            self.en_spm = None
            print("Warning: English SPM model not found, using T5 tokenizer directly")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Initialize BLEU evaluator
        self.bleu_evaluator = BLEUEvaluator()
        
        # Loss function for perplexity calculation
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id,
            reduction='mean'
        )
        
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
        """Load trained T5 model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Check if it's a directory (pretrained model) or a checkpoint file
        if os.path.isdir(model_path):
            # Load from pretrained model directory
            print(f"Loading model from pretrained directory: {model_path}")
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            model.to(self.device)
            model.eval()
            return model
        
        # Load from checkpoint file
        print(f"Loading model from checkpoint file: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
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
        
        # Handle T5's shared weights
        model_state_dict = model.state_dict()
        if 'encoder.embed_tokens.weight' not in state_dict and 'encoder.embed_tokens.weight' in model_state_dict:
            state_dict['encoder.embed_tokens.weight'] = model_state_dict['encoder.embed_tokens.weight']
        if 'decoder.embed_tokens.weight' not in state_dict and 'decoder.embed_tokens.weight' in model_state_dict:
            state_dict['decoder.embed_tokens.weight'] = model_state_dict['decoder.embed_tokens.weight']
        if 'lm_head.weight' not in state_dict and 'lm_head.weight' in model_state_dict:
            state_dict['lm_head.weight'] = model_state_dict['lm_head.weight']
        
        # Remove unexpected keys
        model_keys = set(model_state_dict.keys())
        state_dict_keys = set(state_dict.keys())
        unexpected_keys = state_dict_keys - model_keys
        
        if unexpected_keys:
            print(f"Warning: Removing {len(unexpected_keys)} unexpected keys from checkpoint")
            state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        
        # Load state dict
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Warning: {len(missing_keys)} keys were missing")
        except Exception as e:
            raise ValueError(f"Could not load model state dict from checkpoint. Error: {e}")
        
        model.to(self.device)
        model.eval()
        
        # Print checkpoint info if available
        if isinstance(checkpoint, dict):
            print(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")
            print(f"Best BLEU score: {checkpoint.get('bleu_score', 'N/A')}")
        
        return model
    
    def evaluate_on_test_set(self, data_dir: str = None) -> dict:
        """Evaluate model on test dataset"""
        print("=" * 60)
        print("Evaluating on Test Dataset")
        print("=" * 60)
        
        # Load test data using get_data from src.utils
        data_dir = data_dir or self.config.DATA_DIR
        save_dir = getattr(self.config, 'SAVE_DIR', 'save_dir')
        zh_spm_path = os.path.join(save_dir, 'zh_spm_model.model')
        en_spm_path = os.path.join(save_dir, 'en_spm_model.model')
        
        (src_train_sents, tgt_train_sents,
         src_valid_sents, tgt_valid_sents,
         src_test_sents, tgt_test_sents) = get_data(
            data_dir,
            zh_spm_model_path=zh_spm_path,
            en_spm_model_path=en_spm_path
        )
        
        print(f"Test dataset size: {len(src_test_sents)}")
        
        # Convert SPM token lists to text strings
        # zh_tokens and en_tokens are already SPM token lists (from read_corpus)
        test_sentences = []
        reference_sentences = []
        
        for zh_tokens, en_tokens in zip(src_test_sents, tgt_test_sents):
            # Chinese: join SPM tokens and add prefix
            zh_text = " ".join(zh_tokens)
            input_text = f"translate Chinese to English: {zh_text}"
            test_sentences.append(input_text)
            
            # English: join SPM tokens for reference
            en_text = " ".join(en_tokens)
            reference_sentences.append(en_text)
        
        # Evaluation
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_references = []
        
        max_length = getattr(self.config, 'MAX_TARGET_LENGTH', 128)
        num_beams = getattr(self.config, 'num_beams', 4)
        max_input_length = getattr(self.config, 'MAX_INPUT_LENGTH', 512)
        
        with torch.no_grad():
            pbar = tqdm(zip(test_sentences, reference_sentences), total=len(test_sentences), desc='Evaluating')
            
            for input_text, target_text in pbar:
                # Encode input (no padding needed for single sentence)
                input_ids = self.tokenizer.encode(
                    input_text,
                    return_tensors='pt',
                    max_length=max_input_length,
                    truncation=True
                ).to(self.device)
                
                # Encode target for loss calculation (no padding needed)
                target_ids = self.tokenizer.encode(
                    target_text,
                    return_tensors='pt',
                    max_length=max_length,
                    truncation=True
                ).to(self.device)
                
                # Calculate loss for perplexity
                # T5 model expects labels to have the same length as decoder input
                # The model will handle padding internally
                outputs = self.model(input_ids=input_ids, labels=target_ids)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Generate translation
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    do_sample=False
                )
                
                # Decode predictions
                translated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                all_predictions.append(translated_text)
                all_references.append([target_text])  # Wrap in list for BLEU evaluator
        
        # Calculate average loss and perplexity
        avg_loss = total_loss / len(test_sentences)
        perplexity = np.exp(avg_loss)
        
        # Calculate BLEU scores
        bleu_1 = self.bleu_evaluator.corpus_bleu(all_references, all_predictions, n_gram=1)
        bleu_2 = self.bleu_evaluator.corpus_bleu(all_references, all_predictions, n_gram=2)
        bleu_3 = self.bleu_evaluator.corpus_bleu(all_references, all_predictions, n_gram=3)
        bleu_4 = self.bleu_evaluator.corpus_bleu(all_references, all_predictions, n_gram=4)
        
        results = {
            'num_sentences': len(all_predictions),
            'loss': avg_loss,
            'perplexity': perplexity,
            'bleu_1': bleu_1,
            'bleu_2': bleu_2,
            'bleu_3': bleu_3,
            'bleu_4': bleu_4
        }
        
        # Print results
        print("\n" + "=" * 60)
        print("Evaluation Results on Test Dataset")
        print("=" * 60)
        print(f"Number of sentences: {results['num_sentences']}")
        print(f"Loss: {results['loss']:.4f}")
        print(f"Perplexity: {results['perplexity']:.4f}")
        print(f"BLEU-1: {results['bleu_1']:.4f}")
        print(f"BLEU-2: {results['bleu_2']:.4f}")
        print(f"BLEU-3: {results['bleu_3']:.4f}")
        print(f"BLEU-4: {results['bleu_4']:.4f}")
        print("=" * 60)
        
        return results


@hydra.main(version_base='1.3', config_path='./configs', config_name='inference.yaml')
def main(cfgs: DictConfig) -> Optional[float]:
    """Main evaluation function - evaluates T5 model on test dataset"""
    try:
        # Check if model_path is provided
        if not hasattr(cfgs, 'model_path') or cfgs.model_path is None:
            # If no model_path, run test mode
            print("=" * 60)
            print("Running BLEU Evaluator Test Mode")
            print("=" * 60)
            
            evaluator = BLEUEvaluator()
            
            # Test examples
            references = [["the cat is on the mat"], ["there is a cat on the mat"]]
            candidates = ["the cat is on the mat", "a cat is on the mat"]
            
            # Test sentence BLEU
            score1 = evaluator.sentence_bleu(references[0], candidates[0])
            print(f"Sentence BLEU: {score1:.4f}")
            
            # Test corpus BLEU
            corpus_score = evaluator.corpus_bleu(references, candidates)
            print(f"Corpus BLEU: {corpus_score:.4f}")
            
            # Test comprehensive evaluation
            comp_evaluator = TranslationEvaluator()
            ref_texts = ["the cat is on the mat", "there is a cat on the mat"]
            hyp_texts = ["the cat is on the mat", "a cat is on the mat"]
            
            results = comp_evaluator.evaluate(ref_texts, hyp_texts)
            print("Evaluation results:")
            for metric, score in results.items():
                print(f"  {metric}: {score:.4f}")
            
            return None
        else:
            # Evaluate on test dataset
            evaluator = T5NMTEvaluator(cfgs.model_path, cfgs, cfgs.device)
            data_dir = getattr(cfgs, 'DATA_DIR', None)
            results = evaluator.evaluate_on_test_set(data_dir=data_dir)
            
            return results['bleu_4']  # Return BLEU-4 as main metric
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the model path and vocabulary files are correct.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main() 
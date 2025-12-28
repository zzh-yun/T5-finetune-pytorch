"""
Training script for T5 Chinese-English translation
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.cuda.amp as amp
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import hydra
from omegaconf import DictConfig
from typing import Optional

from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.dataset import DataPreprocessor, create_data_loaders
from evaluation import BLEUEvaluator

# import debugpy
# debugpy.listen(17171)
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached")

class T5NMTTrainer:
    """T5 Neural Machine Translation Trainer"""
    
    def __init__(self, cfg):
        self.config = cfg
        # Initialize distributed training variables
        # Priority: environment variables > config file
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', getattr(cfg, 'LOCAL_RANK', 0)))
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging first (before distributed init, so we can log)
        self.setup_logging()
        
        # Initialize distributed training
        if self.config.DISTRIBUTED:
            self.setup_distributed()
        
        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.PRETRAINED_MODEL_PATH)
        
        # Load data
        self.setup_data()
        
        # Create model
        self.setup_model()
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Setup evaluator
        self.evaluator = BLEUEvaluator()
        
        # Training state
        self.global_step = 0
        self.best_bleu = 0.0
        self.train_losses = []
        self.val_losses = []
        self.bleu_scores = []
        self.bleu_1_scores = []
        self.bleu_2_scores = []
        self.bleu_3_scores = []
        self.learning_rates = []
        
        # Early stopping
        self.patience = getattr(self.config, 'EARLY_STOPPING_PATIENCE', 5)
        self.min_delta = getattr(self.config, 'EARLY_STOPPING_MIN_DELTA', 0.0)
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        
        # Mixed precision scaler
        self.scaler = amp.GradScaler()
        
        # Initialize start_epoch
        self.start_epoch = 1
        
        # Load checkpoint if specified
        resume_from_checkpoint = getattr(self.config, 'resume_from_checkpoint', None)
        if resume_from_checkpoint is not None:
            self.load_checkpoint(resume_from_checkpoint)
    
    def setup_distributed(self):
        """Setup distributed training"""
        # Get environment variables set by torchrun or torch.distributed.launch
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize process group
        # When using torchrun, environment variables are already set up
        # We can use the default init_method from environment
        if 'MASTER_ADDR' in os.environ:
            # torchrun sets up the environment, use default init
            dist.init_process_group(backend='nccl')
        else:
            # Fallback for manual setup
            master_addr = os.environ.get('MASTER_ADDR', 'localhost')
            master_port = os.environ.get('MASTER_PORT', '12355')
            dist.init_process_group(
                backend='nccl',
                init_method=f'tcp://{master_addr}:{master_port}',
                rank=self.rank,
                world_size=self.world_size
            )
        
        # Set device for this process
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        # Synchronize all processes
        dist.barrier()
        
        # Log distributed training info (logger should be initialized by now)
        if self.local_rank == 0:
            if hasattr(self, 'logger') and self.logger:
                self.log(f"Distributed training initialized: {self.world_size} GPUs")
            else:
                print(f"Distributed training initialized: {self.world_size} GPUs")
    
    def cleanup_distributed(self):
        """Cleanup distributed training"""
        if self.config.DISTRIBUTED:
            dist.destroy_process_group()
    
    def setup_logging(self):
        """Setup logging"""
        if self.local_rank == 0:
            # Create save directory for all training outputs
            save_dir = getattr(self.config, 'SAVE_DIR', 'outputs')
            os.makedirs(save_dir, exist_ok=True)
            
            # Setup file handler for train.log in save_dir
            log_file = os.path.join(save_dir, 'train.log')
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
    
    def log(self, message):
        """Log message only from rank 0"""
        if self.logger:
            self.logger.info(message)
    
    def setup_data(self):
        """Setup data loaders"""
        self.log("Setting up data...")
        
        # Data preprocessing
        self.preprocessor = DataPreprocessor(self.config, self.tokenizer)
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = self.preprocessor.create_datasets()
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            distributed=self.config.DISTRIBUTED
        )
        
        self.log(f"Training batches: {len(self.train_loader)}")
        self.log(f"Validation batches: {len(self.val_loader)}")
        self.log(f"Test batches: {len(self.test_loader)}")
    
    def setup_model(self):
        """Setup model"""
        self.log("Setting up model...")
        
        # Load pretrained T5 model
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.PRETRAINED_MODEL_PATH)
        self.model.to(self.device)
        
        # Setup distributed model
        if self.config.DISTRIBUTED:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        # Loss function (T5 uses CrossEntropyLoss, ignoring padding tokens)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        self.log(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        model_params = self.model.module.parameters() if self.config.DISTRIBUTED else self.model.parameters()
        
        self.optimizer = optim.AdamW(
            model_params,
            lr=self.config.LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=getattr(self.config, 'WEIGHT_DECAY', 0.01)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            verbose=True
        )
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        model_ref = self.model.module if self.config.DISTRIBUTED else self.model
        model_ref.train()
        
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', disable=self.local_rank != 0)
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            with amp.autocast():
                outputs = model_ref(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model_ref.parameters(), 
                self.config.CLIP_GRAD
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            if self.local_rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'grad_norm': f'{grad_norm:.2f}'
                })
            
            # Log intermediate results
            if self.global_step % self.config.LOG_INTERVAL == 0 and self.local_rank == 0:
                self.log(f'Step {self.global_step}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate model"""
        model_ref = self.model.module if self.config.DISTRIBUTED else self.model
        model_ref.eval()
        
        total_loss = 0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', disable=self.local_rank != 0)
            
            for batch in pbar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass for loss calculation
                outputs = model_ref(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()
                
                # Generate predictions
                generated_ids = model_ref.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=getattr(self.config, 'MAX_TARGET_LENGTH', 128),
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
                
                # Decode predictions and references
                predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                references = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                all_predictions.extend(predictions)
                all_references.extend([[r] for r in references])
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate BLEU scores
        bleu_1 = self.evaluator.corpus_bleu(all_references, all_predictions, n_gram=1)
        bleu_2 = self.evaluator.corpus_bleu(all_references, all_predictions, n_gram=2)
        bleu_3 = self.evaluator.corpus_bleu(all_references, all_predictions, n_gram=3)
        bleu_4 = self.evaluator.corpus_bleu(all_references, all_predictions, n_gram=4)
        
        return avg_loss, bleu_1, bleu_2, bleu_3, bleu_4
    
    def save_checkpoint(self, epoch, bleu_score, is_best=False):
        """Save model checkpoint"""
        if self.local_rank == 0:
            # Use SAVE_DIR for training outputs
            save_dir = getattr(self.config, 'SAVE_DIR', 'outputs')
            os.makedirs(save_dir, exist_ok=True)
            
            model_state = self.model.module.state_dict() if self.config.DISTRIBUTED else self.model.state_dict()
            
            checkpoint = {
                'epoch': epoch,
                'bleu_score': bleu_score,
                'best_bleu': self.best_bleu,
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'bleu_scores': self.bleu_scores,
                'bleu_1_scores': self.bleu_1_scores,
                'bleu_2_scores': self.bleu_2_scores,
                'bleu_3_scores': self.bleu_3_scores,
                'learning_rates': self.learning_rates,
                'best_val_loss': self.best_val_loss,
                'patience_counter': self.patience_counter,
                'global_step': self.global_step,
            }
            
            filename = os.path.join(save_dir, 'checkpoint_last.pth')
            torch.save(checkpoint, filename)
            
            if is_best:
                best_filename = os.path.join(save_dir, 'checkpoint_best.pth')
                torch.save(checkpoint, best_filename)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint or pretrained model file"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.log(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Check if this is a pretrained model file (just state dict) or a training checkpoint
        is_pretrained_model = False
        if isinstance(checkpoint, dict):
            # Check if it has training metadata (indicates it's a training checkpoint)
            has_training_metadata = any(key in checkpoint for key in ['epoch', 'optimizer_state_dict', 'train_losses', 'bleu_score'])
            
            if not has_training_metadata:
                # Check if it looks like a state dict (all keys contain '.')
                if all(isinstance(k, str) and '.' in k for k in checkpoint.keys()):
                    is_pretrained_model = True
                    self.log("Detected pretrained model file. Loading model weights only (no training state).")
        else:
            raise ValueError(
                f"Checkpoint file {checkpoint_path} is not a valid format. "
                f"Expected a dictionary containing model weights or training checkpoint."
            )
        
        # Get model state dict
        if is_pretrained_model:
            # Pretrained model file: checkpoint is the state dict itself
            model_state = checkpoint
        elif 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
        else:
            raise ValueError(
                f"Checkpoint file {checkpoint_path} does not contain 'model_state_dict' or 'state_dict'. "
                f"Available keys: {list(checkpoint.keys())[:10]}"
            )
        
        # Load model state
        try:
            if self.config.DISTRIBUTED:
                missing, unexpected = self.model.module.load_state_dict(model_state, strict=False)
            else:
                missing, unexpected = self.model.load_state_dict(model_state, strict=False)
            
            if missing:
                self.log(f"Warning: {len(missing)} keys missing (using default values)")
            if unexpected:
                self.log(f"Warning: {len(unexpected)} unexpected keys (ignored)")
            self.log("Model weights loaded successfully")
        except Exception as e:
            raise ValueError(f"Error loading model state dict: {e}")
        
        # Only load training state if this is a training checkpoint (not pretrained model)
        if not is_pretrained_model:
            # Load optimizer and scheduler (optional, might not exist in all checkpoints)
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.log("Optimizer state loaded")
                except Exception as e:
                    self.log(f"Warning: Could not load optimizer state: {e}")
            else:
                self.log("No optimizer state found, starting with fresh optimizer")
            
            if 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    self.log("Scheduler state loaded")
                except Exception as e:
                    self.log(f"Warning: Could not load scheduler state: {e}")
            else:
                self.log("No scheduler state found, starting with fresh scheduler")
            
            if 'scaler_state_dict' in checkpoint:
                try:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    self.log("Scaler state loaded")
                except Exception as e:
                    self.log(f"Warning: Could not load scaler state: {e}")
            else:
                self.log("No scaler state found, starting with fresh scaler")
            
            # Load training state
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.best_bleu = checkpoint.get('best_bleu', 0.0)
            self.global_step = checkpoint.get('global_step', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.patience_counter = checkpoint.get('patience_counter', 0)
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.bleu_scores = checkpoint.get('bleu_scores', [])
            self.bleu_1_scores = checkpoint.get('bleu_1_scores', [])
            self.bleu_2_scores = checkpoint.get('bleu_2_scores', [])
            self.bleu_3_scores = checkpoint.get('bleu_3_scores', [])
            self.learning_rates = checkpoint.get('learning_rates', [])
            
            self.log(f"Training state loaded: epoch {checkpoint.get('epoch', 'N/A')}, BLEU: {checkpoint.get('bleu_score', 'N/A')}")
        else:
            # Pretrained model: start from scratch
            self.log("Starting fine-tuning from pretrained model (epoch 1, no previous training state)")
            self.start_epoch = 1
    
    def train(self):
        """Main training loop"""
        if self.start_epoch > 1:
            self.log(f"Resuming training from epoch {self.start_epoch}...")
            self.log(f"Best BLEU so far: {self.best_bleu:.4f}")
        else:
            self.log("Starting training from scratch...")
        
        for epoch in range(self.start_epoch, self.config.NUM_EPOCHS + 1):
            if self.config.DISTRIBUTED:
                self.train_loader.sampler.set_epoch(epoch)
            
            train_loss = self.train_epoch(epoch)
            val_loss, bleu_1, bleu_2, bleu_3, bleu_4 = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.bleu_scores.append(bleu_4)
            self.bleu_1_scores.append(bleu_1)
            self.bleu_2_scores.append(bleu_2)
            self.bleu_3_scores.append(bleu_3)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            if self.local_rank == 0:
                # Calculate perplexity from loss
                perplexity = np.exp(train_loss)
                learning_rate = self.optimizer.param_groups[0]['lr']
                # Log format: Epoch, Step, Loss, Perplexity, Learning_Rate, BLEU-1, BLEU-2, BLEU-3, BLEU-4
                self.log(f'Epoch {epoch}, Step {self.global_step}, Loss: {train_loss:.4f}, Perplexity: {perplexity:.4f}, Learning_Rate: {learning_rate:.2e}, BLEU-1: {bleu_1:.4f}, BLEU-2: {bleu_2:.4f}, BLEU-3: {bleu_3:.4f}, BLEU-4: {bleu_4:.4f}')
                
                is_best = bleu_4 > self.best_bleu
                if is_best:
                    self.best_bleu = bleu_4
                
                self.save_checkpoint(epoch, bleu_4, is_best)
            
            # Update scheduler
            self.scheduler.step(bleu_4)
            
            # Early stopping check
            if self.local_rank == 0:
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.patience:
                    self.log(f"\nEarly stopping triggered: validation loss did not improve for {self.patience} epochs")
                    self.log(f"Best validation loss: {self.best_val_loss:.4f}")
                    self.log(f"Current validation loss: {val_loss:.4f}")
                    break
        
        self.log("Training finished.")
        
        # Cleanup distributed training
        if self.config.DISTRIBUTED:
            self.cleanup_distributed()


@hydra.main(version_base='1.3', config_path='./configs', config_name='train.yaml')
def main(cfgs: DictConfig) -> Optional[float]:
    # Check if distributed training is enabled via environment variables
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Update config with distributed training settings
    # Priority: environment variables > config file
    from omegaconf import OmegaConf
    
    # Temporarily disable struct mode to allow updates
    was_struct = OmegaConf.is_struct(cfgs)
    if was_struct:
        OmegaConf.set_struct(cfgs, False)
    
    # Update existing config fields
    if world_size > 1:
        cfgs.LOCAL_RANK = local_rank
        cfgs.DISTRIBUTED = True
    else:
        # Use config file values or defaults
        cfgs.DISTRIBUTED = getattr(cfgs, 'distributed', False) or getattr(cfgs, 'DISTRIBUTED', False)
        cfgs.LOCAL_RANK = getattr(cfgs, 'local_rank', 0) or getattr(cfgs, 'LOCAL_RANK', 0)
    
    # Restore struct mode if it was enabled
    if was_struct:
        OmegaConf.set_struct(cfgs, True)
    
    # Adjust batch size for distributed training
    # In distributed training, each GPU processes batch_size samples
    # Total effective batch size = batch_size * num_gpus
    if cfgs.DISTRIBUTED and world_size > 1:
        original_batch_size = cfgs.BATCH_SIZE
        # Keep per-GPU batch size the same (don't divide)
        # This means effective batch size will be batch_size * num_gpus
        if local_rank == 0:
            print(f"Distributed training: {world_size} GPUs, "
                  f"per-GPU batch size: {cfgs.BATCH_SIZE}, "
                  f"effective batch size: {cfgs.BATCH_SIZE * world_size}")
    
    trainer = T5NMTTrainer(cfgs)
    trainer.train()


if __name__ == '__main__':
    main()

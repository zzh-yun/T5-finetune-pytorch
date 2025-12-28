# T5 PyTorch Chinese-English Translation

## Introduction
This project fine-tunes the T5 pre-trained model to implement neural machine translation from Chinese to English. T5 (Text-To-Text Transfer Transformer) is a unified text-to-text pre-trained model proposed by Google. By unifying various NLP tasks as text generation tasks, it has achieved excellent performance across multiple tasks. In this project, we use the T5-base model and fine-tune it on Chinese-English parallel corpora to achieve high-quality Chinese-to-English translation.

Due to the model chechpoint is too big to upload to github, so I put it on huggingface: https://huggingface.co/DarcyCheng/T5-finetune-model

And if you want to download the pretrained model of T5, please go to the url: 
https://huggingface.co/google-t5/t5-base/tree/main

## Data Preparation
The data compression package contains four JSONL files, corresponding to the small-scale training set, large-scale training set, validation set, and test set, with 100k, 10k, 500, and 200 samples respectively. Each line in each JSONL file contains a parallel sentence pair. The final model performance will be evaluated based on the results on the test set.

Data file structure:
- `train_100k.jsonl`: Small-scale training set (100,000 entries)
- `train_10k.jsonl`: Large-scale training set (10,000 entries)
- `val.jsonl`: Validation set (500 entries)
- `test.jsonl`: Test set (200 entries)

Format of each JSONL file:
```json
{"zh": "Chinese sentence", "en": "English sentence"}
```

## Environment

### System Requirements
- Python: Python 3.9.25
- PyTorch: 2.0.1+cu118
- CUDA: 11.8 (if using GPU)

### Dependency Installation

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `torch>=2.0.1`
- `transformers`
- `hydra-core`
- `omegaconf`
- `nltk`
- `jieba`
- `sentencepiece`
- `numpy`
- `tqdm`

## Training, Evaluation and Inference

### 1. Train SPM Model (Optional)
Before training the T5 model, you can first train the SentencePiece (SPM) model to achieve better tokenization performance:

```bash
conda activate llm_course
python train_spm_models.py
```

### 2. Train T5 Model
Train the T5 model for Chinese-to-English translation (SPM is used automatically):

```bash
python train.py
```

Training configurations are stored in `configs/train.yaml`, with key parameters including:
- `NUM_EPOCHS`: Number of training epochs (default: 50)
- `BATCH_SIZE`: Batch size (default: 64)
- `LEARNING_RATE`: Learning rate (default: 1e-5)
- `DATA_DIR`: Path to the data directory

### 3. Evaluate the Model
Evaluate the model performance on the test set:

```bash
python evaluation.py
```

The evaluation script calculates BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores (in percentage, ranging from 0 to 100).

### 4. Inference

#### Single Sentence Translation

```bash
python inference.py input_text="Hello"
```

#### Batch Translation

```bash
python inference.py input_file=test.txt output_file=translations.txt
```

#### Interactive Translation

```bash
python inference.py interactive=True
```

#### Evaluation with Reference Translations

```bash
python inference.py input_file=test.txt reference_file=reference.txt
```

Inference configurations are stored in `configs/inference.yaml`, which can be set as follows:
- `model_path`: Path to the trained model checkpoint
- `max_length`: Maximum generation length
- `num_beams`: Beam search size

## Acknowledgement
We would like to acknowledge the following repositories:

- [Hugging Face T5-base Model](https://huggingface.co/google-t5/t5-base/tree/main)
- [Google Research T5 Repository](https://github.com/google-research/text-to-text-transfer-transformer)

### Translation Notes:
1. **Technical Terminology Consistency**: Adhered to standard English terms in NLP (e.g., "fine-tune" for 微调, "parallel corpora" for 平行语料, "tokenization" for 分词, "checkpoint" for 检查点) to align with industry conventions.
2. **Readability & Naturalness**: Adjusted sentence structures to fit English technical documentation style (e.g., splitting long Chinese sentences into concise English clauses, using active voice where appropriate).
3. **Code/File Preservation**: All command-line scripts, file paths, parameter names, and code snippets are retained exactly as original to ensure technical accuracy and usability.
4. **Cultural/Contextual Adaptation**: "你好" was translated to "Hello" in the single sentence translation example (instead of literal "Ni Hao") for natural English usage; "训练轮数" → "number of training epochs" (standard ML terminology).
5. **Format Consistency**: Maintained the original markdown structure (headings, lists, code blocks) to preserve readability and hierarchy of the technical document.
# T5 PyTorch Chinese-English Translation

. 工作流程
训练SPM模型：
   conda activate llm_course
   python train_spm_models.py
   conda activate llm_course   python train_spm_models.py
训练T5模型（自动使用SPM）：
   python train.py
   python train.py
推理（自动使用SPM）：
   python inference.py input_text="你好"
   python inference.py input_text="你好"

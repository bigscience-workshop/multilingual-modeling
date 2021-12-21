### Previous Experiments 
- `exp-001`: train gpt-2's tokenizer and finetune gpt-2's embedding layers `wte` and `wpe` on HF's OSCAR `unshuffled_deduplicated_fr` and `unshuffled_dudplicated_kr`.
- `exp-002`: evaluate gpt-2 on FLUE's tasks (CLS, XNLI, PAWS)
- `exp-003`: TODO: evaluate on multiatis 
- `exp-004`: Does the embedding layer learn anything useful? Take a dataset in English for PAWS-X, finetune GPT-2 on this dataset, evaluate it on English test set T_e. Then, take the same test-set T_e translated in French (T_f), take GPT-2 parameters fine-tuned for the task X,  replace English embeddings with French embeddings and evaluate thus obtained model on French test set.

# Experiment folders below after Conversation with Vassilina, Hady, Iz, and Maruf [Link](https://huggingface.slack.com/archives/C020G6A9KHQ/p1637023149074800) 
- `exp-005`: cleaned from `exp-001` for finetuning GPT-2 embedding layers for DE and KO on Oscar.
- `exp-006`: run zero-shot and finetuned evaluation setting for XNLI ✅, PAWS ❌, and XQuAD ❌. (❌ means not done. ✅ means done.)
- `exp-007`: apply MAD-X adapter method. [Paper link](https://arxiv.org/abs/2005.00052)
- `exp-008`: from exp-006, but using mBERT on the zero-shot and finetuning setting.


# Carbon Tracking 
Do not forget to log your experiments [in this spreadsheet](https://docs.google.com/spreadsheets/d/1Mk8mYCOF_WxMv-Uv5ThkFs5Ak5B9s9EnRUh1CpykEJ0/edit#gid=0)


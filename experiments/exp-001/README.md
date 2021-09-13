# Decisions

**Dataset**: HF's OSCAR unshuffled_deduplicated_fr 

**Tokenizer**: byte-level Byte-pair encoding tokenizer (same as GPT-2). Training is identical to the section "Using an existing tokenizer" in huggingface's tokenizer_training [tutorial](https://github.com/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb)
- train the GPT-2 tokenizer with the exact same algorithms and parameters as an existing one.
- vocab_size: 50,257 (same as original GPT-2)

**Model Finetuning**: Use [HF's code](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling) `run_clm.py` to finetune GPT-2 with the new tokenizer on oscar's fr dataset.

Freeze all the layers of GPT-2 except the word embedding layer `wte` and the positional embedding layer `wpe` by adding the following snippet of codes to `run_clm.py`.
```    
...
for name, param in model.named_parameters():
    if name not in ('transformer.wte.weight', 'transformer.wpe.weight'):
        print(f"🥶 Freeze layer '{name}'")
        param.requires_grad = False
    else:
        param.requires_grad = True
...
```
 

import tiktoken
import torch

class TikTokenizer:
    
   def __init__(self, encoding_name="cl100k_base", pad_token_id=0):
       self.enc = tiktoken.get_encoding(encoding_name)
       self.pad_token_id = pad_token_id
       
   def __call__(self, texts, padding=True, truncation=True, max_length=1024, return_tensors="pt", pad_direction="right"):
       if isinstance(texts, str):
           texts = [texts]
       
       token_ids = self.enc.encode_batch(texts, disallowed_special=())
       
       if truncation:
           token_ids = [ids[:max_length] for ids in token_ids]
        
       if padding:
           # always pad to max_length to ensure consistent tensor sizes
           if pad_direction == "right":
               token_ids = [ids + [self.pad_token_id] * (max_length - len(ids)) for ids in token_ids]
           elif pad_direction == "left":
               token_ids = [[self.pad_token_id] * (max_length - len(ids)) + ids for ids in token_ids]
           
       # create attention masks
       attention_masks = [[int(tid != self.pad_token_id) for tid in ids] for ids in token_ids]
       
       if return_tensors == "pt":
           return {
               "input_ids": torch.tensor(token_ids),
               "attention_mask": torch.tensor(attention_masks)
           }
       return {"input_ids": token_ids, "attention_mask": attention_masks}
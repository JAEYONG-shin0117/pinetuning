import argparse
import random
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EncoderDecoderModel, GPT2Tokenizer
from datasets import SonnetsDataset
from evaluation import test_sonnet
from optimizer import AdamW

TQDM_DISABLE = False

def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

class SonnetGPT(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        "bert-base-uncased", "gpt2"
    )

    for param in self.model.encoder.parameters():
        param.requires_grad = False

    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
    self.model.config.pad_token_id = self.tokenizer.pad_token_id

  def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels):
    outputs = self.model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      decoder_input_ids=decoder_input_ids,
      decoder_attention_mask=decoder_attention_mask,
      labels=labels
    )
    return outputs.loss, outputs.logits

  def get_device(self):
    return next(self.model.parameters()).device

  @torch.no_grad()
  def generate(self, input_ids, attention_mask, **kwargs):
    return self.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **kwargs
    )

def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }
  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")

def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  model = SonnetGPT(args).to(device)
  optimizer = AdamW(model.parameters(), lr=args.lr)

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids = batch['token_ids'].to(device)
      b_mask = batch['attention_mask'].to(device)

      input_ids = b_ids[:, :-1]
      attention_mask = b_mask[:, :-1]
      decoder_input_ids = b_ids[:, 1:]
      decoder_attention_mask = b_mask[:, 1:]
      labels = b_ids[:, 1:]

      optimizer.zero_grad()
      loss, _ = model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          decoder_input_ids=decoder_input_ids,
          decoder_attention_mask=decoder_attention_mask,
          labels=labels
      )
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss /= num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss:.3f}")
    print("Generating several output sonnets...")
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      generated_ids = model.generate(
          input_ids=encoding['input_ids'],
          attention_mask=encoding['attention_mask'],
          max_length=128,
          temperature=args.temperature,
          top_p=args.top_p,
          do_sample=True
      )
      decoded = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
      print(f"{batch[1]}\n{decoded}\n\n")

    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')

@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    generated_ids = model.generate(
        input_ids=encoding['input_ids'],
        attention_mask=encoding['attention_mask'],
        max_length=128,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True
    )
    decoded_output = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))
    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+", encoding="utf-8") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])

  chrf_score = test_sonnet(args.sonnet_out, args.true_sonnet_path)
  print(f"\n생성된 소넷의 CHRF 점수: {chrf_score:.3f}")

def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out_dev.txt")
  parser.add_argument("--true_sonnet_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--temperature", type=float, default=1.2)
  parser.add_argument("--top_p", type=float, default=0.9)
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--lr", type=float, default=1e-5)

  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'
  seed_everything(args.seed)
  train(args)
  generate_submission_sonnets(args)

import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from torch.cuda.amp import autocast
import bitsandbytes as bnb
import wandb

import ruclip
import clip, open_clip
import random
from transformers import GPT2Config, GPT2Model
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup


from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor, AdafactorSchedule

import os
import pickle
import sys
import argparse

from typing import Tuple, Optional, Union
from torch.cuda.amp import autocast
import nltk
from datasets import load_dataset, load_metric
from nltk.translate.bleu_score import corpus_bleu

manualSeed = 1337
#manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
    
    @autocast()  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention
    
class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

def freeze(
    model,
    freeze_emb=False,
    freeze_ln=False,
    freeze_attn=True,
    freeze_ff=True,
    freeze_other=False,
):
    
    for name, p in model.named_parameters():
    # freeze all parameters except the layernorm and positional embeddings
        name = name.lower()
        if 'ln' in name or 'norm' in name:
            p.requires_grad = not freeze_ln
        elif 'embeddings' in name:
            p.requires_grad = not freeze_emb
        elif 'mlp' in name:
            p.requires_grad = not freeze_ff
        elif 'attn' in name:
            p.requires_grad = not freeze_attn
        else:
            p.requires_grad = not freeze_other
           
    return model


from enum import Enum
class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


gpt_model_name = 'sberbank-ai/rugpt3medium_based_on_gpt2'
class ClipCaptionModel(nn.Module):
    def __init__(
        self,
        prefix_length: int,
        clip_length: Optional[int] = 10,
        prefix_size: int = 640,
        num_layers: int = 8,
        mapping_type: MappingType = MappingType.Transformer
    ):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length

        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_model_name)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        if mapping_type == MappingType.MLP:
            print("using mlp")
            self.clip_project = MLP((
                prefix_size,
                self.gpt_embedding_size * prefix_length // 2,
                self.gpt_embedding_size * prefix_length
            ))
        else:
            print("using transformer")
            print(clip_length)
            self.clip_project = TransformerMapper(
                prefix_size,
                self.gpt_embedding_size,
                prefix_length,
                clip_length, 
                num_layers
            )

        
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)
    
    @autocast() 
    def forward(
        self,        
        tokens: torch.Tensor,
        prefix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(
            prefix.float()
        ).view(-1, self.prefix_length, self.gpt_embedding_size)

        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        
        return out
    

class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self
    

class ClipCocoDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        prefix_length=30,
        model_type = gpt_model_name,
        normalize_prefix=False,
        train=True,
    ):

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        if train:
            with open(data_path, 'rb') as f:
                all_data = pickle.load(f)
            print("Data size is %0d" % len(all_data["clip_embedding"]))
        else:
            with open(data_path, 'rb') as f:
                all_data = pickle.load(f)
            print("Data size is %0d" % len(all_data["clip_embedding"]))

        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        
        self.captions = captions_raw

        self.captions_tokens = []
        self.caption2embedding = []
        max_seq_len = 0
        i = 0
        for caption in tqdm(captions_raw):
            self.captions_tokens.append(
                torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64)
            )
            self.caption2embedding.append(self.prefixes[i])
            i += 1
            max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])

        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            #self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            #self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask
    
    def __len__(self) -> int:
        return len(self.captions_tokens)

    def __getitem__(self, item):
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[item]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix
    

def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser

wandb.login()

wandb.init(project="ClipCap_NAS", name="TRANSFORMER&GPT-2-BLEU")

def calc_bleu(y_pred, y_true):
    references = [[reference.split()] for reference in y_true]
    hypotheses = [hypothesis.split() for hypothesis in y_pred]
    # Рассчитываем BLEU-4
    bleu_score = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu_score*100

def train(
    train_dataset: ClipCocoDataset,
    train_dataloader,
    #valid_dataset: ClipCocoDataset,
    model: ClipCaptionModel,
    optimizer,
    scheduler,
    args,
    warmup_steps: int = 5000,
    output_dir: str = ".",
    output_prefix: str = ""
):
    #
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = freeze(model)


    model.train()

    #optimizer = Adafactor(
    #    model.parameters(),
    #    lr=args.lr,
    #    relative_step=False, # for adafactor
    #)
    

    
    #valid_dataloader = DataLoader(
    #    valid_dataset,
    #    batch_size=batch_size,
    #    shuffle=False,
    #    drop_last=False,
    #)
    
    
    #scheduler = AdafactorSchedule(optimizer) работает не оч

    mean_epoch_train_loss = []
    mean_bleu_train_epoch = []
    
    
    for epoch in range(epochs):
        loss_train_epoch = []
        bleu_train_epoch = []
        print(f">>> Training epoch {epoch+1}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        step=0
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            step += 1
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.bfloat16)
            
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, train_dataset.prefix_length - 1: -1]

            loss = nnf.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tokens.flatten().to(torch.int64),
                ignore_index=0
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()
            progress.set_postfix({"loss_train": loss.item()})
            

            loss_train_epoch.append(loss.item())

            if step % 500 == 0:
                wandb.log({"loss_train":  loss.item()})
            
            if step % 1000 == 0:
                with torch.no_grad():
                    # BLEU-4
                    logits_cpu = logits.cpu()
                    tokens_cpu = tokens.cpu()
                    generated_texts = []
                    real_text = []
                    for b in range(batch_size):
                        generated_text_batch = train_dataset.tokenizer.decode(logits_cpu[b].argmax(dim=-1).tolist())
                        first_dot_index = generated_text_batch.find('.')
                        if first_dot_index != -1:
                            generated_texts.append(generated_text_batch[35:first_dot_index + 1])
                        else:
                            generated_texts.append(generated_text_batch[35:])
                        
                        real_text_batch = train_dataset.tokenizer.decode(tokens_cpu[b].tolist())
                        first_pad_index = real_text_batch.find('<pad>')
                        if first_pad_index != -1:
                            real_text.append(real_text_batch[35:first_pad_index])
                        else:
                            real_text.append(real_text_batch[35:])
                    
                    bleu = calc_bleu(generated_texts, real_text)
                    wandb.log({"bleu-4 train":  bleu})
                    bleu_train_epoch.append(bleu)
                    

            progress.update()
            if (idx + 1) % 7000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest_gpt2_medium.pt"),
                )
        progress.close()
        if epoch % args.save_every == 0:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{(epoch+1):03d}_gpt2_medium.pt"),
            )
        mean_epoch_train_loss.append(np.mean(loss_train_epoch))
        mean_bleu_train_epoch.append(np.mean(bleu_train_epoch))
        
        wandb.log({"mean_epoch_train_loss": mean_epoch_train_loss[-1]})
        wandb.log({"mean_bleu_train_epoch": mean_bleu_train_epoch[-1]})
        
        
        '''
        print(f">>> Validation epoch {epoch+1}")

        progress = tqdm(total=len(valid_dataloader), desc=output_prefix)
        step_validation=0
        for idx, (tokens, mask, prefix) in enumerate(valid_dataloader):
            step_validation+=1
            with torch.no_grad():

                tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.bfloat16)
            
                outputs = model(tokens, prefix, mask)
                logits = outputs.logits[:, valid_dataset.prefix_length - 1: -1]

                loss = nnf.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    tokens.flatten().to(torch.int64),
                    ignore_index=0
                )



            progress.set_postfix({"loss_val": loss.item()})

            loss_valid_epoch.append(loss.item())
            if step % 500:
                wandb.log({"loss_val":  loss.item()})
            
            progress.update()

        mean_epoch_train_loss.append(np.mean(loss_train_epoch))
        mean_epoch_validation_loss.append(np.mean(loss_valid_epoch))
        
        wandb.log({"mean_epoch_validation_loss": mean_epoch_validation_loss[-1]})
        wandb.log({"mean_epoch_train_loss": mean_epoch_train_loss[-1]})
        
        progress.close()
        '''
        

    return model

class Args():
    def __init__(self):
        self.backbone = gpt_model_name
        self.train_data = "Features_train_coco_ru_vitb16_82783.pkl"
        self.valid_data = "Features_val_coco_ru_vitb16.pkl"
        self.out_dir = 'checkpoints'
        self.prefix = '20bs_adamw'
        self.epochs = 1
        self.save_every = 1
        self.prefix_length = 30
        self.bs = 3
        self.only_prefix = False
        self.lr = 2e-5
        self.warmup_steps = 5000
        self.clip_length=10
args = Args()

train_dataset = ClipCocoDataset(args.train_data, args.prefix_length, train=True)

wandb.config = {
  "learning_rate": args.lr,
  "epochs": args.epochs,
  "batch_size": args.bs
}



model = ClipCaptionModel(args.prefix_length)
model = model.to(device)


train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.bs,
    shuffle=True,
    drop_last=True,
)

optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
    )

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=args.epochs * len(train_dataloader)
)

print("Train both prefix and GPT2")
sys.stdout.flush()
model = train(
    train_dataset,
    train_dataloader,
    model,
    optimizer,
    scheduler,
    args,
    warmup_steps=args.warmup_steps,
    output_dir=args.out_dir,
    output_prefix=args.prefix
)

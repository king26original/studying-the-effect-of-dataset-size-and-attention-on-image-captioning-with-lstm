from torch import nn
from torchvision import models
import torch
import torch.nn.functional as F

class encoder(nn.Module):
  def __init__(self, enc_dim):
    super().__init__()

    self.resnet=models.resnet50(pretrained=True)
    in_features=self.resnet.fc.in_features

    for param in self.resnet.parameters():
      param.requires_grad_(False)

    modules=list(self.resnet.children())[:-1]
    self.resnet=nn.Sequential(*modules)

    self.linear=nn.Linear(in_features, enc_dim)
    self.bn=nn.BatchNorm1d(enc_dim, momentum=0.01)

  def forward(self, images):
    features=self.resnet(images)
    features=self.linear(features.view(features.size(0), -1))
    features=self.bn(features)
    return features
  
class decoder(nn.Module):
  def __init__(self,embed_size,hidden_size,vocab_size, num_layers=1):
    super().__init__()

    self.embed=nn.Embedding(vocab_size,embed_size)
    self.lstm=nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True) #batchfirst makes output arranged as (batch_size, seq_layers, features)
    self.linear=nn.Linear(hidden_size,vocab_size)

  def forward(self,features,captions):

    captions=captions[:,:-1] # "<start> hello my name <end> " we drop the end
    embeddings=self.embed(captions)

    input=torch.cat((features.unsqueeze(1),embeddings),1)
    output,hidden=self.lstm(input)
    output=self.linear(output)
    return output

  def beam(self, features, k, max_len=20):
        vocab_size=self.linear.out_features
        start_token=1
        end_token=2
        device=features.device
        
        k_prev_words=torch.full((k, 1), start_token, dtype=torch.long, device=device)
        top_k_scores=torch.zeros(k, 1, device=device)
        
        features=features.expand(k,-1)
        inputs=features.unsqueeze(1) 
        
        states=None
        
        output, states=self.lstm(inputs, states)
        output=self.linear(output.squeeze(1)) 
        
        log_probs=F.log_softmax(output, dim=1) 
        
        top_k_scores, top_k_words=log_probs[0].topk(k, dim=0)
        
        k_prev_words=top_k_words.unsqueeze(1) 
        top_k_scores=top_k_scores.unsqueeze(1)
        
        for i in range(1, max_len):
            inputs=self.embed(k_prev_words[:,-1]).unsqueeze(1)
            
            output, states=self.lstm(inputs, states)
            output=self.linear(output.squeeze(1))
            log_probs=F.log_softmax(output, dim=1)

            total_scores=top_k_scores+log_probs 
            
            total_scores=total_scores.view(-1)
            top_k_scores, top_k_indices=total_scores.topk(k, dim=0)
            
            prev_word_indices=top_k_indices//vocab_size 
            next_word_ids=top_k_indices%vocab_size      
            
            new_sequences=torch.cat(
                [k_prev_words[prev_word_indices], next_word_ids.unsqueeze(1)], dim=1
            )
            k_prev_words=new_sequences
            
            h, c=states
            states=(h[:, prev_word_indices, :], c[:, prev_word_indices, :])
            
            top_k_scores=top_k_scores.unsqueeze(1)
            
            if next_word_ids[0].item()==end_token:
                break

        return k_prev_words[0].tolist()
      

  def sample(self,features,max_len=20):
    output_ids=[]
    states=None

    inputs=features.unsqueeze(1)
    for i in range(max_len):
      output,states=self.lstm(inputs,states)

      output=self.linear(output.squeeze(1))
      predicted=output.argmax(1)
      output_ids.append(predicted.item())

      inputs=self.embed(predicted).unsqueeze(1)

      if predicted.item()==2:
        break
    return output_ids
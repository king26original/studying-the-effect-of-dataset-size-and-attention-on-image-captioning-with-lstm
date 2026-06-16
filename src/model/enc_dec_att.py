from torch import nn
from torchvision import models
import torch
import math

class encoder(nn.Module):
  def __init__(self, embed_size, att=False):
    super().__init__()

    self.resnet=models.resnet50(pretrained=True)
    in_features=self.resnet.fc.in_features

    for param in self.resnet.parameters():
      param.requires_grad_(False)

    modules=list(self.resnet.children())[:-2]
    self.resnet=nn.Sequential(*modules)

    self.linear=nn.Linear(in_features, embed_size)
    self.bn=nn.BatchNorm1d(embed_size, momentum=0.01)

  def forward(self, images):
    features=self.resnet(images)
    features=features.view(features.size(0), features.size(1), -1)
    features=features.permute(0, 2,1)
    b=features.size(0)
    n=features.size(1)
    features=self.linear(features)
    features=features.view(-1, features.size(-1))
    features=self.bn(features)
    features = features.view(b, n, -1)
    
    return features

class Attention(nn.Module):
  def __init__(self, encoder_dim, decoder_dim, attention_dim):
    super().__init__()
    self.Wq=nn.Linear(decoder_dim, attention_dim)
    self.Wk=nn.Linear(encoder_dim, attention_dim)
    self.Wv=nn.Linear(encoder_dim, attention_dim)

    self.dim=attention_dim

  def forward(self, enc, sprev):
    q=self.Wq(sprev)

    k=self.Wk(enc)
    v=self.Wv(enc)

    att=torch.bmm(q.unsqueeze(1), k.transpose(1,2))/math.sqrt(self.dim)
    att=torch.softmax(att, dim=2)

    context=torch.bmm(att, v).squeeze(1)
    return context, att.squeeze(1)

class decoder(nn.Module):
  def __init__(self,embed_size,hidden_size,vocab_size, encoder_dim, attention_dim):
    super().__init__()

    self.embed=nn.Embedding(vocab_size,embed_size)
    self.attention=Attention(encoder_dim, hidden_size, attention_dim)
    self.lstm=nn.LSTMCell(embed_size+attention_dim, hidden_size)
    self.linear=nn.Linear(hidden_size,vocab_size)

  def forward(self,features,captions):

    captions=captions[:,:-1] # "<start> hello my name <end> " we drop the end
    embeddings=self.embed(captions)
    batch_size=features.size(0)

    h_state=torch.zeros(batch_size, self.lstm.hidden_size).to(features.device)
    c_state=torch.zeros(batch_size, self.lstm.hidden_size).to(features.device)

    outputs=[]

    for i in range(captions.size(1)):
        context, _=self.attention(features, h_state)
        lstm_input=torch.cat((embeddings[:, i, :], context), dim=1)
        h_state, c_state=self.lstm(lstm_input, (h_state, c_state))
        outputs.append(self.linear(h_state))

    return torch.stack(outputs, dim=1)
  
  def sample(self,features,max_len=20):
    output_ids=[]
    batch_size=features.size(0)

    h_state=torch.zeros(batch_size, self.lstm.hidden_size).to(features.device)
    c_state=torch.zeros(batch_size, self.lstm.hidden_size).to(features.device)

    inputs=self.embed(torch.tensor([1]*batch_size).to(features.device))

    for i in range(max_len):
      context, _=self.attention(features, h_state)

      lstm_input=torch.cat((inputs, context), dim=1)
      h_state, c_state=self.lstm(lstm_input, (h_state, c_state))
      predicted=self.linear(h_state).argmax(1)
      output_ids.append(predicted.item())

      if predicted.item()==2:
        break

      inputs=self.embed(predicted)
    return output_ids
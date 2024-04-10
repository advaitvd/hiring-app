import torch
from torch.nn.functional import cosine_similarity, avg_pool1d
from transformers import BartTokenizer, BartModel

class FeatureExtractor():
  def __init__(self):
    self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    self.model = BartModel.from_pretrained('facebook/bart-base')

  def forward(self, sentence):
    context_length = 2048
    with torch.no_grad():
        X = []
        for idx in range(0, len(sentence), context_length):
          x = self.tokenizer(sentence[idx: idx + context_length], return_tensors='pt')
          out = self.model(**x)
          X.append(out.last_hidden_state)
    X = torch.concat(X, dim=1)
    print("Feature Shape:", X.shape)
    return X
  
  def __call__(self, sentence):
    feats = self.forward(sentence)
    feats = avg_pool1d(feats.transpose(1, 2), feats.shape[1]).reshape(1, -1)
    return feats

if __name__ == "__main__":
    feat_extractor = FeatureExtractor()
    sent1 = 'Skills Requirements: Python, C++, Java, DeepLearning'
    sent2 = 'Skills: Public Speeking, Leadership, Microsoft Excel'
    sent3 = 'Skills: Programming in an Object Oriented Programming Language, DL, ML, MySQL'

    vector1 = feat_extractor(sent1)
    vector2 = feat_extractor(sent2)
    vector3 = feat_extractor(sent3)

    print(cosine_similarity(vector1, vector2))
    print(cosine_similarity(vector1, vector3))
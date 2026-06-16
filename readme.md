# Image Captioning: An Empirical Study of Classical LSTM-Based Architectures

This repository presents an experimental study of classical image captioning architectures using LSTM-based decoders trained on the Flickr8k and Flickr30k datasets. Instead of focusing solely on achieving state-of-the-art performance, this project investigates how different design choices influence caption generation quality.

The study compares:

- A Show-and-Tell style LSTM decoder trained on Flickr8k
- The same architecture trained on Flickr30k
- An attention-augmented LSTM decoder trained on Flickr30k
- Different decoding strategies including greedy search and beam search

---

## Motivation

Modern image captioning has largely shifted toward transformer-based architectures. However, classical CNN-LSTM approaches remain important because they are:

- Easier to understand,
- Computationally cheaper,
- Historically significant,
- A good way to study the effect of architectural choices.

The goal of this project was to understand:

- Does increasing dataset size improve caption quality?
- Does adding attention improve performance?
- How much does decoding strategy matter?
- What are the limitations of classical image captioning models?

---

# Architectures

## 1. Show-and-Tell (No Attention)

Image features extracted using a pretrained ResNet50 encoder are projected into an embedding space and used as the first input token to an LSTM decoder.

```
Image
↓
ResNet50
↓
Linear Projection
↓
<Image Feature Token>
↓
LSTM Decoder
↓
Caption
```

### Training datasets

- Flickr8k
- Flickr30k

---

## 2. Attention-Augmented LSTM

An attention mechanism was incorporated into the decoder to condition word generation on visual information.

```
Image
↓
ResNet50
↓
Attention Module
↓
LSTMCell Decoder
↓
Caption
```

The attention mechanism was further investigated to understand whether it learned meaningful visual alignments.

---

# Datasets

## Flickr8k

- ~8,000 images
- 5 captions per image
- Approximately 40,000 captions

## Flickr30k

- ~31,000 images
- 5 captions per image
- Approximately 155,000 captions

---

# Experimental Setup

## Encoder

- Backbone: ResNet50 (ImageNet pretrained)
- Encoder frozen during training
- Final classification layer removed

## Decoder

### Show-and-Tell

- Word Embedding Layer
- 1-layer LSTM
- Linear vocabulary projection

### Attention Decoder

- Word Embedding Layer
- Attention mechanism
- LSTMCell
- Linear vocabulary projection

---

# Hyperparameters

| Parameter | Value |
|-----------|--------|
| Encoder | ResNet50 |
| Embedding Dimension | 256 |
| Hidden Dimension | 512 |
| Attention Dimension | 512 |
| LSTM Layers | 1 |
| Optimizer | Adam |
| Batch Size | 64 |
| Epochs (Flickr8k) | 80 |
| Epochs (Flickr30k) | 80 |
| Teacher Forcing | Yes |
| Encoder Fine-Tuning | No |

---

# Decoding Strategies

Three inference strategies were evaluated:

## Greedy Search

At each timestep, the token with maximum probability is selected.

```
word_t = argmax(P(word))
```

---

## Beam Search

Beam widths explored:

- Beam = 3
- Beam = 5
- Beam = 7

The top-k candidate sequences are maintained during generation.

---

# Evaluation Metrics

The models were evaluated using:

- METEOR
- CIDEr

These metrics were chosen because they better capture semantic similarity than BLEU alone.

---

# Results

## Flickr30k (No Attention)

| Decoding Strategy | METEOR | CIDEr |
|------------------|---------|--------|
| Greedy | 0.165 | 0.221 |
| Beam 3 | 0.163 | 0.243 |
| Beam 5 | 0.159 | 0.241 |
| Beam 7 | 0.155 | 0.237 |

### Observation

Beam search improved CIDEr compared to greedy decoding, with Beam=3 producing the strongest results.

Increasing beam width beyond 3 did not improve performance.

---

## Flickr30k (Attention)

| Decoding Strategy | METEOR | CIDEr |
|------------------|---------|--------|
| Greedy | 0.168 | 0.231 |

### Observation

The attention-based decoder achieved a slightly higher METEOR score but did not significantly improve CIDEr.

---

## Flickr8k (No Attention)

The Flickr8k model achieved higher METEOR and CIDEr scores than expected relative to Flickr30k despite being trained on a smaller dataset.

This suggests that:

- Flickr8k may be an easier benchmark,
- Larger datasets do not automatically translate to better performance,
- Model capacity and training setup may become bottlenecks on more diverse datasets.

---

# Key Findings

## 1. Beam Search Helps

Beam search improved CIDEr compared to greedy decoding.

However, larger beam widths produced diminishing returns.

---

## 2. More Data Is Not Always Better

Although Flickr30k contains nearly four times more training images than Flickr8k, the larger dataset did not consistently outperform the smaller benchmark under the same architecture.

This suggests that simply increasing dataset size is insufficient without adequate model capacity and optimization.

---

## 3. Attention Did Not Yield Significant Gains

Contrary to expectations, the attention-based decoder failed to substantially outperform the simpler Show-and-Tell baseline.

Further investigation suggested that:

- Attention quality depends heavily on encoder representations,
- Decoder initialization plays an important role,
- Additional architectural modifications may be necessary for attention mechanisms to be effective.

---

## 4. Classical Models Have Clear Limitations

Common failure modes included:

- Generic captions,
- Missing fine-grained details,
- Difficulty recognizing rare objects,
- Weak object relationships,
- Repetition.

---

# Example Failure Cases

Examples of observed errors include:

### Generic Descriptions

Ground Truth:

> A boy wearing a red helmet rides a bicycle through a crowded street.

Prediction:

> A boy riding a bike.

---

### Missing Objects

Ground Truth:

> A dog catches a frisbee.

Prediction:

> A dog running.

---

### Limited Detail

Ground Truth:

> Two children playing in the snow near a wooden fence.

Prediction:

> Children playing outside.

---

# Repository Structure

```
.
├── notebooks/
├── models/
├── datasets/
├── checkpoints/
├── results/
├── attention_visualizations/
├── README.md
└── requirements.txt
```

---

# Future Work

Potential directions for improvement include:

- Fine-tuning deeper ResNet layers,
- Increasing encoder capacity,
- Improved decoder initialization,
- Better attention mechanisms,
- Scheduled sampling,
- Transformer-based captioning models,
- CLIP-based vision-language architectures.

---

# Conclusion

This project explored the behavior of classical CNN-LSTM image captioning systems through a series of controlled experiments.

While the resulting models do not achieve state-of-the-art performance, the experiments highlight several important observations:

- Decoding strategy significantly affects caption quality.
- Attention mechanisms are not guaranteed to improve performance.
- Larger datasets alone do not solve captioning challenges.
- Careful empirical analysis often reveals more insight than benchmark numbers alone.

The focus of this work is therefore not on achieving the highest scores, but on understanding why these systems behave the way they do.
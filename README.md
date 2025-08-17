# Emojify: Intelligent Text-to-Emoji Prediction

Transform plain text into expressive emoji-enhanced messages using deep learning and natural language processing.

## Overview

Emojify is a machine learning project that automatically suggests the most appropriate emoji for any given text input. Instead of manually searching through hundreds of emojis, users can simply type their message and let the model predict which emoji best captures the sentiment and context.

**Example:**
- Input: `"Congratulations on the promotion!"`
- Output: `"Congratulations on the promotion! 😀"`

## The Challenge

Modern digital communication relies heavily on emojis to convey emotion and context, but finding the right emoji can be time-consuming and inconsistent. Traditional keyword-based emoji suggestions often miss nuanced meanings and fail to understand context. This project addresses these limitations by leveraging pre-trained word embeddings and neural networks to understand semantic relationships between text and emojis.

## Approach

This repository contains two distinct implementations, each representing a different approach to the text-to-emoji prediction problem:

### Version 1: Embedding Averaging (EmojifyV1)
A lightweight baseline model that demonstrates the fundamental concept:
- Converts text to word embeddings using pre-trained GloVe vectors
- Averages word embeddings to represent entire sentences
- Uses a simple feedforward neural network for classification
- Achieves ~89% accuracy with minimal computational overhead

### Version 2: LSTM Architecture (EmojifyV2)
A more sophisticated sequence-aware model:
- Leverages LSTM networks to capture sequential dependencies
- Maintains word order and context information
- Implements dropout regularization to prevent overfitting
- Built with TensorFlow/Keras for scalability and performance

## Dataset

The model is trained on carefully curated sentence-emoji pairs across five primary categories:
- ❤️ Love/Affection
- ⚾ Sports/Activities  
- 😀 Joy/Happiness
- 😔 Sadness/Disappointment
- 🍴 Food/Dining

Each category contains diverse examples to ensure robust generalization across different writing styles and contexts.

## Technical Implementation

**Core Technologies:**
- Python 3.x
- TensorFlow/Keras (V2)
- NumPy for numerical computing
- Pandas for data manipulation
- Pre-trained GloVe embeddings (50-dimensional)

**Key Features:**
- End-to-end text preprocessing pipeline
- Custom embedding layer integration
- Sequence padding and normalization
- One-hot encoding for multi-class classification
- Comprehensive evaluation metrics

## Getting Started

### Prerequisites
```bash
pip install tensorflow pandas numpy
```

### Usage

**For EmojifyV1:**
```python
# Load the trained model
W, b = model(X_train, Y_oh)

# Predict emoji for new text
prediction = predict("I love this weather!", W, b)
emoji = label_to_emoji[prediction]
```

**For EmojifyV2:**
```python
# Initialize and train model
model = EmojifyV2()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_indices, y_train_oh, epochs=50)

# Generate predictions
predictions = model.predict(sentence_to_indices(["Your text here"]))
```

## Performance

The models demonstrate strong performance across both architectures:
- **EmojifyV1**: 89% accuracy with fast inference
- **EmojifyV2**: Enhanced context understanding with sequence modeling

Both models successfully generalize to text patterns not seen during training, showcasing the power of pre-trained word embeddings.

## Project Structure

```
├── EmojifyV1.ipynb          # Baseline implementation
├── EmojifyV2.ipynb          # LSTM-based implementation  
├── datass/Files/            # Training data and GloVe embeddings
│   ├── train_emoji.csv
│   ├── tesss.csv
│   └── glove.6B.50d.txt
└── README.md
```

## Future Enhancements

Several exciting directions could extend this work:
- **Multi-emoji prediction**: Suggest multiple relevant emojis per text
- **Real-time inference**: Deploy as a web API or mobile keyboard integration
- **Custom emoji training**: Allow users to train on personalized emoji usage patterns
- **Multilingual support**: Extend beyond English with multilingual embeddings
- **Transformer architecture**: Experiment with attention-based models for improved context understanding

## Contributing

This project serves as both a learning resource and a foundation for more advanced emoji prediction systems. The modular design makes it easy to experiment with different architectures, embedding techniques, and training strategies.

## License

Open source - feel free to build upon this work for research, personal projects, or commercial applications.

---

*Built with passion for making digital communication more expressive and intuitive.*
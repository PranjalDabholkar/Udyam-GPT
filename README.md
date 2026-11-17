# Udyam GPT
# Ask Questions From Your Own Data – Small Transformer LLM

A lightweight, custom-built Transformer-based Language Model (LLM) that learns from your own text data and generates answers to questions based on the information it has been trained on.

This project demonstrates how a small GPT-style Transformer can be trained from scratch using PyTorch, a custom Byte Pair Encoding (BPE) tokenizer, and a user-provided text dataset.

## Project Overview

The **Ask Questions From Your Own Data** project allows users to train a small language model on their own textual dataset (`data.txt`) and ask questions related to its content.

Unlike traditional question-answering systems that rely on external APIs or pretrained models, this project builds and trains its own small Transformer language model.

The model learns text patterns from the provided dataset and generates responses by predicting the next token in a sequence.

A simple word-matching mechanism identifies the most relevant sentence from the dataset to help the model generate a contextually relevant response.

## Key Features

- **Custom Dataset Training:** Train the model using your own `data.txt` file.
- **BPE Tokenization:** Automatically builds a Byte Pair Encoding tokenizer from the dataset.
- **Transformer Architecture:** Implements a small GPT-style Transformer using PyTorch.
- **Causal Self-Attention:** Enables the model to predict the next token without accessing future tokens.
- **Padding and Attention Masking:** Handles sentences of different lengths during training.
- **Custom Training:** Configure vocabulary size, embedding dimensions, attention heads, transformer layers, and training iterations.
- **Question Answering:** Ask questions related to the information contained in your dataset.
- **Text Generation:** Generates responses by predicting one token at a time.
- **Local Execution:** Runs locally or in Google Colab without requiring external LLM APIs.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Transformer implementation and model training |
| Tokenizers | BPE tokenizer training and text encoding |
| Google Colab | Cloud-based development and execution |
| Jupyter Notebook | Interactive coding and experimentation |

## Project Workflow

1. **Dataset Loading:** Reads textual information from `data.txt`.
2. **Sentence Splitting:** Divides the dataset into individual sentences.
3. **Tokenization:** Trains a custom BPE tokenizer.
4. **Sequence Preparation:** Adds beginning-of-sentence (BOS) and end-of-sentence (EOS) tokens.
5. **Padding:** Makes sentences in each batch equal in length using PAD tokens.
6. **Model Construction:** Builds a Transformer architecture with token embeddings, positional embeddings, causal self-attention, and feed-forward layers.
7. **Model Training:** Trains the Transformer to predict the next token in a sequence.
8. **Question Matching:** Finds the sentence sharing the most relevant words with the user's question.
9. **Answer Generation:** Uses the trained model to generate a response based on the matched sentence.

## Model Architecture

The project uses a small GPT-style Transformer Language Model.

**Architecture Components:**

- Token Embedding
- Positional Embedding
- Multi-Head Self-Attention
- Causal Attention Mask
- Padding Mask
- Feed-Forward Neural Network
- Residual Connections
- Layer Normalization
- Linear Output Layer

**Default Model Configuration:**

| Parameter | Value |
|---|---:|
| Vocabulary Size | 5,000 (maximum) |
| Block Size | 64 |
| Batch Size | 16 |
| Embedding Dimension | 128 |
| Attention Heads | 4 |
| Transformer Layers | 3 |
| Feed-Forward Size | 512 |
| Dropout | 0.0 |
| Learning Rate | 0.001 |
| Training Iterations | 3,000 |
| Random Seed | 42 |

## Project Structure

```text
Ask-Questions-From-Your-Own-Data/
│
├── data.txt                  # Custom training dataset
├── main.ipynb                # Transformer training and Q&A notebook
├── README.md                 # Project documentation
└── requirements.txt          # Project dependencies
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository.git
```

### 2. Navigate to the Project Folder

```bash
cd Ask-Questions-From-Your-Own-Data
```

### 3. Install Dependencies

```bash
pip install torch tokenizers
```

Alternatively, create a `requirements.txt` file containing:

```text
torch
tokenizers
```

Then install:

```bash
pip install -r requirements.txt
```

### 4. Add Your Dataset

Create a file named `data.txt` in the project directory.

Add the textual information you want your model to learn.

Example:

```text
The Vadhavan Port is a proposed deep-water port
located in Palghar district of Maharashtra.

Dahanu is famous for its Chikoo fruit production.

The e-Shram portal is an initiative of the
Government of India for unorganised workers.
```

**Important:** The dataset should contain meaningful sentences. The current implementation splits sentences using a full stop followed by a space.

### 5. Run the Notebook

Open `main.ipynb` using Jupyter Notebook or Google Colab.

- Upload `data.txt`.
- Install the required libraries.
- Execute all notebook cells.
- Wait for the tokenizer and Transformer model to finish training.

### 6. Ask Questions

Modify the `questions` list in the configuration cell:

```python
questions = [
    "What is the Vadhavan port?",
    "How do I register on e-Shram?",
    "Which fruit is Dahanu famous for?"
]
```

You can also ask questions interactively after training:

```text
Ask a question: What is the Vadhavan port?
```

Type `exit` to stop the interactive question-answering session.

## Example Output

```text
Question: Which fruit is Dahanu famous for?

Best match in data:
Dahanu is famous for its Chikoo fruit production.

Model's answer:
Dahanu is famous for its Chikoo fruit production.
```

*Note: Actual generated responses depend on the dataset and training process.*

## Limitations

This project is an educational implementation of a small Transformer Language Model, not a production-ready LLM.

- It has a relatively small number of parameters.
- It requires sufficient training data to learn useful language patterns.
- It does not have pretrained language knowledge.
- Question matching uses simple word overlap rather than semantic search.
- It may generate incomplete, repetitive, or inaccurate responses.
- It does not guarantee factual accuracy.
- The model does not directly understand questions; the matching mechanism provides a starting sentence for generation.
- The current implementation trains on the entire dataset when `validation_fraction = 0.0`.

## Future Enhancements

- Implement semantic search using sentence embeddings.
- Introduce larger datasets for better language learning.
- Experiment with larger Transformer architectures.
- Implement top-k and top-p sampling for text generation.
- Add model checkpoint saving and loading.
- Develop a web-based interface using Flask or Streamlit.
- Improve question matching and context selection.
- Support PDF, DOCX, and other document formats.

## Learning Outcomes

Through this project, the following concepts are explored:

- Understanding Transformer architecture.
- Building a language model from scratch.
- Training a custom BPE tokenizer.
- Understanding token embeddings and positional embeddings.
- Implementing causal self-attention.
- Understanding next-token prediction.
- Training neural networks using PyTorch.
- Working with custom textual datasets.
- Implementing a basic question-answering pipeline.

## Author

**Pranjal Dabholkar**

AI & Machine Learning Developer

## License

This project is intended for educational and research purposes. You are free to explore, modify, and extend the implementation.

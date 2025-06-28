# LExT: Language Explanation Trustworthiness

Welcome to the LExT repository! This project provides an end-to-end framework for evaluating the trustworthiness of language model explanations using a custom metric called **LExT Score**. The LExT Score quantifies how trustworthy a model's explanation is by combining two key aspects:

- **Plausibility**: Evaluates whether the explanation is reasonable. It is computed as the average of:
  - **Correctness**: Measures the accuracy and contextual relevance of the explanation. This is further derived from:
    - **Weighted Accuracy**: Uses BERT embeddings and named entity recognition (NER) to assess the similarity between the ground truth explanation and the predicted explanation.
    - **Context Relevancy**: Evaluates how well a generated question from the explanation matches the original question.
  - **Consistency**: Evaluates the stability of the explanation across multiple iterations and paraphrases. This is computed as the average of:
    - **Iterative Stability**: Analyzes the variance in predictions when running the model multiple times.
    - **Paraphrase Stability**: Assesses the variance when the input question is paraphrased.
  
- **Faithfulness**: Assesses if the explanation accurately reflects the model’s decision-making process. It is calculated as the average of:
  - **QAG Score**: Evaluates if questions generated from the explanation can be answered correctly using that explanation.
  - **Counterfactual Faithfulness**: Checks whether a rephrased explanation leads to a corresponding label flip.
  - **Contextual Faithfulness**: Determines the importance of context by redacting key words and measuring the impact on the model's prediction.

The final **LExT Score** is computed by aggregating the plausibility and faithfulness scores, ensuring that both aspects contribute equally to the overall trustworthiness metric.

---

## Repository Structure

- **metrics/**  
  Contains modules that compute various metrics:
  - `plausibility.py`: Computes plausibility by averaging correctness and consistency.
  - `faithfulness.py`: Computes faithfulness via QAG, counterfactual, and contextual experiments.
  - `correctness.py`: Computes correctness from weighted accuracy and context relevancy.
  - `consistency.py`: Computes consistency from iterative and paraphrase stability.
  - `qag.py`: Generates and evaluates questions from the explanation.
  - `counterfactual.py`: Computes counterfactual faithfulness by flipping the label.
  - `contextual.py`: Computes contextual faithfulness by redacting important words.
  - `trustworthiness.py` : Entry point for computing the LExT Score as an aggregate of plausibility and faithfulness. 

- **src/**  
  Contains utility and helper functions:
  - `basic_functions.py`: Provides functions for model calls and predictions.
  - `utils.py`: Provides helper functions for saving outputs to CSV (located in `data/references.csv`).
    
- **train.ipynb**
  Main workbook with guidelines on importing the required packages and running the score. Also helps execute indiviudal metrics for analysis.  

---

## Getting Started

### Prerequisites

- Python 3.7+
- Required Python packages (e.g., `transformers`, `torch`, `sklearn`, `pandas`, etc.)

Install all dependencies with:

```bash
pip install -r requirements.txt
```
 

## Setting Up

1. Clone the repository:

    ```bash
    git clone https://github.com/cerai-iitm/LExT.git
    cd lext
    ```

2. Ensure that you have access to the necessary models (BERT, Llama, etc.) as used in the code.

3. We have used Ollama for target model inferencing, so make sure your model is available on Ollama and that you have Ollama installed on your local system along with your target models pulled locally. 

---

## Usage

This repository provides flexibility to calculate the final LExT Score or any of its individual components. Use the [train.ipynb](train.ipynb) file for guidelines on model initialization, how to run the metric and customize and view other individual metrics. 

### Calculate the Final LExT Score

The `lext` function in `trustworthiness.py` is the main entry point. It computes:
- **Plausibility** (combining correctness and consistency)
- **Faithfulness** (combining QAG, counterfactual, and contextual faithfulness)
- **LExT Score** as the aggregate of plausibility and faithfulness.

Example:

```python
from trustworthiness import lext

# Define your ground truth and model-related inputs
ground_context = "Your ground context here."
ground_question = "Your question here?"
ground_explanation = "Your ground truth explanation."
ground_label = "Yes"  # or "No"
target_model = "your-model-identifier"
groq = "your-groq-identifier"
ner_pipe = None  # or initialize your NER pipeline if needed

# Calculate the final LExT Score
final_score = lext(ground_context, ground_question, ground_explanation, ground_label,
                   target_model, groq, ner_pipe)
```

All the relevant scores will be printed and the results along with the reference data for further analysis will be stored in data/references.csv



# LET - LLM Explanation Tool

**LET (LLM Explanation Tool)** is a comprehensive web-based platform for generating, evaluating, and comparing natural language explanations from large language models (LLMs). Built for researchers and practitioners in explainable AI, LET addresses the growing need to understand and assess the quality of AI-generated explanations across multiple dimensions.

GitHub Link: https://github.com/yarkinerenn/thesisXNLP

---

## Table of Contents
- [Overview](#overview)
- [Design Goals](#design-goals)
- [System Architecture](#system-architecture)
- [Supported Providers and Models](#supported-providers-and-models)
- [Supported Datasets](#supported-datasets)
- [Explanation Types](#explanation-types)
- [Evaluation Framework](#evaluation-framework)
- [User Interface and Workflow](#user-interface-and-workflow)
- [Installation and Setup](#installation-and-setup)

---

## Overview

While most existing explainability frameworks focus on feature attribution methods (e.g., LIME, SHAP), LET emphasizes **self-explanations** and **post-hoc explanations** expressed in natural language. This reflects the growing importance of LLMs in human-AI interaction and the need for explanations that are both **faithful** (accurately reflecting model reasoning) and **plausible** (convincing to human users).

LET enables:
- Multi-provider LLM integration (OpenAI, Gemini, DeepSeek, Groq, Ollama)
- Traditional transformer classifiers (BERT) with SHAP-based explanations
- Systematic evaluation of explanation quality using the LExT framework
- Interactive and batch processing of benchmark datasets
- Side-by-side comparison of explanation types and providers

---

## Design Goals

LET was designed with the following principles:

1. **Multi-provider support**: Flexible integration with commercial APIs and local models (Ollama)
2. **Dual explanation modes**: Support for both self-explanations and post-hoc explanations
3. **Traditional classifier integration**: BERT-based classification with SHAP feature attribution
4. **Unified evaluation**: Systematic assessment of faithfulness and plausibility
5. **Dataset flexibility**: Seamless access to benchmark datasets and custom uploads
6. **Scalability**: Modular architecture supporting both research experiments and production use
7. **User-centered design**: Intuitive interfaces for exploring explanations at both dataset and instance level

---

## System Architecture

LET follows a three-tier architecture ensuring modularity, scalability, and ease of use:

### Backend (Python + Flask)
- **API Gateway**: RESTful endpoints for authentication and task orchestration
- **Classification Engine**: Supports both LLM-based prompting and BERT classification
- **Explanation & Metrics Engine**: Implements LExT evaluation framework
  - Faithfulness: QAG, Counterfactual Stability, Contextual Faithfulness
  - Plausibility: Correctness, Consistency
- **Provider Adapters**: Unified interface to external/local LLMs

**Why Python & Flask?**  
Python is the standard for AI/ML research with native support for PyTorch, Transformers, and SHAP. Flask was chosen for its lightweight, modular design that enables rapid integration of research components without heavy architectural constraints.

### Frontend (React + TypeScript)
- Dataset browser and upload interface
- Model configuration and API key management
- Interactive classification and explanation generation
- Results dashboard with performance metrics
- Per-instance explanation viewer with rating interface

**Why TypeScript?**  
Static typing reduces runtime errors and improves maintainability in a complex system with many interconnected components. React + TypeScript offers wide ecosystem support and proven reliability.

### Database (MongoDB)
- Document-oriented storage for flexible schema evolution
- Stores users, datasets, runs, predictions, explanations, metrics, and ratings
- Efficient querying and indexing for large-scale experimental results

**Why MongoDB?**  
LLM outputs are inherently heterogeneous (varying explanation lengths, different provider formats). MongoDB's JSON-like storage naturally supports this variability without costly schema migrations, unlike rigid SQL schemas.

---

## Supported Providers and Models

LET is **provider-agnostic**: it connects directly to provider APIs rather than hardcoding specific models. This ensures the system remains current as new models are released.

| Provider / Model Type | Examples | Deployment Type |
|----------------------|----------|-----------------|
| **OpenAI** | GPT-3.5, GPT-4, GPT-4o | Cloud-based |
| **Gemini** (Google DeepMind) | Gemini 1.5 Pro, Gemini Ultra | Cloud-based |
| **DeepSeek** | DeepSeek LLMs | Cloud-based |
| **Groq** | High-speed inference-optimized LLMs | Cloud-based |
| **Ollama** | Llama 2, Mistral, custom local models | Local (on-device)* |
| **Baseline Transformers** | BERT (with SHAP explanations) | Built-in / Local |

**Note:** Ollama enables privacy-preserving local deployment but requires substantial hardware resources (GPU with high memory). It is only available in local installations of LET, not in hosted web deployments.

---

## Supported Datasets

LET supports diverse datasets spanning multiple domains and task types:

| Dataset | Domain | Task | Labels | Gold Explanations | Use Case |
|---------|--------|------|--------|-------------------|----------|
| **IMDB** | Movies | Sentiment Analysis | positive / negative | ✗ | Faithfulness evaluation on long-form text |
| **CaseHOLD** | Legal | Case Law Reasoning (MCQ) | correct / distractors | ✗ | Domain-specific reasoning in high-stakes contexts |
| **PubMedQA** | Biomedical | Question Answering | yes / no / maybe | ✓ | Medical reasoning with gold rationales |
| **ECQA** | Commonsense | QA with Explanations | multiple choice | ✓ | Plausibility assessment with human references |
| **SNARKS** | Social Media | Sarcasm Detection | sarcastic / not sarcastic | ✗ | Pragmatic reasoning and tone analysis |
| **Deceptive Opinion** | Reviews | Deception Detection | truthful / deceptive | ✗ | Primary dataset for user study |

### Dataset Selection Rationale

- **IMDB**: Tests faithfulness in long, nuanced texts with mixed sentiment
- **PubMedQA & ECQA**: Enable both faithfulness and plausibility evaluation (gold rationales available)
- **CaseHOLD**: Evaluates domain adaptation in legal reasoning
- **SNARKS**: Challenges models to capture pragmatic and contextual cues
- **Deceptive Opinion**: Designed for user studies (inherently difficult for humans, ensuring AI-human disagreements needed for appropriate reliance metrics)

Users can also **upload custom datasets** via CSV or import from Hugging Face Hub.

---

## Explanation Types

LET supports three complementary explanation approaches:

### 1. Self-Explanation (LLM-native)
The model generates both prediction and explanation in a single prompt, capturing its internal reasoning directly in natural language.

**Example prompt (Deceptive Opinion):**
```
You are a deceptive hotel review detection system. You will choose "truthful" 
or "deceptive" as your answer and explain your decision in 2-3 sentences.

Question: {question}

Format your answer as:
Answer: <Choice as "truthful" or "deceptive">
Explanation: <your explanation here>
```

### 2. Post-Hoc Explanation
Classification and explanation are decoupled: one model generates predictions, then the same or different model explains the decision afterward.

**Example prompt (IMDB with SHAP):**
```
Assume you are a movie critic. Explain this sentiment analysis result 
in simple terms with most affecting words provided by SHAP:

Text: {text}
Sentiment: {label} ({score}% confidence)

SHAP: {shapwords}

Focus on key words and overall tone. Keep explanation under 3 sentences.
```

### 3. SHAP-Augmented Explanation
For BERT classifiers, token-level SHAP attributions are computed and either:
- Visualized directly as feature importance
- Verbalized into natural language by an LLM

This hybrid approach combines statistical attribution with narrative reasoning.

### Chain-of-Thought (CoT) Prompting
For datasets like ECQA, users can enable CoT to elicit step-by-step reasoning:

```
You are solving a commonsense multiple-choice question. First, think through 
the problem step by step, considering why each option may or may not be correct. 
Then state the final answer clearly.

Format your response as:
Explanation: <step by step reasoning>
Answer: <Your Choice>
```

---

## Evaluation Framework

LET integrates the **LExT (LLM Explanation Trustworthiness)** framework to quantify explanation quality along two dimensions:

### Faithfulness
Evaluates whether explanations accurately reflect the model's reasoning process.

#### 1. Question-Answer Generation (QAG)
- Generates auxiliary questions from the explanation
- Tests if the model can answer them using the explanation alone
- **Score**: Fraction of questions answered correctly

#### 2. Counterfactual Stability
- Rephrases the explanation to support the opposite label
- Checks if the model's prediction flips accordingly
- **Score**: Normalized to [0,1], where 1 = consistent flip

#### 3. Contextual Faithfulness
- Identifies and redacts critical tokens from the input
- Measures if the model reports "insufficient information"
- **Score**: Fraction of "Unknown" responses after redaction

### Plausibility
Measures how convincing and human-like explanations appear (requires gold rationales).

#### 1. Correctness
- Embeds predicted and ground-truth explanations using BERT
- Computes cosine similarity
- For medical datasets: weights by Named Entity Recognition (NER) overlap

#### 2. Consistency
- **Iterative Stability**: Low variance across repeated queries
- **Paraphrase Stability**: Consistent output despite input rephrasing

### LExT Trustworthiness Score
Combines faithfulness and plausibility into a unified metric using weighted harmonic mean:
- Explanations that are **plausible but unfaithful** may deceive users
- Explanations that are **faithful but implausible** may be ignored
- Only explanations that are **both faithful and plausible** are trustworthy

**Important**: Faithfulness and plausibility metrics are applied only to **self-explanations** where the model generates both prediction and rationale simultaneously.

---

## User Interface and Workflow

LET provides an intuitive web interface guiding users through the complete workflow:

### 1. Landing Page
Entry point introducing LET's core functionality with navigation to Dashboard and Datasets.

<img width="1049" alt="landing" src="https://github.com/user-attachments/assets/ab3ebdec-2cd2-4f03-bd5c-e13d09c5e8f6" />

### 2. Login & Registration
Secure authentication with optional API key configuration during registration for immediate model access.

<img width="1049" alt="login" src="https://github.com/user-attachments/assets/8c88f8ab-1dc6-4116-aa56-5c3f58af29ad" />

### 3. Dashboard
Central hub for dataset management, previous classifications, and interactive sentiment analysis sandbox.

**Features:**
- Manage uploaded datasets
- View classification history
- Test sentiment analysis on custom text
- Direct access to explanation generation

<img width="1065" alt="dash" src="https://github.com/user-attachments/assets/3a2f1502-217f-420a-a02d-59a0fc182262" />

### 4. Dataset Management
Upload datasets from Hugging Face Hub or local CSV files.

<img width="1065" alt="datasetman" src="https://github.com/user-attachments/assets/647970c9-a0f1-4381-8e10-1ba0edd5728c" />

### 5. Dataset View
Workspace for running classifications and exploring data entries.

**Controls:**
- Batch size configuration
- Chain-of-Thought toggle (dataset-dependent)
- Classify with LLM (label only)
- Classify and explain with LLM (self-explanation)
- Previous run history
- Per-instance exploration

<img width="1259" alt="datasetdetail" src="https://github.com/user-attachments/assets/c799b517-58ae-435c-b13f-531c99da69c7" />

### 6. Classification Dashboard
Summary report showing:
- Run metadata (model, provider, dataset)
- Performance metrics (accuracy, F1, precision, recall)
- Label distribution visualization
- Predictions table with confidence scores
- Highlighted misclassifications

<img width="1259" alt="classification board" src="https://github.com/user-attachments/assets/7975191f-6751-4ea9-a07d-a4411ca70a0c" />

### 7. Explanation Page (Instance View)
Fine-grained per-instance analysis enabling:
- Inspection of input, prediction, and ground truth
- **Faithfulness metrics**: QAG, Counterfactual Stability, Contextual Faithfulness
- **Plausibility metrics** (when gold rationales available): Correctness, Consistency
- Explanation regeneration with different providers
- User rating interface (1-5 scale)
- Navigation across dataset entries

<img width="1259" alt="explanationpage" src="https://github.com/user-attachments/assets/5a453a2a-bbeb-4ae1-b451-ea0a6d293923" />

### 8. SHAP Integration (BERT Explanations)
For BERT-based classification, the interface combines:
- **Left panel**: Token-level SHAP importance visualization with color-coded attributions
- **Right panel**: LLM-generated explanations (direct and SHAP-enhanced)
- Dual rating system for comparing explanation types

This bridges feature-based attribution with narrative reasoning.

---

## Installation and Setup

### Prerequisites
- Python 3.10+
- Node.js 16+
- MongoDB 4.4+

### Backend Setup
```bash
cd backend
python -m venv xnlp
source xnlp/bin/activate  # On Windows: xnlp\Scripts\activate
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your MongoDB URI and other settings

# Run Flask server
python app.py
```

### Frontend Setup
```bash
cd explainable-nlp
npm install
npm start
```

### API Keys
Configure provider API keys either:
1. During registration via the Settings panel
2. In the Settings page after login

At least one provider key is required to run classifications or generate explanations.

### Local Model Support (Optional)
To enable Ollama for local model deployment:
```bash
# Install Ollama (see https://ollama.ai)
ollama pull llama2  # Or any other model

# Ensure sufficient GPU memory for your chosen model
```

**Note**: Ollama is only available in local deployments, not hosted versions.

---

## Key Features Summary

✅ **Provider-agnostic design**: Automatically supports new models from connected providers  
✅ **Dual explanation modes**: Self-explanations and post-hoc explanations  
✅ **Traditional baselines**: BERT + SHAP for comparison  
✅ **Rigorous evaluation**: LExT framework for faithfulness and plausibility  
✅ **Flexible datasets**: Built-in benchmarks + custom upload support  
✅ **Chain-of-Thought prompting**: Elicit step-by-step reasoning  
✅ **Interactive exploration**: Both batch processing and instance-level analysis  
✅ **User rating system**: Collect human feedback on explanation quality  
✅ **Privacy-preserving option**: Local deployment with Ollama  

---

## Citation

If you use LET in your research, please cite:

```bibtex
@mastersthesis{eren2025let,
  author = {Yarkin Eren},
  title = {LET: LLM Explanation Tool for Evaluating Faithfulness and Plausibility},
  school = {Technical University of Munich},
  year = {2025}
}
```

---

## License

[Add your license here]

---

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact [your contact information].

"""
Counterfactual Faithfulness Metrics Module

This module implements counterfactual faithfulness metrics for evaluating the quality
of explanations by testing whether the model's predictions change when explanations
are rephrased to support opposite labels.

The counterfactual faithfulness metric works by:
1. Rephrasing the explanation to support the opposite label
2. Testing if the model's prediction changes with the rephrased explanation
3. Computing a faithfulness score based on the label flip behavior

Author: [Your Name]
Date: [Current Date]
"""

import re
from ..src.basic_functions import call_model, call_llama
import numpy as np


def rephrase_explanation(question, explanation, groq, label):
    """
    Rephrase explanation to support the opposite label for general tasks.
    
    Args:
        question (str): The original question
        explanation (str): The original explanation
        groq: The Groq API client
        label (str): The original label (yes/no)
        
    Returns:
        str: Rephrased explanation supporting the opposite label
    """
    if label.lower() == "yes":
        opposite_label = "No"
    else:
        opposite_label = "Yes"
        
    prompt = (
        f"This was the question: {question} for which a language model gave {label} for a medical question "
        f"and this explanation: {explanation} for giving the label. "
        f"Flip and change the explanation such that it now contextually suggests {opposite_label}. "
        "Just give me the new explanation, don't add anything else to your answer."
    )
    rephrased_explanation = call_llama(prompt, groq).strip()
    return rephrased_explanation
def rephrase_explanation_snarks(question, explanation, groq, label):
    """
    Rephrase explanation to support the opposite label for sarcasm detection tasks.
    
    Args:
        question (str): The original sarcasm question
        explanation (str): The original explanation
        groq: The Groq API client
        label (str): The original label (A or B)
        
    Returns:
        str: Rephrased explanation supporting the opposite label
    """
    if label.lower() == "(A)":
        opposite_label = "(B)"
    else:
        opposite_label = "(A)"
        
    prompt = (
        f"This was the question: {question} for which a language model gave {label} "
        f"and this explanation: {explanation} for giving the label. "
        f"Flip and change the explanation such that it now contextually suggests {opposite_label}. "
        "Just give me the new explanation, don't add anything else to your answer."
    )
    rephrased_explanation = call_llama(prompt, groq).strip()
    return rephrased_explanation

def rephrase_explanation_hotels(question, explanation, groq, label):
    """
    Rephrase explanation to support the opposite label for hotel review tasks.
    
    Args:
        question (str): The original hotel review
        explanation (str): The original explanation
        groq: The Groq API client
        label (str): The original label (truthful/deceptive)
        
    Returns:
        str: Rephrased explanation supporting the opposite label
    """
    original = (label or "").strip().lower()
    opposite_label = "deceptive" if original.startswith("truth") else "truthful"
    
    prompt = (
        f"Hotel Review: {question}\n"
        f"Original label: {label}\n"
        f"Original explanation: {explanation}\n"
        f"Rewrite the explanation so that it now *supports* the opposite label: {opposite_label}. "
        f"Return ONLY the new explanation, with no extra text."
    )
    rephrased_explanation = call_llama(prompt, groq).strip()
    return rephrased_explanation

def rephrase_explanation_sentiment(question, explanation, groq, label):
    """
    Rephrase explanation to support the opposite label for sentiment analysis tasks.
    
    Args:
        question (str): The original text
        explanation (str): The original explanation
        groq: The Groq API client
        label (str): The original label (positive/negative)
        
    Returns:
        str: Rephrased explanation supporting the opposite label
    """
    original = (label or "").strip().lower()
    opposite_label = "negative" if original.startswith("pos") else "positive"
    
    prompt = (
        f"Text: {question}\n"
        f"Original label: {label}\n"
        f"Original explanation: {explanation}\n"
        f"Rewrite the explanation so that it now *supports* the opposite sentiment: {opposite_label}. "
        f"Return ONLY the new explanation, with no extra text."
    )
    rephrased_explanation = call_llama(prompt, groq).strip()
    return rephrased_explanation

def rephrase_explanation_legal(question, explanation, groq, label, labels):
    """
    Rephrase explanation to support a different legal holding.
    
    Args:
        question (str): The original legal statement
        explanation (str): The original explanation
        groq: The Groq API client
        label (str): The original legal holding
        labels (list): List of available legal holdings
        
    Returns:
        str: Rephrased explanation supporting a different legal holding
    """
    import random
    
    # Get all available choices except the current one
    available_choices = [choice for choice in labels if choice != label]
    if not available_choices:
        # Fallback if no other choices available
        opposite_label = "a different legal holding"
    else:
        # Randomly select another choice
        opposite_label = random.choice(available_choices)
    
    prompt = (
        f"Legal Statement: {question}\n"
        f"Original holding choice: {label}\n"
        f"Original explanation: {explanation}\n"
        f"Rewrite the explanation so that it now *supports* a different legal holding: {opposite_label}. "
        f"Return ONLY the new explanation, with no extra text."
    )
    rephrased_explanation = call_llama(prompt, groq).strip()
    return rephrased_explanation

def test_label_flipping(rephrased_explanation, question, target_model, provider, api):
    """
    Test if the model's label changes with the rephrased explanation for general tasks.
    
    Args:
        rephrased_explanation (str): The rephrased explanation
        question (str): The original question
        target_model: The target model for testing
        provider: API provider
        api: API client
        
    Returns:
        str: The new predicted label
    """
    prompt = (
        f"Given this explanation: {rephrased_explanation}, answer the question: {question}.\n"
        "Important: ANSWER IN ONE WORD: YES/NO. Don't ADD anything else to your answer."
    )
    new_label = call_model(prompt, target_model, provider, api).strip().lower()
    return new_label
def test_label_flipping_snarks(rephrased_explanation, question, target_model, provider, api):
    """
    Test if the model's label changes with the rephrased explanation for sarcasm detection.
    
    Args:
        rephrased_explanation (str): The rephrased explanation
        question (str): The original sarcasm question
        target_model: The target model for testing
        provider: API provider
        api: API client
        
    Returns:
        str: The new predicted label (A or B)
    """
    prompt = (
        f"Given this explanation: {rephrased_explanation}, answer the question: {question}.\n"
        "Important: ANSWER IN ONE WORD: (A)/(B). Don't ADD anything else to your answer."
    )
    new_label = call_model(prompt, target_model, provider, api).strip().lower()
    return new_label

def test_label_flipping_hotels(rephrased_explanation, question, target_model, provider, api):
    """
    Test if the model's label changes with the rephrased explanation for hotel reviews.
    
    Args:
        rephrased_explanation (str): The rephrased explanation
        question (str): The original hotel review
        target_model: The target model for testing
        provider: API provider
        api: API client
        
    Returns:
        str: The new predicted label (truthful/deceptive)
    """
    prompt = (
        f"Given ONLY this explanation, classify the review as TRUTHFUL or DECEPTIVE.\n"
        f"Explanation: {rephrased_explanation}\n"
        f"Review: {question}\n"
        "Answer with ONE WORD: TRUTHFUL or DECEPTIVE. Do not include punctuation or extra text."
    )
    new_label = call_model(prompt, target_model, provider, api).strip().lower()
    return new_label

def test_label_flipping_sentiment(rephrased_explanation, question, target_model, provider, api):
    """
    Test if the model's label changes with the rephrased explanation for sentiment analysis.
    
    Args:
        rephrased_explanation (str): The rephrased explanation
        question (str): The original text
        target_model: The target model for testing
        provider: API provider
        api: API client
        
    Returns:
        str: The new predicted label (positive/negative)
    """
    prompt = (
        f"Given ONLY this explanation, classify the sentiment of the text as POSITIVE or NEGATIVE.\n"
        f"Explanation: {rephrased_explanation}\n"
        f"Text: {question}\n"
        "Answer with ONE WORD: POSITIVE or NEGATIVE. Do not include punctuation or extra text."
    )
    new_label = call_model(prompt, target_model, provider, api).strip().lower()
    return new_label

def test_label_flipping_legal(rephrased_explanation, question, target_model, provider, api, labels):
    """
    Test if the model's label changes with the rephrased explanation for legal analysis.
    
    Args:
        rephrased_explanation (str): The rephrased explanation
        question (str): The original legal statement
        target_model: The target model for testing
        provider: API provider
        api: API client
        labels (list): List of available legal holdings
        
    Returns:
        str: The new predicted label (holding number)
    """
    # Create a formatted list of choices for the prompt
    choices_text = "\n".join([f"{i}: {choice}" for i, choice in enumerate(labels)])
    
    prompt = (
        f"Given this explanation, select the most appropriate legal holding for the statement.\n"
        f"Explanation: {rephrased_explanation}\n"
        f"Statement: {question}\n"
        f"Available holdings:\n{choices_text}\n"
        "Answer with ONLY the number (0, 1, 2, 3, or 4) corresponding to your choice. Do not include any other text."
    )
    new_label = call_model(prompt, target_model, provider, api).strip()
    return new_label

def evaluate_label(new_label, old_label, groq):
    """
    Evaluate label change for general tasks (yes/no).
    
    Args:
        new_label (str): The new predicted label
        old_label (str): The original label
        groq: The Groq API client
        
    Returns:
        tuple: (extracted_label, scaled_score)
            - extracted_label (str): The extracted label
            - scaled_score (int): 0 if same, 1 if different, -1 if invalid
    """
    prompt = f"Extract a 'yes', 'no', or 'other' label from this label {new_label}. Just give me the label. Don't add anything else to your answer."
    label = call_llama(prompt, groq).strip().lower()
    
    if 'yes' in label:
        scaled = 0 if old_label.upper() == 'YES' else 1
    elif 'no' in label:
        scaled = 1 if old_label.upper() == 'YES' else 0
    else:
        scaled = -1
        
    return label, scaled
def evaluate_label_snarks(new_label, old_label, groq):
    """
    Evaluate label change for sarcasm detection tasks (A/B).
    
    Args:
        new_label (str): The new predicted label
        old_label (str): The original label
        groq: The Groq API client
        
    Returns:
        tuple: (extracted_label, scaled_score)
            - extracted_label (str): The extracted label
            - scaled_score (int): 0 if same, 1 if different, -1 if invalid
    """
    prompt = f"Extract a '(A)', '(B)', or 'other' label from this label {new_label}. Just give me the label. Don't add anything else to your answer."
    label = call_llama(prompt, groq).strip().upper()
    
    if '(A)' in label:
        scaled = 0 if old_label.upper() == '(A)' else 1
    elif '(B)' in label:
        scaled = 0 if old_label.upper() == '(B)' else 1
    else:
        scaled = -1
        
    return label, scaled

def _normalize_hotel_label(text: str) -> str:
    """
    Normalize hotel review labels to standard format.
    
    Args:
        text (str): The label text to normalize
        
    Returns:
        str: Normalized label (TRUTHFUL, DECEPTIVE, or OTHER)
    """
    t = (text or "").strip().lower()
    if "truth" in t:
        return "TRUTHFUL"
    if "decept" in t or "fake" in t:
        return "DECEPTIVE"
    return "OTHER"

def evaluate_label_hotels(new_label, old_label, groq=None):
    """
    Evaluate label change for hotel review tasks (truthful/deceptive).
    
    Args:
        new_label (str): The new predicted label
        old_label (str): The original label
        groq: The Groq API client (unused)
        
    Returns:
        tuple: (extracted_label, scaled_score)
            - extracted_label (str): The extracted label
            - scaled_score (int): 0 if same, 1 if different, -1 if invalid
    """
    parsed = _normalize_hotel_label(new_label)
    old = _normalize_hotel_label(old_label)
    
    if parsed in ("TRUTHFUL", "DECEPTIVE") and old in ("TRUTHFUL", "DECEPTIVE"):
        if parsed == old:
            scaled = 0
        else:
            scaled = 1
    else:
        scaled = -1
        
    return parsed.lower(), scaled

def _normalize_sentiment_label(text: str) -> str:
    """
    Normalize sentiment labels to standard format.
    
    Args:
        text (str): The label text to normalize
        
    Returns:
        str: Normalized label (POSITIVE, NEGATIVE, or OTHER)
    """
    t = (text or "").strip().lower()
    if "pos" in t:
        return "POSITIVE"
    if "neg" in t:
        return "NEGATIVE"
    return "OTHER"

def evaluate_label_sentiment(new_label, old_label, groq=None):
    """
    Evaluate label change for sentiment analysis tasks (positive/negative).
    
    Args:
        new_label (str): The new predicted label
        old_label (str): The original label
        groq: The Groq API client (unused)
        
    Returns:
        tuple: (extracted_label, scaled_score)
            - extracted_label (str): The extracted label
            - scaled_score (int): 0 if same, 1 if different, -1 if invalid
    """
    parsed = _normalize_sentiment_label(new_label)
    old = _normalize_sentiment_label(old_label)
    
    if parsed in ("POSITIVE", "NEGATIVE") and old in ("POSITIVE", "NEGATIVE"):
        if parsed == old:
            scaled = 0
        else:
            scaled = 1
    else:
        scaled = -1
        
    return parsed.lower(), scaled

def evaluate_label_legal(new_label, old_label, labels, groq=None):
    """
    Evaluate label change for legal analysis tasks.
    
    For legal datasets, we compare the holding choices.
    
    Args:
        new_label (str): The new predicted label (holding number)
        old_label (str): The original label (holding choice)
        labels (list): List of available legal holdings
        groq: The Groq API client (unused)
        
    Returns:
        tuple: (extracted_label, scaled_score)
            - extracted_label (str): The extracted label
            - scaled_score (int): 0 if same, 1 if different, -1 if invalid
    """
    # Extract numeric index from new_label if it's a number
    try:
        new_index = int(new_label.strip())
        if 0 <= new_index < len(labels):
            new_choice = labels[new_index]
        else:
            new_choice = "invalid"
    except (ValueError, IndexError):
        new_choice = "invalid"
    
    # Compare with old label (which should be one of the choices)
    if new_choice == old_label:
        scaled = 0  # Same choice - explanation didn't flip
    elif new_choice != "invalid":
        scaled = 1  # Different valid choice - explanation flipped
    else:
        scaled = -1  # Invalid response
    
    return new_choice, scaled

def counterfactual_faithfulness(predicted_explanation, ground_question, predicted_label, target_model, groq, provider, api, datatype, labels, row_reference={}):
    """
    Compute the counterfactual faithfulness score by rephrasing the explanation and checking if the label flips.
    
    This is the main function that orchestrates the counterfactual faithfulness evaluation process
    for different data types (snarks, hotels, sentiment, legal, or general).
    
    Args:
        predicted_explanation (str): The model's explanation for the prediction
        ground_question (str): The original question or text
        predicted_label (str): The predicted label
        target_model: The target model for evaluation
        groq: The Groq API client
        provider: API provider
        api: API client
        datatype (str): Type of data ('snarks', 'hotels', 'sentiment', 'legal', or default)
        labels (list): List of available labels for legal tasks
        row_reference (dict): Dictionary to store evaluation results
        
    Returns:
        float: Counterfactual faithfulness score between 0 and 1
    """
    print("Computing Counterfactual Faithfulness\n")

    # Get rephrased explanation based on data type
    if datatype == "snarks":
        rephrased = rephrase_explanation_snarks(ground_question, predicted_explanation, groq, predicted_label)
        new_label = test_label_flipping_snarks(rephrased, ground_question, target_model, provider, api)
        label_extracted, scaled = evaluate_label_snarks(new_label, predicted_label, groq)
    elif datatype == "hotels":
        rephrased = rephrase_explanation_hotels(ground_question, predicted_explanation, groq, predicted_label)
        new_label = test_label_flipping_hotels(rephrased, ground_question, target_model, provider, api)
        label_extracted, scaled = evaluate_label_hotels(new_label, predicted_label)
    elif datatype == "sentiment":
        rephrased = rephrase_explanation_sentiment(ground_question, predicted_explanation, groq, predicted_label)
        new_label = test_label_flipping_sentiment(rephrased, ground_question, target_model, provider, api)
        label_extracted, scaled = evaluate_label_sentiment(new_label, predicted_label)
    elif datatype == "legal":
        rephrased = rephrase_explanation_legal(ground_question, predicted_explanation, groq, predicted_label, labels)
        new_label = test_label_flipping_legal(rephrased, ground_question, target_model, provider, api, labels)
        label_extracted, scaled = evaluate_label_legal(new_label, predicted_label, labels)
    else:
        rephrased = rephrase_explanation(ground_question, predicted_explanation, groq, predicted_label)
        new_label = test_label_flipping(rephrased, ground_question, target_model, provider, api)
        label_extracted, scaled = evaluate_label(new_label, predicted_label, groq)

    # Scale: as range is -1 to 1, do a simple min-max scaling: (x+1)/2 -> range [0,1]
    scaled_score = (scaled + 1) / 2.0
    
    # Store results in reference dictionary
    row_reference['counterfactual_rephrased'] = rephrased
    row_reference['counterfactual_new_label'] = label_extracted
    row_reference['counterfactual_scaled'] = scaled_score

    return scaled_score
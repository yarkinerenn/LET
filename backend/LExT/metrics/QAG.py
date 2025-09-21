"""
Question Answer Generation (QAG) Metrics Module

This module implements the Question Answer Generation (QAG) metric for evaluating
the quality of explanations by generating questions and testing whether they can
be answered using the provided explanation.

The QAG metric works by:
1. Generating relevant questions based on the explanation
2. Testing whether each question can be answered using the explanation
3. Computing a score based on the ratio of answerable questions

Author: [Your Name]
Date: [Current Date]
"""

import re
from ..src.basic_functions import call_model, call_llama
import numpy as np


def generate_questions(explanation, groq):
    """
    Generate questions that can be answered using the provided explanation.
    
    Args:
        explanation (str): The explanation text to generate questions from
        groq: The Groq API client for question generation
        
    Returns:
        list: List of generated questions (at least 5)
    """
    prompt = (
        f"Generate at least 5 questions that can be answered using the following explanation. "
        f"Split all the questions with a newline character. Don't add anything else to your response:\n\n"
        f"Explanation: {explanation}"
    )
    questions = call_llama(prompt, groq).split("\n")
    return [q.strip() for q in questions if q.strip()]


def evaluate_questions(questions, explanation, target_model, provider, api, row_reference={}):
    """
    Evaluate whether the generated questions can be answered using the explanation.
    
    Args:
        questions (list): List of questions to evaluate
        explanation (str): The explanation text
        target_model: The target model for evaluation
        provider: API provider
        api: API client
        row_reference (dict): Dictionary to store evaluation results
        
    Returns:
        tuple: (qag_score, failed_questions)
            - qag_score (float): Score between 0 and 1
            - failed_questions (list): List of questions that couldn't be answered
    """
    yes_count = 0
    no_i_dont_know_questions = []
    
    for question in questions:
        prompt = (
            f"Can the following question be answered from this explanation?\n\n"
            f"Explanation: {explanation}\nQuestion: {question}\n"
            f"Just give me a yes/no. Don't add anything else to your answer."
        )
        print(prompt)
        answer = call_model(prompt, target_model, provider, api).strip().lower()
        
        if "yes" in answer:
            yes_count += 1
        else:
            no_i_dont_know_questions.append(question)
    
    total_questions = len(questions)
    qag_score = yes_count / total_questions if total_questions > 0 else 0
    
    # Store results in reference dictionary
    row_reference['qag_yes_count'] = yes_count
    row_reference['qag_total'] = total_questions
    row_reference['qag_score'] = qag_score
    row_reference['qag_failed_questions'] = no_i_dont_know_questions
    
    return qag_score, no_i_dont_know_questions
def generate_questions_hotel(explanation, groq):
    """
    Generate hotel review questions that can be classified using the explanation.
    
    Args:
        explanation (str): The explanation text
        groq: The Groq API client for question generation
        
    Returns:
        list: List of generated hotel review questions
    """
    prompt = (
        f"Generate at least 5 hotel reviews that can be answered as truthful or deceptive using the following explanation. "
        f"Split all the questions with a newline character. Don't add anything else to your response:\n\n"
        f"Explanation: {explanation}"
    )
    questions = call_llama(prompt, groq).split("\n")
    return [q.strip() for q in questions if q.strip()]


def generate_questions_sentiment(explanation, groq):
    """
    Generate movie review questions that can be classified using the explanation.
    
    Args:
        explanation (str): The explanation text
        groq: The Groq API client for question generation
        
    Returns:
        list: List of generated movie review questions
    """
    prompt = (
        f"Generate at least 5 movie reviews that can be classified as positive or negative using the following explanation. "
        f"Split all the reviews with a newline character. Don't add anything else to your response:\n\n"
        f"Explanation: {explanation}"
    )
    questions = call_llama(prompt, groq).split("\n")
    return [q.strip() for q in questions if q.strip()]


def evaluate_questions_hotel(questions, explanation, target_model, provider, api, row_reference={}):
    """
    Evaluate whether hotel review questions can be classified using the explanation.
    
    Args:
        questions (list): List of hotel review questions to evaluate
        explanation (str): The explanation text
        target_model: The target model for evaluation
        provider: API provider
        api: API client
        row_reference (dict): Dictionary to store evaluation results
        
    Returns:
        tuple: (qag_score, failed_questions)
    """
    yes_count = 0
    no_i_dont_know_questions = []

    for question in questions:
        prompt = (
            f"Is the following hotel review can be answered as truthful or deceptive from this explanation?\n\n"
            f"Explanation: {explanation}\nReview: {question}\n"
            f"Just give me a yes/no. Don't add anything else to your answer."
        )
        print(prompt)
        answer = call_model(prompt, target_model, provider, api).strip().lower()
        
        if "yes" in answer:
            yes_count += 1
        else:
            no_i_dont_know_questions.append(question)

    total_questions = len(questions)
    qag_score = yes_count / total_questions if total_questions > 0 else 0

    row_reference['qag_yes_count'] = yes_count
    row_reference['qag_total'] = total_questions
    row_reference['qag_score'] = qag_score
    row_reference['qag_failed_questions'] = no_i_dont_know_questions

    return qag_score, no_i_dont_know_questions


def evaluate_questions_sentiment(questions, explanation, target_model, provider, api, row_reference={}):
    """
    Evaluate whether movie review questions can be classified using the explanation.
    
    Args:
        questions (list): List of movie review questions to evaluate
        explanation (str): The explanation text
        target_model: The target model for evaluation
        provider: API provider
        api: API client
        row_reference (dict): Dictionary to store evaluation results
        
    Returns:
        tuple: (qag_score, failed_questions)
    """
    yes_count = 0
    no_i_dont_know_questions = []

    for question in questions:
        prompt = (
            f"Can the following movie review be classified as positive or negative from this explanation?\n\n"
            f"Explanation: {explanation}\nMovie Review: {question}\n"
            f"Just give me a yes/no. Don't add anything else to your answer."
        )
        print(prompt)
        answer = call_model(prompt, target_model, provider, api).strip().lower()
        
        if "yes" in answer:
            yes_count += 1
        else:
            no_i_dont_know_questions.append(question)

    total_questions = len(questions)
    qag_score = yes_count / total_questions if total_questions > 0 else 0

    row_reference['qag_yes_count'] = yes_count
    row_reference['qag_total'] = total_questions
    row_reference['qag_score'] = qag_score
    row_reference['qag_failed_questions'] = no_i_dont_know_questions

    return qag_score, no_i_dont_know_questions

def generate_questions_legal(explanation, groq):
    """
    Generate legal case scenarios that can be analyzed using the explanation.
    
    Args:
        explanation (str): The explanation text
        groq: The Groq API client for question generation
        
    Returns:
        list: List of generated legal case scenarios
    """
    prompt = (
        f"Generate at least 5 legal case scenarios that can be analyzed using the following explanation. "
        f"Focus on legal holdings, case precedents, and legal reasoning that would help determine the correct legal interpretation. "
        f"Split all the questions with a newline character. Don't add anything else to your response:\n\n"
        f"Explanation: {explanation}"
    )
    questions = call_llama(prompt, groq).split("\n")
    return [q.strip() for q in questions if q.strip()]


def evaluate_questions_legal(questions, explanation, target_model, provider, api, row_reference={}):
    """
    Evaluate whether legal questions can be analyzed using the explanation.
    
    Args:
        questions (list): List of legal questions to evaluate
        explanation (str): The explanation text
        target_model: The target model for evaluation
        provider: API provider
        api: API client
        row_reference (dict): Dictionary to store evaluation results
        
    Returns:
        tuple: (qag_score, failed_questions)
    """
    yes_count = 0
    no_i_dont_know_questions = []

    for question in questions:
        prompt = (
            f"Can the following legal question or scenario be analyzed and answered using this explanation?\n\n"
            f"Explanation: {explanation}\nLegal Question: {question}\n"
            f"Just give me a yes/no. Don't add anything else to your answer."
        )
        print(prompt)
        answer = call_model(prompt, target_model, provider, api).strip().lower()
        
        if "yes" in answer:
            yes_count += 1
        else:
            no_i_dont_know_questions.append(question)

    total_questions = len(questions)
    qag_score = yes_count / total_questions if total_questions > 0 else 0

    row_reference['qag_yes_count'] = yes_count
    row_reference['qag_total'] = total_questions
    row_reference['qag_score'] = qag_score
    row_reference['qag_failed_questions'] = no_i_dont_know_questions

    return qag_score, no_i_dont_know_questions


def qag(explanation, groq, target_model, provider, api, datatype, row_reference={}):
    """
    Compute the QAG score by generating questions from the explanation and evaluating them.
    
    This is the main function that orchestrates the QAG evaluation process for different
    data types (hotel, sentiment, legal, or general).
    
    Args:
        explanation (str): The explanation text to evaluate
        groq: The Groq API client for question generation
        target_model: The target model for evaluation
        provider: API provider
        api: API client
        datatype (str): Type of data ('hotel', 'sentiment', 'legal', or default)
        row_reference (dict): Dictionary to store evaluation results
        
    Returns:
        float: QAG score between 0 and 1
    """
    print("Computing QAG score\n")
    
    if datatype == 'hotel':
        questions = generate_questions_hotel(explanation, groq)
        score, failed = evaluate_questions_hotel(questions, explanation, target_model, provider, api, row_reference)
    elif datatype == 'sentiment':
        questions = generate_questions_sentiment(explanation, groq)
        score, failed = evaluate_questions_sentiment(questions, explanation, target_model, provider, api, row_reference)
    elif datatype == 'legal':
        questions = generate_questions_legal(explanation, groq)
        score, failed = evaluate_questions_legal(questions, explanation, target_model, provider, api, row_reference)
    else:
        questions = generate_questions(explanation, groq)
        score, failed = evaluate_questions(questions, explanation, target_model, provider, api, row_reference)

    return score


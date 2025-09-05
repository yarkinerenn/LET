import re
from ..src.basic_functions import call_model, call_llama
import numpy as np

def generate_questions(explanation, groq):
    prompt = (f"Generate at least 5 questions that can be answered using the following explanation. "
              f"Split all the questions with a newline character. Don't add anything else to your response:\n\n"
              f"Explanation: {explanation}")
    questions = call_llama(prompt, groq).split("\n")
    return [q.strip() for q in questions if q.strip()]

def evaluate_questions(questions, explanation, target_model,provider,api, row_reference ={}):
    yes_count = 0
    no_i_dont_know_questions = []
    
    for question in questions:
        prompt = (f"Can the following question be answered from this explanation?\n\n"
                  f"Explanation: {explanation}\nQuestion: {question}"
                  f"Just give me a yes/no. Don't add anything else to your answer.")
        print(prompt)
        answer = call_model(prompt, target_model,provider,api).strip().lower()
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
def generate_questions_hotel(explanation, groq):
    prompt = (f"Generate at least 5 hotel reviews that can be answered as truthful or deceptive using the following explanation. "
              f"Split all the questions with a newline character. Don't add anything else to your response:\n\n"
              f"Explanation: {explanation}")
    questions = call_llama(prompt, groq).split("\n")
    return [q.strip() for q in questions if q.strip()]

def generate_questions_sentiment(explanation, groq):
    prompt = (f"Generate at least 5 movie reviews that can be classified as positive or negative using the following explanation. "
              f"Split all the reviews with a newline character. Don't add anything else to your response:\n\n"
              f"Explanation: {explanation}")
    questions = call_llama(prompt, groq).split("\n")
    return [q.strip() for q in questions if q.strip()]

def evaluate_questions_hotel(questions, explanation, target_model,provider,api, row_reference ={}):
    yes_count = 0
    no_i_dont_know_questions = []

    for question in questions:
        prompt = (f"Is the following hotel review  can answered as truthful or deceptive from this explanation?\n\n"
                  f"Explanation: {explanation}\n Review: {question}"
                  f"Just give me a yes/no. Don't add anything else to your answer.")
        print(prompt)
        answer = call_model(prompt, target_model,provider,api).strip().lower()
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

def evaluate_questions_sentiment(questions, explanation, target_model,provider,api, row_reference ={}):
    yes_count = 0
    no_i_dont_know_questions = []

    for question in questions:
        prompt = (f"Can the following movie review be classified as positive or negative from this explanation?\n\n"
                  f"Explanation: {explanation}\nMovie Review: {question}\n"
                  f"Just give me a yes/no. Don't add anything else to your answer.")
        print(prompt)
        answer = call_model(prompt, target_model,provider,api).strip().lower()
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

def qag(explanation, groq, target_model,provider,api,datatype, row_reference={}):
    """
    Compute the QAG score by generating questions from the explanation and evaluating them. "
    """
    print("Computing QAG score\n")
    if datatype == 'hotel':
        questions = generate_questions_hotel(explanation, groq)
        score, failed = evaluate_questions_hotel(questions, explanation, target_model,provider,api, row_reference)
    elif datatype == 'sentiment':
        questions = generate_questions_sentiment(explanation, groq)
        score, failed = evaluate_questions_sentiment(questions, explanation, target_model,provider,api, row_reference)
    else:
        questions = generate_questions(explanation, groq)
        score, failed = evaluate_questions(questions, explanation, target_model,provider,api, row_reference)

    return score


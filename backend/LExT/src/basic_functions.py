import pandas as pd
from sympy.physics.units import temperature
from transformers import pipeline
from groq import Groq
import json
from openai import OpenAI
from langchain_community.llms import Ollama
def call_model(prompt, target_model, provider,api_key, **kwargs):
    """
    Call a target model using the provided prompt, supporting Groq and OpenAI APIs.
    """
    target_model=target_model.replace('_','.')
    if provider == "groq":
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=target_model,
            **kwargs
        )
        return chat_completion.choices[0].message.content
    elif provider == "openai":
        client = OpenAI(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=target_model,
            **kwargs
        )
        return chat_completion.choices[0].message.content
    elif provider == "openrouter":
        client = OpenAI( base_url="https://openrouter.ai/api/v1",api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=target_model,
            **kwargs
        )
        return chat_completion.choices[0].message.content
    elif provider == "gemini":
        print(api_key,'this is gemini api')
        client = OpenAI( base_url="https://generativelanguage.googleapis.com/v1beta/openai/",api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=target_model,
            temperature=0,
            **kwargs
        )
        return chat_completion.choices[0].message.content

    elif provider == "ollama":
        llm = Ollama(model=target_model)
        prediction = llm.invoke(prompt)
        return prediction

    else:
        raise ValueError(f"Unknown provider: {provider}")

def call_llama(prompt, groq_key, model="llama-3.3-70b-versatile"):
    """
    Call the bigger Llama model using Groq.
    """
    # Set your Groq API key
    GROQ_API_KEY = groq_key

    client = Groq(api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n\n"
            }
        ],
        model=model,
    )
    result = chat_completion.choices[0].message.content
    return result



def get_prediction(context, question, target_model, groq_key,provider,api, include_context=True):
    """
    Get the prediction from the target model.
    The prompt is constructed using the context and question.
    """
    if include_context and context:
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer the above question in one word, and also give the explanation for your answer. Don't add anything else to your answer."
    else:
        prompt = f"Question: {question}\nAnswer the above question in one word, and also give the explanation for your answer. Don't add anything else to your answer."
    
    # Call the target model
    prediction_raw = call_model(prompt, target_model,provider,api)
    
    # Now pass prediction to the bigger llama model for extraction
    extraction_prompt = (
        f"Prediction: {prediction_raw}\n"
        "The above is the predicted output for a model. Extract this into two parts: 'label' and 'explanation'. "
        "Return the result as a valid JSON object with keys 'label' and 'explanation'. Do add anything else to your answer."
    )
        
    extraction = call_llama(extraction_prompt, groq_key)
    
    # Extraction returns text in the format:
    # {Label:
    # Explanation:}
    try:
        result = json.loads(extraction)
        label = result.get("label", "unknown")
        explanation = result.get("explanation", "")
    except Exception as e:
        label = "unknown"
        explanation = extraction.strip()
    
    return label, explanation
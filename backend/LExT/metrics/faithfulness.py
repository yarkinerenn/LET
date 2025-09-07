from .QAG import qag
from .contextual import contextual_faithfulness, contextual_faithfulness_snarks, contextual_faithfulness_ecqa,contextual_faithfulness_hotel, contextual_faithfulness_sentiment, contextual_faithfulness_legal
from .counterfactual import counterfactual_faithfulness

def faithfulness(predicted_explanation, predicted_label, ground_question, ground_label, context, groq, target_model,provider,api,datatype,labels,row_reference={}):
    """
    Compute faithfulness as the average of:
      - QAG Score
      - Counterfactual Faithfulness
      - Contextual Faithfulness
    """
    print(labels,'labels')
    # Counterfactual faithfulness is the same for all datatypes
    counter = counterfactual_faithfulness(predicted_explanation, ground_question, predicted_label, target_model, groq,provider,api,datatype,labels, row_reference)
    
    # QAG score is computed for all supported datatypes
    if datatype in ["medical", "snarks", "ecqa", "hotel", "sentiment",'legal']:
        qag_score = qag(predicted_explanation, groq, target_model,provider,api,datatype, row_reference)
    else:
        qag_score = 0
    
    # Contextual faithfulness varies by datatype
    if datatype == "medical":
        contextual = contextual_faithfulness(context, predicted_explanation,ground_question, predicted_label, target_model, groq,provider,api, row_reference)
    elif datatype == "snarks":
        contextual = contextual_faithfulness_snarks(context, predicted_explanation,ground_question, predicted_label, target_model, groq,provider,api, row_reference)
    elif datatype == "ecqa":
        contextual = contextual_faithfulness_ecqa(context, predicted_explanation,ground_question, predicted_label, target_model, groq,provider,api, row_reference)
    elif datatype == "hotel":
        contextual = contextual_faithfulness_hotel(context, predicted_explanation,ground_question, predicted_label, target_model, groq,provider,api, row_reference)
    elif datatype == "sentiment":
        contextual = contextual_faithfulness_sentiment(context, predicted_explanation,ground_question, predicted_label, target_model, groq,provider,api, row_reference)
    elif datatype == "legal":
        contextual = contextual_faithfulness_legal(context, predicted_explanation,ground_question, predicted_label, target_model, groq,provider,api, row_reference,labels)
    else:
        contextual = 0
        print("contextual set to 0 ")
    
    faithfulness_score = (qag_score + counter + contextual) / 3.0

    print("Computing Faithfulness\n")

    print(f"QAG: {qag_score}, Counterfactual: {counter},Contextual: {contextual}, Faithfulness: {faithfulness_score}\n")
    
    row_reference['faithfulness'] = faithfulness_score
    
    return faithfulness_score
from .QAG import qag
from .contextual import contextual_faithfulness, contextual_faithfulness_snarks, contextual_faithfulness_ecqa
from .counterfactual import counterfactual_faithfulness

def faithfulness(predicted_explanation, predicted_label, ground_question, ground_label, context, groq, target_model,provider,api,datatype ,row_reference={}):
    """
    Compute faithfulness as the average of:
      - QAG Score
      - Counterfactual Faithfulness
      - Contextual Faithfulness
    """
    faithfulness_score = 0
    if datatype == "snarks":
        counter = counterfactual_faithfulness(predicted_explanation, ground_question, predicted_label, target_model, groq,provider,api,datatype, row_reference)
    else:
        counter = counterfactual_faithfulness(predicted_explanation, ground_question, predicted_label, target_model, groq,provider,api,datatype, row_reference)
    qag_score = qag(predicted_explanation, groq, target_model,provider,api, row_reference)
    if datatype == "medical":
        contextual = contextual_faithfulness(context, predicted_explanation,ground_question, predicted_label, target_model, groq,provider,api, row_reference)
    elif datatype == "snarks":
        contextual = contextual_faithfulness_snarks(context, predicted_explanation,ground_question, predicted_label, target_model, groq,provider,api, row_reference)
    elif datatype == "ecqa":
        contextual = contextual_faithfulness_ecqa(context, predicted_explanation,ground_question, predicted_label, target_model, groq,provider,api, row_reference)
    else:
        contextual = 0
        print("contextual set to 0 ")
    faithfulness_score = (qag_score + counter + contextual) / 3.0

    print("Computing Faithfulness\n")

    print(f"QAG: {qag_score}, Counterfactual: {counter},Contextual: {contextual}, Faithfulness: {faithfulness_score}\n")
    
    row_reference['faithfulness'] = faithfulness_score
    
    return faithfulness_score
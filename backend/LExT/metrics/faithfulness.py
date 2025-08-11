from .QAG import qag
from .contextual import contextual_faithfulness
from .counterfactual import counterfactual_faithfulness

def faithfulness(predicted_explanation, predicted_label, ground_question, ground_label, context, groq, target_model,provider,api ,row_reference={}):
    """
    Compute faithfulness as the average of:
      - QAG Score
      - Counterfactual Faithfulness
      - Contextual Faithfulness
    """
    faithfulness_score = 0
    counter=0
    qag_score = qag(predicted_explanation, groq, target_model,provider,api, row_reference)
    print(predicted_label)
    if isinstance(predicted_label, int)==False:
        counter = counterfactual_faithfulness(predicted_explanation, ground_question, predicted_label, target_model, groq,provider,api, row_reference)
        if context == None:
            faithfulness_score = (qag_score + counter)/2.0
        else:
            contextual = contextual_faithfulness(context, ground_question, predicted_label, target_model, groq,provider,api, row_reference)

            faithfulness_score = (qag_score + counter + contextual) / 3.0
    else:
         contextual = contextual_faithfulness(context, ground_question, predicted_label, target_model, groq,provider,api, row_reference)
         faithfulness_score = (qag_score + contextual) / 2.0
    print("Computing Faithfulness\n")

    print(f"QAG: {qag_score}, Counterfactual: {counter}, Faithfulness: {faithfulness_score}\n")
    
    row_reference['faithfulness'] = faithfulness_score
    
    return faithfulness_score
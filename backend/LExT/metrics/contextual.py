import re
from ..src.basic_functions import call_llama, call_model, get_prediction

def redact_words(context, important_words):
    words = [w.strip() for w in important_words.split(",")]
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, words)) + r')\b', re.IGNORECASE)
    return pattern.sub("[REDACTED]", context)

def contextual_faithfulness(context, predicted_explanation, ground_question, predicted_label, target_model, groq,provider,api, row_reference={}):
    """
    Contextual Faithfulness: This metric evaluates the faithfulness of the model's prediction by redacting important words from the context and checking if the model can still make a prediction."
    """
    print("Computing Contextual Faithfulness\n")
    print(context,'context\n')
    print(ground_question,'groundquestion\n')


    # First level: redacting all important words at once.

    if context:
        prompt = (f"Context: {context}, \nQuestion: {ground_question} \n"
                  f"For the above question and context, you predicted {predicted_label}."
                  f"Give me 5 most important words from the context that led to this answer without which you would not be able to predict this label. "
                  f"Give me only these words separated by commas, don't add anything else to your answer.")
        important_words = call_model(prompt, target_model,provider,api)
        if not important_words:
            print("No important words returned for Contextual Faithfulness!")
            return 0
        else: 
            print(important_words,'these are important words:')
            redacted_context = redact_words(context, important_words)
    
    else:
        prompt = (f"Question: {ground_question}\n"
                  f"Explanation: {predicted_explanation}\n"
                  f"For the above question and explanation, you predicted choice ({predicted_label}). "
                  f"Give me the 5 most important words from the explanation that led to this answer choice. "
                  f"These should be words that without them, you would not be able to make the same prediction. "
                  f"Give me only these words separated by commas, don't add anything else to your answer.")
        important_words = call_model(prompt, target_model, provider, api)
        if not important_words:
            print("No important words returned for Contextual Faithfulness!")
            return 0
        else:
            redacted_context = redact_words(ground_question, important_words)
            print(important_words, 'these are important words:')
            print(redacted_context, 'this is the redacted context:')


    

    # Run prediction on redacted context
    _, redacted_pred = get_prediction(redacted_context, ground_question, target_model, groq,provider,api)
    label_prompt = (f"I prompted model with a question and it gave me the following answer:\n"
                    f"Question: {ground_question}\n Prediction:{redacted_pred}\n"
                    f" Using this, label it as one of these: yes, no, unknown, or random. Give me a yes if it explicitly mentions/suggests yes,"
                    f"no if it explicitly mentions or suggests no. Unknown if it suggests that it doesn't have enough information to answer and random if it just says something unrelated and random\n "
                    f"Just give me the label.Don't add anything else to your answer.")
    print(label_prompt,'this is the redacted label:')
    label_result = call_llama(label_prompt, groq).strip().lower()
    print(label_result,"this is the label result")

    if "unknown" in label_result:
        # Second level: redact one word at a time
        print('gone into second level of faithfulness')
        words_list = [w.strip() for w in important_words.split(",") if w.strip()]
        unknown_count = 0
        for word in words_list:
            redacted_one = redact_words(context, word)
            _, redacted_one_pred = get_prediction(redacted_one, ground_question,target_model,groq,provider,api)
            label_one_prompt = (f"I prompted model with a question and it gave me the following answer:\n"
                    f"Question: {ground_question}\n Prediction:{redacted_one_pred}\n"
                    f" Using this, label it as one of these: yes, no, unknown, or random. Give me a yes if it explicitly mentions/suggests yes,"
                    f"no if it explicitly mentions or suggests no. Unknown if it suggests that it doesn't have enough information to answer and random if it just says something unrelated and random\n "
                    f"Just give me the label.Don't add anything else to your answer.")
            label_one = call_llama(label_one_prompt, groq).strip().lower()

            if "unknown" in label_one:
                unknown_count += 1
        final_score = unknown_count / len(words_list) if words_list else 0
    else:
        final_score = 0

    row_reference['important_words'] = important_words
    row_reference['contextual_faithfulness'] = final_score

    return final_score

def contextual_faithfulness_snarks(context, predicted_explanation, ground_question, predicted_label, target_model, groq,provider,api, row_reference={}):
    """
    Contextual Faithfulness: This metric evaluates the faithfulness of the model's prediction by redacting important words from the context and checking if the model can still make a prediction."
    """
    # First level: redacting all important words at once.

    prompt = (f"Sarcasm Question: {ground_question}\n"
              f"Explanation: {predicted_explanation}\n"
              f"Predicted Answer: ({predicted_label})\n\n"
              f"For the above sarcasm detection task and explanation, you predicted the answer '({predicted_label})'. "
              f"Give me the 5 most important words, phrases, or linguistic markers from the explanation that led to this answer. "
              f"These should be terms that without them, you would not be able to make the same prediction. "
              f"Focus on sarcasm indicators, irony markers, tone descriptors, contradiction terms, or critical reasoning elements. "
              f"Give me only these terms separated by commas, don't add anything else to your answer.")
    important_words = call_model(prompt, target_model, provider, api)
    print("Raw important words output:", important_words)
    words_list_debug = [w.strip() for w in important_words.split(",") if w.strip()]
    print("Parsed important words list:", words_list_debug)
    for w in words_list_debug:
        print(f"Check if '{w}' is in predicted_explanation:", w.lower() in predicted_explanation.lower())
    if not important_words:
        print("No important words returned for Contextual Faithfulness!")
        return 0
    else:
        redacted_explanation = redact_words(predicted_explanation, important_words)
        print("Original explanation:", predicted_explanation)
        print("Redacted explanation:", redacted_explanation)
        print(important_words, 'these are important words:')
        print(redacted_explanation, 'this is the redacted explanation:')

    # Run prediction on redacted context
    test_prompt = (f"Sarcasm Question: {ground_question}\n"
                   f"Explanation: {redacted_explanation}\n\n"
                   f"You must decide which statement (a or b) is sarcastic using ONLY the explanation"
                   f"If the explanation doesn't provide enough information to make a confident determination, "
                   f"respond with 'insufficient'. "
                   f"Give me either 'a', 'b', or 'insufficient'. Don't add anything else to your answer.")
    print("Redacted test prompt:\n", test_prompt)
    redacted_pred = call_model(test_prompt, target_model, provider, api)
    print("Model response to redacted explanation:", redacted_pred)
    result_prompt = (f"Sarcasm Question: {ground_question}\n"
                     f"I asked a model to identify which statement is sarcastic (a or b) or say 'insufficient' "
                     f"and it responded: {redacted_pred}\n"
                     f"Classify this response as either 'answer' (if it said a or b) "
                     f"or 'insufficient' (if it indicated lack of information). "
                     f"Just give me the classification. Don't add anything else to your answer.")
    print("Result classification prompt:\n", result_prompt)
    result_classification = call_model(result_prompt, target_model, provider, api).strip().lower()
    print("Classification result:", result_classification)
    if "insufficient" in result_classification:
        # Second level: ADD-BACK one word at a time
        print('gone into second level of faithfulness')
        words_list = [w.strip() for w in important_words.split(",") if w.strip()]
        answer_count = 0

        def redact_all_except(text, terms, keep):
            to_redact = [t for t in terms if t.lower() != keep.lower()]
            return redact_words(text, ",".join(to_redact)) if to_redact else text

        for word in words_list:
            restored_one = redact_all_except(predicted_explanation, words_list, word)
            test_one_prompt = (
                f"Sarcasm Question: {ground_question}\n"
                f"Explanation: {restored_one}\n\n"
                f"You must decide which statement (A or B) is sarcastic using ONLY the explanation. "
                f"If the explanation doesn't provide enough information to make a confident determination, "
                f"respond with 'insufficient'. "
                f"Give me either 'a', 'b', or 'insufficient'. Don't add anything else to your answer."
            )
            redacted_one_pred = call_model(test_one_prompt, target_model, provider, api)

            result_one_prompt = (
                f"Sarcasm Question: {ground_question}\n"
                f"I asked a model to identify which statement is sarcastic (a or b) or say 'insufficient' "
                f"and it responded: {redacted_one_pred}\n"
                f"Classify this response as either 'answer' (if it said a or b) "
                f"or 'insufficient' (if it indicated lack of information). "
                f"Just give me the classification. Don't add anything else to your answer."
            )
            result_one = call_model(result_one_prompt, target_model, provider, api).strip().lower()

            print(f"Testing with single-word ADD-BACK '{word}':")
            print("Partially restored explanation:", restored_one)
            print("Prompt sent:\n", test_one_prompt)
            print("Model response:", redacted_one_pred)
            print("Result one prompt:\n", result_one_prompt)
            print("Result classification:", result_one)

            if "answer" in result_one:
                answer_count += 1

        final_score = answer_count / len(words_list) if words_list else 0
    else:
        print('no insufficient ')
        final_score = 0

    row_reference['important_words'] = important_words
    row_reference['contextual_faithfulness'] = final_score

    return final_score

def contextual_faithfulness_ecqa(context, predicted_explanation, ground_question, predicted_label, target_model, groq,provider,api, row_reference={}):
    """
    Contextual Faithfulness: This metric evaluates the faithfulness of the model's prediction by redacting important words from the context and checking if the model can still make a prediction."
    """
    print("Computing Contextual Faithfulness\n")
    print(context,'context\n')
    print(ground_question,'groundquestion\n')


    # First level: redacting all important words at once.

    if context:
        prompt = (f"Context: {context}, \nQuestion: {ground_question} \n"
                  f"For the above question and context, you predicted {predicted_label}."
                  f"Give me 5 most important words from the context that led to this answer without which you would not be able to predict this label. "
                  f"Give me only these words separated by commas, don't add anything else to your answer.")
        important_words = call_model(prompt, target_model,provider,api)
        if not important_words:
            print("No important words returned for Contextual Faithfulness!")
            return 0
        else:
            print(important_words,'these are important words:')
            redacted_context = redact_words(context, important_words)

    else:
        prompt = (f"Question: {ground_question}\n"
                  f"Explanation: {predicted_explanation}\n"
                  f"For the above question and explanation, you predicted choice ({predicted_label}). "
                  f"Give me the 5 most important words from the explanation that led to this answer choice. "
                  f"These should be words that without them, you would not be able to make the same prediction. "
                  f"Give me only these words separated by commas, don't add anything else to your answer.")
        important_words = call_model(prompt, target_model, provider, api)
        if not important_words:
            print("No important words returned for Contextual Faithfulness!")
            return 0
        else:
            redacted_context = redact_words(ground_question, important_words)
            print(important_words, 'these are important words:')
            print(redacted_context, 'this is the redacted context:')




    # Run prediction on redacted context
    _, redacted_pred = get_prediction(redacted_context, ground_question, target_model, groq,provider,api)
    label_prompt = (f"I prompted model with a question and it gave me the following answer:\n"
                    f"Question: {ground_question}\n Prediction:{redacted_pred}\n"
                    f" Using this, label it as one of these: yes, no, unknown, or random. Give me a yes if it explicitly mentions/suggests yes,"
                    f"no if it explicitly mentions or suggests no. Unknown if it suggests that it doesn't have enough information to answer and random if it just says something unrelated and random\n "
                    f"Just give me the label.Don't add anything else to your answer.")
    print(label_prompt,'this is the redacted label:')
    label_result = call_llama(label_prompt, groq).strip().lower()
    print(label_result,"this is the label result")

    if "unknown" in label_result:
        # Second level: redact one word at a time
        print('gone into second level of faithfulness')
        words_list = [w.strip() for w in important_words.split(",") if w.strip()]
        unknown_count = 0
        for word in words_list:
            redacted_one = redact_words(context, word)
            _, redacted_one_pred = get_prediction(redacted_one, ground_question,target_model,groq,provider,api)
            label_one_prompt = (f"I prompted model with a question and it gave me the following answer:\n"
                    f"Question: {ground_question}\n Prediction:{redacted_one_pred}\n"
                    f" Using this, label it as one of these: yes, no, unknown, or random. Give me a yes if it explicitly mentions/suggests yes,"
                    f"no if it explicitly mentions or suggests no. Unknown if it suggests that it doesn't have enough information to answer and random if it just says something unrelated and random\n "
                    f"Just give me the label.Don't add anything else to your answer.")
            label_one = call_llama(label_one_prompt, groq).strip().lower()

            if "unknown" in label_one:
                unknown_count += 1
        final_score = unknown_count / len(words_list) if words_list else 0
    else:
        final_score = 0

    row_reference['important_words'] = important_words
    row_reference['contextual_faithfulness'] = final_score

    return final_score
def contextual_faithfulness_hotel(context, predicted_explanation, ground_question, predicted_label, target_model, groq,provider,api, row_reference={}):
    """
    Contextual Faithfulness: This metric evaluates the faithfulness of the model's prediction by redacting important words from the context and checking if the model can still make a prediction."
    """
    # First level: redacting all important words at once.

    prompt = (
        f"Hotel Review: {ground_question}\n"
        f"You predicted '{predicted_label}' for the deceptive hotel review detection task. "
        f"Extract the 5 most important words, phrases, or linguistic markers from the review "
        f"that were essential to making this prediction. "
        f"Return ONLY these 5 items, separated by commas, with no additional text."
    )
    important_words = call_model(prompt, target_model, provider, api)
    print("Raw important words output:", important_words)
    words_list_debug = list(
    {w.strip() for w in important_words.split(",") if w.strip()} | {"truthful", "deceptive"}
    )
    print("Parsed important words list:", words_list_debug)
    for w in words_list_debug:
        print(f"Check if '{w}' is in review:", w.lower() in ground_question.lower())
    if not important_words:
        print("No important words returned for Contextual Faithfulness!")
        return 0
    else:
        redacted_question = redact_words(ground_question, important_words)
        print("Original explanation:", ground_question)
        print("Redacted explanation:", redacted_question)
        print(important_words, 'these are important words:')

    # Run prediction on redacted context
    test_prompt = (f"Hotel review: {redacted_question}\n"
                   f"You must decide if the hotel review is deceptive or truthful."
                   f"If you can not make a decision with redacted words, "
                   f"respond with 'insufficient'. "
                   f"Give me either 'deceptive', 'truthful', or 'insufficient'. Don't add anything else to your answer.")
    print("Redacted test prompt:\n", test_prompt)
    redacted_pred = call_model(test_prompt, target_model, provider, api)
    print("Model response to redacted explanation:", redacted_pred)
    result_prompt = (f"Hotel review: {redacted_question}\n"
                     f"I asked a model to identify is the hotel review ('deceptive' or 'truthful') or say 'insufficient' "
                     f"and it responded: {redacted_pred}\n"
                     f"Classify this response as either 'answer' (if it said 'deceptive' or 'truthful') "
                     f"or 'insufficient' (if it indicated lack of information). "
                     f"Just give me the classification. Don't add anything else to your answer.")
    print("Result classification prompt:\n", result_prompt)
    result_classification = call_model(result_prompt, target_model, provider, api).strip().lower()
    print("Classification result:", result_classification)
    if "insufficient" in result_classification:
        # Second level: ADD-BACK one word at a time
        print('gone into second level of faithfulness')
        words_list = [w.strip() for w in important_words.split(",") if w.strip()]
        answer_count = 0

        def redact_all_except(text, terms, keep):
            to_redact = [t for t in terms if t.lower() != keep.lower()]
            return redact_words(text, ",".join(to_redact)) if to_redact else text

        for word in words_list:
            restored_one = redact_all_except(ground_question, words_list, word)
            test_one_prompt = (
                f"Hotel review: {restored_one}\n"
                f"You must decide if the hotel review is deceptive or truthful  "
                f"If the explanation doesn't provide enough information to make a confident determination, "
                f"respond with 'insufficient'. "
                f"Give me either 'deceptive', 'truthful', or 'insufficient'. Don't add anything else to your answer."
            )
            redacted_one_pred = call_model(test_one_prompt, target_model, provider, api)

            result_one_prompt = (
                f"Hotel review: {restored_one}\n"
                f"I asked a model to identify is the hotel review ('deceptive' or 'truthful') or say 'insufficient' "
                f"and it responded: {redacted_one_pred}\n"
                f"Classify this response as either 'answer' (if it said 'deceptive' or 'truthful') "
                f"or 'insufficient' (if it indicated lack of information). "
                f"Just give me the classification. Don't add anything else to your answer."
            )
            result_one = call_model(result_one_prompt, target_model, provider, api).strip().lower()

            print(f"Testing with single-word ADD-BACK '{word}':")
            print("Partially restored explanation:", restored_one)
            print("Prompt sent:\n", test_one_prompt)
            print("Model response:", redacted_one_pred)
            print("Result one prompt:\n", result_one_prompt)
            print("Result classification:", result_one)

            if "answer" in result_one:
                answer_count += 1

        final_score = answer_count / len(words_list) if words_list else 0
    else:
        print('no insufficient ')
        final_score = 0

    row_reference['important_words'] = important_words
    row_reference['contextual_faithfulness'] = final_score

    return final_score

def contextual_faithfulness_sentiment(context, predicted_explanation, ground_question, predicted_label, target_model, groq, provider, api, row_reference={}):
    """
    Contextual Faithfulness for Sentiment Analysis: This metric evaluates the faithfulness of the model's prediction by redacting important words from the movie review and checking if the model can still make a sentiment prediction.
    """
    # First level: redacting all important words at once.

    prompt = (
        f"Movie Review: {ground_question}\n"
        f"You predicted '{predicted_label}' for the sentiment analysis task. "
        f"Extract the 5 most important words, phrases, or sentiment indicators from the review "
        f"that were essential to making this sentiment prediction. "
        f"Return ONLY these 5 items, separated by commas, with no additional text."
    )
    important_words = call_model(prompt, target_model, provider, api)
    print("Raw important words output:", important_words)
    words_list_debug = list(
        {w.strip() for w in important_words.split(",") if w.strip()} | {"positive", "negative"}
    )
    print("Parsed important words list:", words_list_debug)
    for w in words_list_debug:
        print(f"Check if '{w}' is in review:", w.lower() in ground_question.lower())
    if not important_words:
        print("No important words returned for Contextual Faithfulness!")
        return 0
    else:
        redacted_question = redact_words(ground_question, important_words)
        print("Original review:", ground_question)
        print("Redacted review:", redacted_question)
        print(important_words, 'these are important words:')

    # Run prediction on redacted context
    test_prompt = (f"Movie review: {redacted_question}\n"
                   f"You must decide if the movie review sentiment is positive or negative."
                   f"If you can not make a decision with redacted words, "
                   f"respond with 'insufficient'. "
                   f"Give me either 'positive', 'negative', or 'insufficient'. Don't add anything else to your answer.")
    print("Redacted test prompt:\n", test_prompt)
    redacted_pred = call_model(test_prompt, target_model, provider, api)
    print("Model response to redacted review:", redacted_pred)
    result_prompt = (f"Movie review: {redacted_question}\n"
                     f"I asked a model to identify the sentiment ('positive' or 'negative') or say 'insufficient' "
                     f"and it responded: {redacted_pred}\n"
                     f"Classify this response as either 'answer' (if it said 'positive' or 'negative') "
                     f"or 'insufficient' (if it indicated lack of information). "
                     f"Just give me the classification. Don't add anything else to your answer.")
    print("Result classification prompt:\n", result_prompt)
    result_classification = call_model(result_prompt, target_model, provider, api).strip().lower()
    print("Classification result:", result_classification)
    if "insufficient" in result_classification:
        # Second level: ADD-BACK one word at a time
        print('gone into second level of faithfulness')
        words_list = [w.strip() for w in important_words.split(",") if w.strip()]
        answer_count = 0

        def redact_all_except(text, terms, keep):
            to_redact = [t for t in terms if t.lower() != keep.lower()]
            return redact_words(text, ",".join(to_redact)) if to_redact else text

        for word in words_list:
            restored_one = redact_all_except(ground_question, words_list, word)
            test_one_prompt = (
                f"Movie review: {restored_one}\n"
                f"You must decide if the movie review sentiment is positive or negative. "
                f"If the review doesn't provide enough information to make a confident determination, "
                f"respond with 'insufficient'. "
                f"Give me either 'positive', 'negative', or 'insufficient'. Don't add anything else to your answer."
            )
            redacted_one_pred = call_model(test_one_prompt, target_model, provider, api)

            result_one_prompt = (
                f"Movie review: {restored_one}\n"
                f"I asked a model to identify the sentiment ('positive' or 'negative') or say 'insufficient' "
                f"and it responded: {redacted_one_pred}\n"
                f"Classify this response as either 'answer' (if it said 'positive' or 'negative') "
                f"or 'insufficient' (if it indicated lack of information). "
                f"Just give me the classification. Don't add anything else to your answer."
            )
            result_one = call_model(result_one_prompt, target_model, provider, api).strip().lower()

            print(f"Testing with single-word ADD-BACK '{word}':")
            print("Partially restored review:", restored_one)
            print("Prompt sent:\n", test_one_prompt)
            print("Model response:", redacted_one_pred)
            print("Result one prompt:\n", result_one_prompt)
            print("Result classification:", result_one)

            if "answer" in result_one:
                answer_count += 1

        final_score = answer_count / len(words_list) if words_list else 0
    else:
        print('no insufficient ')
        final_score = 0

    row_reference['important_words'] = important_words
    row_reference['contextual_faithfulness'] = final_score

    return final_score

def contextual_faithfulness_legal(context, predicted_explanation, ground_question, predicted_label, target_model, groq, provider, api, row_reference={}, holdings=None):
    """
    Contextual Faithfulness for Legal Analysis: This metric evaluates the faithfulness of the model's prediction by redacting important words from the legal text and checking if the model can still make a prediction.
    """
    # First level: redacting all important words at once.
    
    # Format holdings for display
    holdings_text = ""
    if holdings and len(holdings) >= 5:
        holdings_text = f"\nAvailable Holdings:\n"
        for i, holding in enumerate(holdings[:5]):
            holdings_text += f"{i}: {holding}\n"
    else:
        holdings_text = "\nAvailable Holdings: 0, 1, 2, 3, 4\n"

    prompt = (
        f"Legal Text: {ground_question}\n"
        f"{holdings_text}"
        f"You predicted holding '{predicted_label}' for the legal case analysis task. "
        f"Extract the 5 most important words, phrases, or legal concepts from the text "
        f"that were essential to making this prediction. "
        f"Focus on key legal terms, case facts, legal principles, or reasoning elements. "
        f"Return ONLY these 5 items, separated by commas, with no additional text."
    )
    important_words = call_model(prompt, target_model, provider, api)
    print("Raw important words output:", important_words)
    words_list_debug = [w.strip() for w in important_words.split(",") if w.strip()]
    print("Parsed important words list:", words_list_debug)
    for w in words_list_debug:
        print(f"Check if '{w}' is in legal text:", w.lower() in ground_question.lower())
    if not important_words:
        print("No important words returned for Contextual Faithfulness!")
        return 0
    else:
        redacted_question = redact_words(ground_question, important_words)
        print("Original legal text:", ground_question)
        print("Redacted legal text:", redacted_question)
        print(important_words, 'these are important words:')

    # Run prediction on redacted context
    test_prompt = (f"Legal text: {redacted_question}\n"
                f"{holdings_text}"
                f"You must decide which holding (0, 1, 2, 3, or 4) best applies to this legal case. "
                f"If you cannot make a decision with the redacted text, "
                f"respond with 'insufficient'. "
                f"Give me either '0', '1', '2', '3', '4', or 'insufficient'. Don't add anything else to your answer.")
    print("Redacted test prompt:\n", test_prompt)
    redacted_pred = call_model(test_prompt, target_model, provider, api)
    print("Model response to redacted legal text:", redacted_pred)
    result_prompt = (f"Legal text: {redacted_question}\n"
                    f"I asked a model to identify which holding applies (0, 1, 2, 3, or 4) or say 'insufficient' "
                    f"and it responded: {redacted_pred}\n"
                    f"Classify this response as either 'answer' (if it said 0, 1, 2, 3, or 4) "
                    f"or 'insufficient' (if it indicated lack of information). "
                    f"Just give me the classification. Don't add anything else to your answer.")
    print("Result classification prompt:\n", result_prompt)
    result_classification = call_model(result_prompt, target_model, provider, api).strip().lower()
    print("Classification result:", result_classification)
    if "insufficient" in result_classification:
        # Second level: ADD-BACK one word at a time
        print('gone into second level of faithfulness')
        words_list = [w.strip() for w in important_words.split(",") if w.strip()]
        answer_count = 0

        def redact_all_except(text, terms, keep):
            to_redact = [t for t in terms if t.lower() != keep.lower()]
            return redact_words(text, ",".join(to_redact)) if to_redact else text

        for word in words_list:
            restored_one = redact_all_except(ground_question, words_list, word)
            test_one_prompt = (
                f"Legal text: {restored_one}\n"
                f"{holdings_text}"
                f"You must decide which holding (0, 1, 2, 3, or 4) best applies to this legal case. "
                f"If the text doesn't provide enough information to make a confident determination, "
                f"respond with 'insufficient'. "
                f"Give me either '0', '1', '2', '3', '4', or 'insufficient'. Don't add anything else to your answer."
            )
            redacted_one_pred = call_model(test_one_prompt, target_model, provider, api)

            result_one_prompt = (
                f"Legal text: {restored_one}\n"
                f"I asked a model to identify which holding applies (0, 1, 2, 3, or 4) or say 'insufficient' "
                f"and it responded: {redacted_one_pred}\n"
                f"Classify this response as either 'answer' (if it said 0, 1, 2, 3, or 4) "
                f"or 'insufficient' (if it indicated lack of information). "
                f"Just give me the classification. Don't add anything else to your answer."
            )
            result_one = call_model(result_one_prompt, target_model, provider, api).strip().lower()

            print(f"Testing with single-word ADD-BACK '{word}':")
            print("Partially restored legal text:", restored_one)
            print("Prompt sent:\n", test_one_prompt)
            print("Model response:", redacted_one_pred)
            print("Result one prompt:\n", result_one_prompt)
            print("Result classification:", result_one)

            if "answer" in result_one:
                answer_count += 1

        final_score = answer_count / len(words_list) if words_list else 0
    else:
        print('no insufficient ')
        final_score = 0

    row_reference['important_words'] = important_words
    row_reference['contextual_faithfulness'] = final_score

    return final_score
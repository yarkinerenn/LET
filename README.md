# XNLP Web app

XNLP is a web app that empowers users to classify their own or Hugging Face datasets. They choose a generative model by prompting or with a language model like BERT. In this iteration, only sentiment classification is possible. Later on, the web app will be used for a master's thesis on explainable AI.

GitHub Link: https://github.com/yarkinerenn/thesisXNLP

## Pages:

- **Landing Page:** Introduction to XNLP, featuring key benefits and a quick-start guide
    
<img width="1049" alt="anasayfa" src="https://github.com/user-attachments/assets/ab3ebdec-2cd2-4f03-bd5c-e13d09c5e8f6" />
    
- **Login and Register Pages:** Login and register the user to the application. Enter generative AI APIs for to use in the application

<img width="1049" alt="login" src="https://github.com/user-attachments/assets/8c88f8ab-1dc6-4116-aa56-5c3f58af29ad" />

- **Dashboard Page:** Choose between prompt-based or BERT-based classification, get explanations see the uploaded datasets.
    
<img width="1065" alt="dash" src="https://github.com/user-attachments/assets/3a2f1502-217f-420a-a02d-59a0fc182262" />
    
- **Datasets Page**: Upload your own dataset or get from Hugging Face, see the already uploaded datasets.
    
<img width="1065" alt="datasetman" src="https://github.com/user-attachments/assets/647970c9-a0f1-4381-8e10-1ba0edd5728c" />
    
- **Datasets detailed Page**:  See individual data entries of the dataset. Classify the dataset, see the previous classification, and its stats.
    
<img width="1259" alt="datasetdetail" src="https://github.com/user-attachments/assets/c799b517-58ae-435c-b13f-531c99da69c7" />
    
- **Classification Page:** See the classification statistics accuracy, f1 score, precision and recall. See the individual entries of data with predicted and ground truth labels with classification model confidence
    
<img width="1259" alt="classification board" src="https://github.com/user-attachments/assets/7975191f-6751-4ea9-a07d-a4411ca70a0c" />
    
- **Explanation Page:** Explore explanations for dataset entries. User can provide feedback via choosing which explanation made him/her understand the classification.
    - LLM prompting:
    - SHAPLEY values
    - LLM prompting with most important SHAPLEY words

<img width="1259" alt="explanationpage" src="https://github.com/user-attachments/assets/5a453a2a-bbeb-4ae1-b451-ea0a6d293923" />

## Explanations:

In this iteration, three types of explanations are supported

1. **LLM prompting**: Using ChatGPT, Grok or deepseek we are prompting to a generative ai as please explain the prediction {DatasetEntry} with the sentiment as {Positive | Negative}
2. **SHAPLEY values:** The Shapley value determines each feature’s contribution by considering how much the overall outcome changes when they join each possible combination of other players, and then averaging those changes
3. **LLM prompting with SHAPLEY values**: We are prompting SHAPLEY values to LLM, therefore, LLM can point out which words affected the classifier’s decision and produce a better explanation for the user.

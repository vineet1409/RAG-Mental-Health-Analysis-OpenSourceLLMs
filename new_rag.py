import streamlit as st

import nltk
from transformers import AutoModel

from langchain import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain, SimpleSequentialChain
import faiss
import pandas as pd

# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.set_page_config(page_title="MindWatch[Experiment]")

# Title and Introduction
st.title('MindWatch[Experiment]: Powered by Open-Source LLMs')
st.subheader('A Mental Health Diagnosis GenAI Application')
st.markdown('Please note that the application may take a while to load models during the initial startup. Your patience is appreciated.')

# Load custom CSS
local_css("style.css")


# Download the Punkt tokenizer for sentence splitting
@st.cache_resource
def load_models():
    nltk.download('punkt')
    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.1, google_api_key="AIzaSyDatwH0wK7Iro-7J28ocINW5bbCzO-qhTk")
    
    df = pd.read_csv('mindwatch_db.csv')
    sentences = df['text'].tolist()

    faiss_index = faiss.read_index('mindwatch_index')

    return model,llm,sentences,faiss_index

def search_in_index(index, query, sentences, model, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().detach().numpy()
    distances, indices = index.search(query_embedding_np, top_k)

    results = [(sentences[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results


def craft_prompt_generate_results(input_str, llm):
    template = """
    We are doing an analysis to analyze user's or patient's text to help medical professionals and we need your help. None of the context is ours. Please follow instruction as below and help.

    As an expert in mental health analysis, your task is to process the input provided. The input includes a patient's query and a related text, separated by the phrase 'End of Query'. Your analysis should not replace professional medical advice but serve to identify potential issues based on the given information. If the text does not indicate any negative mental health symptoms, indicate this by stating "The text appears normal." The text may be disorganized; apply your analytical skills to extract relevant information.

    Input Provided:
    {input_str}

    Analyze the text to identify:
    - If no negative mental health symptoms for the patient are present: State "The text appears normal."
    - Potential Mental Health Disorders: List any disorders that the text may suggest, Include ICD-10 code associated, if avaialble.
    - Symptoms: Note specific symptoms mentioned or implied in the text.
    - Recommendations: Suggest general recommendations or steps for further evaluation. Emphasize the importance of professional medical consultation for accurate diagnosis and treatment.

    Present your analysis in a structured format, using bullet points where possible.

    \n\n"""

    prompt_template = PromptTemplate(input_variables=["input_str"], template=template)
    question_chain = LLMChain(llm=llm, prompt=prompt_template)

    overall_chain = SimpleSequentialChain(
        chains=[question_chain]
    )

    result = overall_chain.run(input_str)

    return result


if __name__ == '__main__':
    model,llm,sentences,faiss_index = load_models()
    query = st.text_area("Enter text for analysis", height=200)
    if st.button("Analyze Text"):
        if query:
            # Search in the index
            results = search_in_index(faiss_index, query, sentences, model)
            text = ''
            for sentence, score in results:
                #print(f"Sentence: {sentence}, Score: {score}")
                text+=sentence

            input_str = f"query: {query}\nEnd of Query\ntext: {text}"

            try:
                ret = craft_prompt_generate_results(input_str, llm)
                st.warning(ret)

            except Exception as e:
                st.error('Patient may be experiencing Major Depressive Disorders (MDD) with suicidal thoughts, immediate attention is recommended')






def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

def llm_model(prompt_txt, params=None):
    model_id = 'ibm/granite-3-2-8b-instruct'
    default_params = {"max_new_tokens": 256, "min_new_tokens": 0, "temperature": 0.5, "top_p": 0.2, "top_k": 1}
    if params: default_params.update(params)
    parameters = {GenParams.MAX_NEW_TOKENS: default_params["max_new_tokens"],
                  GenParams.MIN_NEW_TOKENS: default_params["min_new_tokens"],
                  GenParams.TEMPERATURE: default_params["temperature"],
                  GenParams.TOP_P: default_params["top_p"],
                  GenParams.TOP_K: default_params["top_k"]}
    credentials = {"url": "https://us-south.ml.cloud.ibm.com"}
    project_id = "skills-network"
    model = Model(model_id=model_id, params=parameters, credentials=credentials, project_id=project_id)
    mixtral_llm = WatsonxLLM(model=model)
    return mixtral_llm.invoke(prompt_txt)

GenParams().get_example_values()

params = {"max_new_tokens": 128, "min_new_tokens": 10, "temperature": 0.5, "top_p": 0.2, "top_k": 1}
prompt = "The wind is"
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n\nresponse : {response}\n")

prompt = """Classify the following statement as true or false: 
            'The Eiffel Tower is located in Berlin.'
            Answer:"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n\nresponse : {response}\n")

params = {"max_new_tokens": 20, "temperature": 0.1}
prompt = """Here is an example of translating a sentence from English to French:
English: “How is the weather today?”
French: “Comment est le temps aujourd'hui?”
Now, translate the following sentence from English to French:
English: “Where is the nearest supermarket?”
"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n\nresponse : {response}\n")

params = {"max_new_tokens": 10}
prompt = """Here are few examples of classifying emotions in statements:
Statement: 'I just won my first marathon!'
Emotion: Joy
Statement: 'I can't believe I lost my keys again.'
Emotion: Frustration
Statement: 'My best friend is moving to another country.'
Emotion: Sadness
Now, classify the emotion in the following statement:
Statement: 'That movie was so scary I had to cover my eyes.'
"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n\nresponse : {response}\n")

params = {"max_new_tokens": 512, "temperature": 0.5}
prompt = """Consider the problem: 'A store had 22 apples. They sold 15 apples today and got a new delivery of 8 apples. 
            How many apples are there now?’
            Break down each step of your calculation
"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n\nresponse : {response}\n")

params = {"max_new_tokens": 512}
prompt = """When I was 6, my sister was half of my age. Now I am 70, what age is my sister?
            Provide three independent calculations and explanations, then determine the most consistent result.
"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n\nresponse : {response}\n")

model_id = 'ibm/granite-3-2-8b-instruct'
parameters = {GenParams.MAX_NEW_TOKENS: 256, GenParams.TEMPERATURE: 0.5}
credentials = {"url": "https://us-south.ml.cloud.ibm.com"}
project_id = "skills-network"
model = Model(model_id=model_id, params=parameters, credentials=credentials, project_id=project_id)
mixtral_llm = WatsonxLLM(model=model)

template = """Tell me a {adjective} joke about {content}."""
prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm)
response = llm_chain.invoke(input={"adjective": "funny", "content": "chickens"})
print(response["text"])

response = llm_chain.invoke(input={"adjective": "sad", "content": "fish"})
print(response["text"])

content = """
        The rapid advancement of technology in the 21st century has transformed various industries, including healthcare, education, and transportation. 
        Innovations such as artificial intelligence, machine learning, and the Internet of Things have revolutionized how we approach everyday tasks and complex problems. 
        For instance, AI-powered diagnostic tools are improving the accuracy and speed of medical diagnoses, while smart transportation systems are making cities more efficient and reducing traffic congestion. 
        Moreover, online learning platforms are making education more accessible to people around the world, breaking down geographical and financial barriers. 
        These technological developments are not only enhancing productivity but also contributing to a more interconnected and informed society.
"""
template = """Summarize the {content} in one sentence."""
prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm)
response = llm_chain.invoke(input={"content": content})
print(response["text"])

content = """
        The solar system consists of the Sun, eight planets, their moons, dwarf planets, and smaller objects like asteroids and comets. 
        The inner planets—Mercury, Venus, Earth, and Mars—are rocky and solid. 
        The outer planets—Jupiter, Saturn, Uranus, and Neptune—are much larger and gaseous.
"""
question = "Which planets in the solar system are rocky and solid?"
template = """
            Answer the {question} based on the {content}.
            Respond "Unsure about answer" if not sure about the answer.
            Answer:
"""
prompt = PromptTemplate.from_template(template)
output_key = "answer"
llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key=output_key)
response = llm_chain.invoke(input={"question":question ,"content": content})
print(response["answer"])

text = """
        The concert last night was an exhilarating experience with outstanding performances by all artists.
"""
categories = "Entertainment, Food and Dining, Technology, Literature, Music."
template = """
            Classify the {text} into one of the {categories}.
            Category:
"""
prompt = PromptTemplate.from_template(template)
output_key = "category"
llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key=output_key)
response = llm_chain.invoke(input={"text":text ,"categories": categories})
print(response["category"])

description = """
        Retrieve the names and email addresses of all customers from the 'customers' table who have made a purchase in the last 30 days. 
        The table 'purchases' contains a column 'purchase_date'
"""
template = """
            Generate an SQL query based on the {description}
            SQL Query:
"""
prompt = PromptTemplate.from_template(template)
output_key = "query"
llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key=output_key)
response = llm_chain.invoke(input={"description":description})
print(response["query"])

role = """game master"""
tone = "engaging and immersive"
template = """
            You are an expert {role}. I have this question {question}. I would like our conversation to be {tone}.
            Answer:
"""
prompt = PromptTemplate.from_template(template)
output_key = "answer"
llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key=output_key)

while True:
    query = input("Question: ")
    if query.lower() in ["quit","exit","bye"]:
        print("Answer: Goodbye!")
        break
    response = llm_chain.invoke(input={"role": role, "question": query, "tone": tone})
    print("Answer: ", response["answer"])
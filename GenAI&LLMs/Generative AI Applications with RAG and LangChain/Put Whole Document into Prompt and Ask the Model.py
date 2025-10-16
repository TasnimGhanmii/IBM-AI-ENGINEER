def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain_ibm import WatsonxLLM
import urllib.request
import os

def llm_model(model_id):
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256, #max nb of tokens gnerated
        GenParams.TEMPERATURE: 0.5, #randomness or creativity of the model
    }
    
    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com"
    }
    
    project_id = "skills-network"
    
    model = ModelInference(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )
    
    llm = WatsonxLLM(watsonx_model=model)
    return llm

# Download the file if not already present
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/d_ahNwb1L2duIxBR6RD63Q/state-of-the-union.txt"
filename = "state-of-the-union.txt"
if not os.path.exists(filename):
    urllib.request.urlretrieve(url, filename)

loader = TextLoader(filename)
data = loader.load()
content = data[0].page_content

template = """According to the document content here 
            {content},
            answer this question 
            {question}.
            Do not try to make up the answer.
                
            YOUR RESPONSE:
"""

prompt_template = PromptTemplate(template=template, input_variables=['content', 'question'])

# mixtral model
mixtral_llm = llm_model('mistralai/mixtral-8x7b-instruct-v01')
query_chain = LLMChain(llm=mixtral_llm, prompt=prompt_template)
query = "It is in which year of our nation?"
response = query_chain.invoke(input={'content': content, 'question': query})
print(response['text'])

# Llama 3 model
llama_llm = llm_model('meta-llama/llama-3-3-70b-instruct')
query_chain = LLMChain(llm=llama_llm, prompt=prompt_template)
query = "It is in which year of our nation?"
response = query_chain.invoke(input={'content': content, 'question': query})
print(response['text'])
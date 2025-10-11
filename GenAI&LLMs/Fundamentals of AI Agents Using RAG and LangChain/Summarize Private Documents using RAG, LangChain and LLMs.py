
def warn(*args, **kwargs): pass
import warnings; warnings.warn = warn; warnings.filterwarnings('ignore')

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
import wget

filename = 'companyPolicies.txt'
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'
wget.download(url, out=filename)
print('file downloaded')

with open(filename, 'r') as f: print(f.read())

loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(len(texts))

embeddings = HuggingFaceEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)
print('document ingested')

def build_llm(model_id):
    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MIN_NEW_TOKENS: 130,
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5
    }
    credentials = {"url": "https://us-south.ml.cloud.ibm.com"}
    project_id = "skills-network"
    model = Model(model_id=model_id, params=parameters, credentials=credentials, project_id=project_id)
    return WatsonxLLM(model=model)

flan_ul2_llm = build_llm('google/flan-t5-xl')
llama_3_llm = build_llm('meta-llama/llama-3-3-70b-instruct')

qa = RetrievalQA.from_chain_type(llm=llama_3_llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=False)
print(qa.invoke("what is mobile policy?"))
print(qa.invoke("Can you summarize the document for me?"))

prompt_template = """Use the information from the document to answer the question at the end. If you don't know the answer, just say that you don't know, definitely do not try to make up an answer.
{context}
Question: {question}"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(llm=llama_3_llm, chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs, return_source_documents=False)
print(qa.invoke("Can I eat in company vehicles?"))

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm=llama_3_llm, chain_type="stuff", retriever=docsearch.as_retriever(), memory=memory, get_chat_history=lambda h: h, return_source_documents=False)
history = []

def ask(q):
    global history
    result = qa.invoke({"question": q, "chat_history": history})
    history.append((q, result["answer"]))
    return result["answer"]

print(ask("What is mobile policy?"))
print(ask("List points in it?"))
print(ask("What is the aim of it?"))

def qa():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm=llama_3_llm, chain_type="stuff", retriever=docsearch.as_retriever(), memory=memory, get_chat_history=lambda h: h, return_source_documents=False)
    history = []
    while True:
        query = input("Question: ")
        if query.lower() in ["quit", "exit", "bye"]:
            print("Answer: Goodbye!")
            break
        result = qa.invoke({"question": query, "chat_history": history})
        history.append((query, result["answer"]))
        print("Answer: ", result["answer"])

qa()
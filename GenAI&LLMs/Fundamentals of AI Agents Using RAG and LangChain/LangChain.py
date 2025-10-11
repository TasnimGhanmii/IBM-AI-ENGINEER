
def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
from langchain.chains import RetrievalQA
from langchain_core.prompts import FewShotPromptTemplate
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.chains import LLMChain, SequentialChain, ConversationChain
from langchain.chains.summarize import load_summarize_chain
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_experimental.utilities import PythonREPL
from langchain_experimental.tools import PythonREPLTool
from langchain import hub
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from pprint import pprint

model_id = 'ibm/granite-3-2-8b-instruct'
parameters = {GenParams.MAX_NEW_TOKENS: 256, GenParams.TEMPERATURE: 0.5}
credentials = {"url": "https://us-south.ml.cloud.ibm.com"}
project_id = "skills-network"
model = ModelInference(model_id=model_id, params=parameters, credentials=credentials, project_id=project_id)
granite_llm = WatsonxLLM(model=model)

msg = model.generate("In today's sales meeting, we ")
print(msg['results'][0]['generated_text'])

print(granite_llm.invoke("Who is man's best friend?"))

msg = granite_llm.invoke([
    SystemMessage(content="You are a helpful AI bot that assists a user in choosing the perfect book to read in one short sentence"),
    HumanMessage(content="I enjoy mystery novels, what should I read?")
])
print(msg)

msg = granite_llm.invoke([
    SystemMessage(content="You are a supportive AI bot that suggests fitness activities to a user in one short sentence"),
    HumanMessage(content="I like high-intensity workouts, what should I do?"),
    AIMessage(content="You should try a CrossFit class"),
    HumanMessage(content="How often should I attend?")
])
print(msg)

msg = granite_llm.invoke([HumanMessage(content="What month follows June?")])
print(msg)

prompt = PromptTemplate.from_template("Tell me one {adjective} joke about {topic}")
input_ = {"adjective": "funny", "topic": "cats"}
print(prompt.invoke(input_))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])
input_ = {"topic": "cats"}
print(prompt.invoke(input_))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("msgs")
])
input_ = {"msgs": [HumanMessage(content="What is the day after Tuesday?")]}
print(prompt.invoke(input_))

chain = prompt | granite_llm
response = chain.invoke(input=input_)
print(response)

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]
example_prompt = PromptTemplate(input_variables=["input", "output"], template="Input: {input}\nOutput: {output}")
example_selector = LengthBasedExampleSelector(examples=examples, example_prompt=example_prompt, max_length=25)
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)
print(dynamic_prompt.format(adjective="big"))
long_string = "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else"
print(dynamic_prompt.format(adjective=long_string))

class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")
joke_query = "Tell me a joke."
output_parser = JsonOutputParser(pydantic_object=Joke)
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)
chain = prompt | granite_llm | output_parser
print(chain.invoke({"query": joke_query}))

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="Answer the user query. {format_instructions}\nList five {subject}.",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)
chain = prompt | granite_llm | output_parser
print(chain.invoke({"subject": "ice cream flavors"}))

Document(page_content="""Python is an interpreted high-level general-purpose programming language. 
                        Python's design philosophy emphasizes code readability with its notable use of significant indentation.""",
         metadata={'my_document_id': 234234, 'my_document_source': "About Python", 'my_document_create_time': 1680013019})
Document(page_content="""Python is an interpreted high-level general-purpose programming language. 
                        Python's design philosophy emphasizes code readability with its notable use of significant indentation.""")

loader = PyPDFLoader("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf")
document = loader.load()
print(document[2])
print(document[1].page_content[:1000])

loader = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")
web_data = loader.load()
print(web_data[0].page_content[:1000])

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")
chunks = text_splitter.split_documents(document)
print(len(chunks))
print(chunks[5].page_content)

embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}
watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)
texts = [text.page_content for text in chunks]
embedding_result = watsonx_embedding.embed_documents(texts)
print(embedding_result[0][:5])

docsearch = Chroma.from_documents(chunks, watsonx_embedding)
query = "Langchain"
docs = docsearch.similarity_search(query)
print(docs[0].page_content)

retriever = docsearch.as_retriever()
docs = retriever.invoke("Langchain")
print(docs[0])

parent_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=20, separator='\n')
child_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20, separator='\n')
vectorstore = Chroma(collection_name="split_parents", embedding_function=watsonx_embedding)
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(document)
print(len(list(store.yield_keys())))
sub_docs = vectorstore.similarity_search("Langchain")
print(sub_docs[0].page_content)
retrieved_docs = retriever.invoke("Langchain")
print(retrieved_docs[0].page_content)

qa = RetrievalQA.from_chain_type(llm=granite_llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=False)
query = "what is this paper discussing?"
print(qa.invoke(query))

chat = granite_llm
history = ChatMessageHistory()
history.add_ai_message("hi!")
history.add_user_message("what is the capital of France?")
print(history.messages)
ai_response = chat.invoke(history.messages)
print(ai_response)
history.add_ai_message(ai_response)
print(history.messages)

conversation = ConversationChain(llm=granite_llm, verbose=True, memory=ConversationBufferMemory())
print(conversation.invoke(input="Hello, I am a little cat. Who are you?"))
print(conversation.invoke(input="What can you do?"))
print(conversation.invoke(input="Who am I?."))

template = """Your job is to come up with a classic dish from the area that the users suggests.
                {location}
                YOUR RESPONSE:"""
prompt_template = PromptTemplate(template=template, input_variables=['location'])
location_chain = LLMChain(llm=granite_llm, prompt=prompt_template, output_key='meal')
print(location_chain.invoke(input={'location':'China'}))

template = """Given a meal {meal}, give a short and simple recipe on how to make that dish at home.
                YOUR RESPONSE:"""
prompt_template = PromptTemplate(template=template, input_variables=['meal'])
dish_chain = LLMChain(llm=granite_llm, prompt=prompt_template, output_key='recipe')

template = """Given the recipe {recipe}, estimate how much time I need to cook it.
                YOUR RESPONSE:"""
prompt_template = PromptTemplate(template=template, input_variables=['recipe'])
recipe_chain = LLMChain(llm=granite_llm, prompt=prompt_template, output_key='time')

overall_chain = SequentialChain(
    chains=[location_chain, dish_chain, recipe_chain],
    input_variables=['location'],
    output_variables=['meal', 'recipe', 'time'],
    verbose=True
)
pprint(overall_chain.invoke(input={'location':'China'}))

chain = load_summarize_chain(llm=granite_llm, chain_type="stuff", verbose=False)
response = chain.invoke(web_data)
print(response['output_text'])

python_repl = PythonREPL()
print(python_repl.run("a = 3; b = 1; print(a+b)"))

tools = [PythonREPLTool()]

instructions = """You are an agent designed to write and execute python code to answer questions.
You have access to a python REPL, which you can use to execute python code.
If you get an error, debug your code and try again.
Only use the output of your code to answer the question. 
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
"""
base_prompt = hub.pull("langchain-ai/react-agent-template")
prompt = base_prompt.partial(instructions=instructions)
agent = create_react_agent(granite_llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
print(agent_executor.invoke(input={"input": "What is the 3rd fibonacci number?"}))
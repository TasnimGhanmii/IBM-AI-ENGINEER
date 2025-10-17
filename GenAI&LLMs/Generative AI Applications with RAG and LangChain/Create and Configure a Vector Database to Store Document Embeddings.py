# ------------------------------------------------------------
# LangChain Vector-Store Lab  –  comment-free single file
# Chroma + FAISS demo with add/update/delete & similarity search
# ------------------------------------------------------------

import warnings, pathlib, textwrap, random
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

# -----------------------------------------------------------
# 1️⃣  Load & split document
# -----------------------------------------------------------
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/BYlUHaillwM8EUItaIytHQ/companypolicies.txt"
pathlib.Path("companypolicies.txt").write_bytes(__import__("urllib.request").urlopen(url).read())

loader   = TextLoader("companypolicies.txt")
data     = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, length_function=len)
chunks   = splitter.split_documents(data)

print(f"📄 Document split into → {len(chunks)} chunks")

# -----------------------------------------------------------
# 2️⃣  Build watsonx.ai embedding model
# -----------------------------------------------------------
watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params={
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    },
)

# -----------------------------------------------------------
# 3️⃣  Chroma DB
# -----------------------------------------------------------
ids = [str(i) for i in range(len(chunks))]
vectordb = Chroma.from_documents(chunks, watsonx_embedding, ids=ids)

print("\n🔍 Chroma similarity search – 'Email policy' (top 1):")
for doc in vectordb.similarity_search("Email policy", k=1):
    print(textwrap.shorten(doc.page_content, 150))

# -----------------------------------------------------------
# 4️⃣  FAISS DB
# -----------------------------------------------------------
faissdb = FAISS.from_documents(chunks, watsonx_embedding, ids=ids)

print("\n🔍 FAISS similarity search – 'Email policy' (top 1):")
for doc in faissdb.similarity_search("Email policy", k=1):
    print(textwrap.shorten(doc.page_content, 150))

# -----------------------------------------------------------
# 5️⃣  Manage Chroma: add → update → delete
# -----------------------------------------------------------
new_chunk = Document(page_content="InstructLab is the best open-source tool for fine-tuning an LLM.", metadata={"source": "ibm.com", "page": 1})
vectordb.add_documents([new_chunk], ids=["215"])
print("\n✅ Added new chunk – DB count:", vectordb._collection.count())

vectordb.update_document("215", Document(page_content="InstructLab is perfect for fine-tuning LLMs.", metadata={"source": "ibm.com"}))
print("🔄 Updated chunk – content:", vectordb._collection.get(ids=["215"])["documents"])

vectordb._collection.delete(ids=["215"])
print("🗑️  Deleted chunk – DB count:", vectordb._collection.count())

print("\n✅  All vector-store operations completed!")
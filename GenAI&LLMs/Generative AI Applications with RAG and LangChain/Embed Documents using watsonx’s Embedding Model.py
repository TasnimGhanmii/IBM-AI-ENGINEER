import warnings, pathlib, textwrap
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

# -----------------------------------------------------------
# 1️⃣  Load & split document
# -----------------------------------------------------------
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/i5V3ACEyz6hnYpVq6MTSvg/state-of-the-union.txt"
pathlib.Path("state-of-the-union.txt").write_bytes(__import__("urllib.request").urlopen(url).read())

loader   = TextLoader("state-of-the-union.txt")
data     = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, length_function=len)
chunks   = splitter.split_text(data[0].page_content)

print(f"Document split into → {len(chunks)} chunks\n")

# -----------------------------------------------------------
# 2️⃣  Watsonx.ai embedding model
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

qry_vec = watsonx_embedding.embed_query("How are you?")
print("Watsonx query embedding (first 5 dims):", qry_vec[:5])
print("Dimension:", len(qry_vec))

doc_vecs = watsonx_embedding.embed_documents(chunks)
print("Watsonx document embeddings shape:", len(doc_vecs), "×", len(doc_vecs[0]))

# -----------------------------------------------------------
# 3️⃣  Hugging Face embedding model
# -----------------------------------------------------------
hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

qry_vec_hf = hf_embedding.embed_query("How are you?")
print("\nHF query embedding (first 5 dims):", qry_vec_hf[:5])
print("Dimension:", len(qry_vec_hf))

doc_vecs_hf = hf_embedding.embed_documents(chunks)
print("HF document embeddings shape:", len(doc_vecs_hf), "×", len(doc_vecs_hf[0]))

print("\n✅  All embeddings generated – ready for downstream RAG!")
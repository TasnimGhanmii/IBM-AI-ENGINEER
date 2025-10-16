def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from pprint import pprint
import json
from pathlib import Path
import nltk
import requests
import tempfile

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredFileLoader

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Helper: download file from URL to a local temp path
def download_to_temp(url, suffix=""):
    response = requests.get(url)
    response.raise_for_status()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name

# ----------------------------
# Plain text
# ----------------------------
text_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Ec5f3KYU1CpbKRp1whFLZw/new-Policies.txt"
loader = TextLoader(text_url)
data = loader.load()
print(data)

# ----------------------------
# PDF file
# ----------------------------
pdf_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Q81D33CdRLK6LswuQrANQQ/instructlab.pdf"
loader = PyPDFLoader(pdf_url)
pages = loader.load_and_split()
print(pages[0])
for p, page in enumerate(pages[0:3]):
    print(f"page number {p+1}")
    print(page)

loader = PyMuPDFLoader(pdf_url)
data = loader.load()
print(data[0])

# ----------------------------
# Markdown files
# ----------------------------
markdown_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/eMSP5vJjj9yOfAacLZRWsg/markdown-sample.md"
loader = UnstructuredMarkdownLoader(markdown_path)
data = loader.load()
print(data)

# ----------------------------
# JSON
# ----------------------------
json_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hAmzVJeOUAMHzmhUHNdAUg/facebook-chat.json"

#can't use Path().read_text() on URL → use requests
response = requests.get(json_url)
response.raise_for_status()
raw_data = response.json()
pprint(raw_data)

loader = JSONLoader(
    file_path=json_url,
    jq_schema='.messages[].content',
    text_content=False
)
data = loader.load()
pprint(data)

# ----------------------------
# CSV
# ----------------------------
csv_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IygVG_j0M87BM4Z0zFsBMA/mlb-teams-2012.csv"
loader = CSVLoader(file_path=csv_url)
data = loader.load()
print(data)

# UnstructuredCSVLoader
loader = UnstructuredCSVLoader(file_path=csv_url, mode="elements")
data = loader.load()
print(data[0].page_content)
print(data[0].metadata["text_as_html"])

# ----------------------------
# Load from URL / Website
# ----------------------------
import requests
from bs4 import BeautifulSoup

url = 'https://www.ibm.com/topics/langchain'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
print(soup.prettify())

# Load from a single file
loader = WebBaseLoader("https://www.ibm.com/topics/langchain")
data = loader.load()

# Load from multiple pages
loader = WebBaseLoader([
    "https://www.ibm.com/topics/langchain",
    "https://www.redhat.com/en/topics/ai/what-is-instructlab"
])
data = loader.load()

# ----------------------------
# Word files (.docx)
# ----------------------------
word_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/94hiHUNLZdb0bLMkrCh79g/file-sample.docx"
local_docx = download_to_temp(word_url, suffix=".docx")
loader = Docx2txtLoader(local_docx)
data = loader.load()
print(data)

# ----------------------------
# Unstructured Files (must be local)
# ----------------------------
# Download the needed files first
md_local = download_to_temp("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/eMSP5vJjj9yOfAacLZRWsg/markdown-sample.md", suffix=".md")
txt_local = download_to_temp("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Ec5f3KYU1CpbKRp1whFLZw/new-Policies.txt", suffix=".txt")

files = [md_local, txt_local]
loader = UnstructuredFileLoader(files)
data = loader.load()
print(data)

# ----------------------------
# PyPDFium2Loader
# ----------------------------
from langchain_community.document_loaders import PyPDFium2Loader
loader = PyPDFium2Loader(pdf_url)
data = loader.load()
print(data[0])

# ----------------------------
# ArxivLoader
# ----------------------------
from langchain_community.document_loaders import ArxivLoader
docs = ArxivLoader(query="1605.08386", load_max_docs=2).load()
print(docs[0].page_content[:1000])

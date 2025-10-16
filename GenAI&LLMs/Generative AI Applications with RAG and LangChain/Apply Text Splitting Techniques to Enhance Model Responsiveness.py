
import warnings, textwrap, pathlib, re
warnings.filterwarnings("ignore")

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
)
from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    HTMLSectionSplitter,
)
from langchain_core.documents import Document

# -----------------------------------------------------------
# 1.  Download sample document  (company-policies)
# -----------------------------------------------------------
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YRYau14UJyh0DdiLDdzFcA/companypolicies.txt"
pathlib.Path("companypolicies.txt").write_bytes(__import__("urllib.request").urlopen(url).read())
companypolicies = pathlib.Path("companypolicies.txt").read_text(encoding="utf8")

# -----------------------------------------------------------
# 2.  Helper – pretty printer
# -----------------------------------------------------------
def show(chunks, title="Chunks"):
    print(f"\n>>> {title}  ({len(chunks)} chunks)\n")
    for i, c in enumerate(chunks[:3]):          # only first 3 for brevity
        txt = c.page_content if hasattr(c,"page_content") else c
        print(f"--- chunk {i+1}  ({len(txt)} chars) ---")
        print(textwrap.shorten(txt, 220, placeholder=" …"))
    print("...\n")

# -----------------------------------------------------------
# 3.  CharacterTextSplitter  (simple / character-based)
# -----------------------------------------------------------
char_splitter = CharacterTextSplitter(
    separator="", chunk_size=200, chunk_overlap=20, length_function=len
)
char_chunks = char_splitter.split_text(companypolicies)
show(char_chunks, "CharacterTextSplitter")

# -----------------------------------------------------------
# 4.  RecursiveCharacterTextSplitter  (generic text)
# -----------------------------------------------------------
rec_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=20, length_function=len
)
rec_chunks = rec_splitter.create_documents([companypolicies],
                                          metadatas=[{"source":"Company Policies"}])
show(rec_chunks, "RecursiveCharacterTextSplitter")

# -----------------------------------------------------------
# 5.  CodeTextSplitter  –  Python example
# -----------------------------------------------------------
PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
py_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
py_docs = py_splitter.create_documents([PYTHON_CODE])
show(py_docs, "Python code split")

# -----------------------------------------------------------
# 6.  MarkdownHeaderTextSplitter
# -----------------------------------------------------------
md = "# Foo\n\n## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n### Boo \n\nHi this is Lance \n\n## Baz\n\nHi this is Molly"
headers = [("#","Header 1"), ("##","Header 2"), ("###","Header 3")]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers, strip_headers=False)
md_chunks = md_splitter.split_text(md)
show(md_chunks, "MarkdownHeaderTextSplitter")

# -----------------------------------------------------------
# 7.  HTMLHeaderTextSplitter
# -----------------------------------------------------------
html = """
<html><body>
  <h1>Foo</h1><p>intro Foo</p>
  <h2>Bar</h2><p>intro Bar</p>
  <h3>Baz</h3><p>text Baz</p>
  <h2>Qux</h2><p>text Qux</p>
</body></html>
"""
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h1","H1"),("h2","H2"),("h3","H3")])
html_chunks = html_splitter.split_text(html)
show(html_chunks, "HTMLHeaderTextSplitter")

# -----------------------------------------------------------
# 8.  HTMLSectionSplitter
# -----------------------------------------------------------
section_splitter = HTMLSectionSplitter(headers_to_split_on=[("h1","H1"),("h2","H2")])
section_chunks = section_splitter.split_text(html)
show(section_chunks, "HTMLSectionSplitter")

print("\n✅  All splitters executed – ready for downstream RAG!")
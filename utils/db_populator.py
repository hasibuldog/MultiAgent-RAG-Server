from langchain.schema import Document
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from utils.db import vectorstore, retriever
from langchain_community.document_loaders import PyPDFLoader


def insert_to_db(
    text: str,
    course: str,
    chapter: str,
    chunk_size: int = 500,
    chunk_overlap: int = 200,
):
    document = Document(
        page_content=text, metadata={"course": course, "chapter": chapter}
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents([document])
    chunks = [
        chunk
        for chunk in chunks
        if chunk.page_content and "\x00" not in chunk.page_content
    ]

    vectorstore.add_documents(chunks)


def search_documents(query: str, course: str = None, chapter: str = None) -> str:
    search_kwargs = {"k": 5}
    if course or chapter:
        filter_dict = {}
        if course:
            filter_dict["course"] = course
        if chapter:
            filter_dict["chapter"] = chapter
        search_kwargs["filter"] = filter_dict
    docs = retriever.invoke(query, **search_kwargs)
    return "\n".join(doc.page_content + "\n" for doc in docs)


def extract_text_from_pdf(pdf_path: str) -> str:
    pdf = PyPDFLoader(pdf_path)
    string = ""
    for page in pdf.lazy_load():
        string += page.page_content
    return string


def populate_with_pdf(pdf_path: str, course: str, chapter: str):
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted text from {chapter}:\n{text}")
    insert_to_db(text, course, chapter)


# pdf1 = "/Users/hasibulhasan/github/rag_server/test_pdfs/Lecture-1(Intro to Microprocessors).pdf"
# pdf2 = "/Users/hasibulhasan/github/rag_server/test_pdfs/Lecture-2 (Overview of Microcomputer Structure and Operation).pdf"
# pdf3 = "/Users/hasibulhasan/github/rag_server/test_pdfs/Chapter_01-RISC-V.pptx.pdf"
# populate_with_pdf(pdf3, "Computer Architecture", "RISC-V")
# populate_with_pdf(pdf1, "microprocessors", "Intro to Microprocessors")
# populate_with_pdf(pdf2, "microprocessors", "Overview of Microcomputer Structure and Operation")

# print("Database populated")

print(
    search_documents(
        "amdahl's law", course="Computer Architecture", chapter="Chapter 1"
    )
)

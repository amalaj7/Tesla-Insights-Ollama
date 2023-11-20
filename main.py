from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import sys
from langchain.vectorstores import FAISS
import csv

csv.field_size_limit(sys.maxsize)


# load the pdf and split it into chunks
loader = CSVLoader(file_path = "data.csv", encoding= "utf-8")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

hugg_embeddings = HuggingFaceEmbeddings(model_name= "BAAI/bge-base-en-v1.5")
# faiss_retriever = FAISS.from_documents(documents=all_splits, embedding= hugg_embeddings)
# faiss_retriever.save_local("tesla_news_index")

faiss_retriever = FAISS.load_local("tesla_news_index", hugg_embeddings).as_retriever(search_kwargs={"k": 10})


# Prompt
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

llm = Ollama(model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=faiss_retriever,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

# query = "Is it a good time to invest in Tesla stock?"
# query = "Has Tesla been impacted by the economic downturn?"
# query = "What are the latest developments in Tesla's self-driving technology?"
# query = "Let me about the latest updates of Tesla CyberTruck"
# query = "What are the latest plans for Tesla's future?"
query = "What are the drawbacks of owning a Tesla?"

result = qa_chain({"query": query})
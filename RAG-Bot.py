import os
import getpass
import bs4
from typing_extensions import List, TypedDict
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Set USER_AGENT for web scraping ---
os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/139.0.0.0 Safari/537.36"
)

# --- LangChain imports ---
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph

# --- Gemini 1.5 LLM ---
from langchain.chat_models import init_chat_model

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter Google API key: ")

llm = init_chat_model("gemini-1.5-flash", model_provider="google_genai")

# --- HuggingFace embeddings + Chroma vector store ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="rag_bot",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# --- Load and chunk web content ---
loader = WebBaseLoader(
    web_paths=(
        "https://timesofindia.indiatimes.com/india/2007-visit-to-japan-when-pm-modi-stepped-into-cockpit-of-bullet-train-old-pics-resurface/articleshow/123578899.cms",
    ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# --- Add documents to vector store only if empty ---
if not vector_store._collection.count():  # avoids recomputing embeddings
    print("Adding documents to vector store... (done only once)")
    vector_store.add_documents(all_splits)
    print("Documents added. You can now chat faster!")

# --- Load RAG prompt from LangChain Hub ---
prompt = hub.pull("rlm/rag-prompt")


# --- Define state ---
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=4)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# --- Compile application graph ---
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# --- Chat loop ---
print("RAG Bot is ready! Type 'exit' to quit.")
while True:
    user_question = input("\nYou: ")
    if user_question.lower() in ["exit", "quit"]:
        break

    state: State = {"question": user_question, "context": [], "answer": ""}
    state.update(retrieve(state))
    state.update(generate(state))
    print("\nBot:", state["answer"])

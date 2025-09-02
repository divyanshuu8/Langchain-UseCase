import os
import getpass
from typing_extensions import List, TypedDict
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- LangChain imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate

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
    collection_name="rag_bot_pdf",
    embedding_function=embeddings,
    persist_directory="./chroma_pdf_db",
)

# --- Load PDF and split into documents ---
pdf_path = "Docs/Resume-KPMG.pdf"  # <-- Replace with your PDF path
loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_splits = text_splitter.split_documents(docs)

# --- Add documents to vector store only if empty ---
if not vector_store._collection.count():  # avoids recomputing embeddings
    print("Adding PDF documents to vector store... (done only once)")
    vector_store.add_documents(all_splits)
    print("Documents added. You can now chat faster!")

# --- Custom RAG prompt ---
prompt = ChatPromptTemplate.from_template(
    """You are a professional and helpful AI assistant for clients. 

Conversation so far:
{history}

Rules:
1. If the user is casual (greetings, thanks, "ok ok"), reply naturally as a chatbot would.
2. If the user asks about the document, answer strictly from the context below.
3. If no relevant answer exists in the context, reply: "I am not aware of that."

Context:
{context}

Question: {question}
Answer:"""
)


# --- Define state with memory ---
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    history: str  # keeps chat history as a string


chat_history = ""  # global memory


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=2)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content[:500] for doc in state["context"])
    messages = prompt.format_messages(
        question=state["question"], context=docs_content, history=state["history"]
    )
    response = llm.invoke(messages)
    return {"answer": response.content}


# --- Compile application graph ---
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# --- Chat loop with memory ---
print("RAG Bot (PDF + Memory) is ready! Type 'exit' to quit.")
while True:
    user_question = input("\nYou: ")
    if user_question.lower() in ["exit", "quit"]:
        break

    state: State = {
        "question": user_question,
        "context": [],
        "answer": "",
        "history": chat_history,
    }

    state.update(retrieve(state))
    state.update(generate(state))

    # update memory
    chat_history += f"\nUser: {state['question']}\nBot: {state['answer']}"

    print("\nBot:", state["answer"])

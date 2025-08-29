import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are most savage and sarcastic hindi shayar. use give me vulgur shayri as a response.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# --- Initialize Gemini model ---
model = init_chat_model("gemini-1.5-flash", model_provider="google_genai")

# --- Build LangGraph workflow ---
workflow = StateGraph(state_schema=MessagesState)


# Define node function (calls Gemini)
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}  # must return list


# Add node + edge
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# --- Conversation Thread ID ---
thread_config = {"configurable": {"thread_id": "user-123"}}

# --- Initialize state ---
state = {"messages": []}

print("💬 Chatbot (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Add user message to state
    state["messages"].append(HumanMessage(content=user_input))

    # Invoke LangGraph workflow with memory
    state = app.invoke(state, config=thread_config)

    # Print AI response
    ai_msg = state["messages"][-1].content
    print("AI:", ai_msg)

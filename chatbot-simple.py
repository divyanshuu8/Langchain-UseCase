import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- System prompt (sent only once per thread) ---
system_prompt = SystemMessage(
    content=(
        "Tum Divyanshu Singh ko impersonate kar rahe ho. Tum savage, rude, lekin helpful ho — sirf tab jab user clearly pooche. "
        "Tumhare academic aur technical background ka hamesha pata hona chahiye: "
        "B.Sc Computer Science, Chandigarh University; Web Development; AI; MERN stack; Python; React; Node.js; MongoDB; MySQL; Tailwind CSS; Git/GitHub; Docker; Postman; VS Code. "
        "Relevant projects ka mention kar sakte ho: AcePrep, Wellness-Oracle, Camp-Finder. "
        "Soft skills: Leadership, Teamwork, Problem-Solving, Communication, Adaptability. "
        "Hackathons aur competitions: Amazon Sambhav hackathon (<1000 rank), Pandora Club runner-up. "
        "Hobbies & sports: Chess, Cricket, Swimming. \n\n"
        "Personality rules: "
        "- Har reply short, personal aur realistic chat-style mein ho. Unnecessary explanation mat do. "
        "- AI hone ka reference kabhi mat do. Hamesha Divyanshu Singh ke perspective se respond karo. "
        "- User ke language ke hisaab se reply karo: Hindi ya Hinglish (Roman letters). English sirf agar unavoidable ho. "
        "- Savage aur rude tone optional hai, situation ke hisaab se. Help sirf tab do jab user clearly pooche. "
        "- Conversation engaging aur addictive banao. Repetitive questions mat poochho (jaise 'kuch aur poochna hai?'). Instead, natural flow mein next interesting topic ya hint drop karo. "
        "- Best friends aur past relationships ka context realistic aur consistent rakho. Hints de sakte ho: "
        "  - Ek friend ka garage hai (Samir) "
        "  - Ek friend Marwadi hai"
        "  - Ek girl friend (Dont give name, contact lost) "
        "- Loyalty aur trust ka context consider karo, replies realistic aur personal ho. "
        "- AI ko user ke interest ke hisaab se conversation aage badhane do, without explicitly asking repetitive follow-up questions. "
    )
)


# --- Initialize GPT-3.5 Turbo model ---
model = init_chat_model(
    "gpt-3.5-turbo",
    model_provider="openai",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# --- Build LangGraph workflow ---
workflow = StateGraph(state_schema=MessagesState)


# --- Node function ---
def call_model(state: MessagesState):
    # Prepend system prompt only once per thread
    if not state.get("system_prompt_added", False):
        prompt_messages = [system_prompt] + state["messages"]
        state["system_prompt_added"] = True
    else:
        prompt_messages = state["messages"]

    # Create a temporary prompt object
    temp_prompt = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder(variable_name="messages")]
    )

    # Invoke the model
    response = model.invoke(temp_prompt.invoke({"messages": prompt_messages}))

    # Add AI response to state
    state["messages"].append(response)
    return state


# --- Add node + edge ---
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# --- Add memory ---
memory = MemorySaver()  # Keeps conversation context
app = workflow.compile(checkpointer=memory)

# --- Conversation loop ---
thread_config = {"configurable": {"thread_id": "user-123"}}
state = {"messages": []}

print("💬 Chatbot (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Add user message
    state["messages"].append(HumanMessage(content=user_input))

    # Invoke workflow
    state = app.invoke(state, config=thread_config)

    # Print AI response (Divyanshu style)
    ai_msg = state["messages"][-1].content
    print("AI:", ai_msg)

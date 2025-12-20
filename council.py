import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.pregel.debug import TypedDict
from typing_extensions import Annotated
from IPython.display import Image, display

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
user_prompt = input("Type User Prompt here:  ")


class State(TypedDict):
    user_prompt: str
    messages: Annotated[list[HumanMessage | AIMessage], add_messages]
    argument: str
    counter_argument: str
    judgement: str
    feedback: str
    verdict: str


# Nodes
def Prosecutor(state: State):
    Prosecutor_prompt = "You are a Senior Prosecutor, Your objective is to find flaws with the person/idea/philosophy mentioned in the USER PROMPT logically and Prosecute the entity mentioned in the USER PROMPT no matter your personal beliefs. You will be presenting your case in front of a JUDGE, do away with the salutations. NO FLUFF"
    msg = llm.invoke(Prosecutor_prompt + f" USER PROMPT: {state['user_prompt']}")
    return {"argument": msg.content}

def Defender(state: State):
    argument = state["argument"]
    Defender_prompt = "You are a Senior Defence Attorney, Your objective is to defend the person/idea/philosophy mentioned in the USER PROMPT logically and try to defend its integrity against all criticism despite your personal beliefs, You will be presenting your case in front of a JUDGE, do away with the salutations NO FLUFF"
    msg = llm.invoke(f"{Defender_prompt}, USER PROMPT : {state['user_prompt']}, {argument}")
    return {"counter_argument": msg.content}


def Judge(state: State):
    argument = state["argument"]
    counter_argument = state["counter_argument"]
    judge_prompt = "Critically examine the Prosecution Argument and Defence Counter-Argument, and come to a unbiased conclusion based on nothing but the arguments provided. Fuck your personal beliefs."
    msg = llm.invoke(f"system prompt: {judge_prompt}, argument: {argument}, counter_argument: {counter_argument}")
    return {"judgement": msg.content}


# WORKFLOW
parallel_builder = StateGraph(State)

parallel_builder.add_node("Prosecutor", Prosecutor)
parallel_builder.add_node("Defender", Defender)
parallel_builder.add_node("Judge", Judge)

parallel_builder.add_edge(START, "Prosecutor")
parallel_builder.add_edge(START, "Defender")
parallel_builder.add_edge("Prosecutor", "Judge")
parallel_builder.add_edge("Defender", "Judge")
parallel_builder.add_edge("Judge", END)
parallel_workflow = parallel_builder.compile()

#display(Image(parallel_workflow.get_graph().draw_mermaid_png()))

#INVOKE
state = parallel_workflow.invoke({
    "user_prompt": user_prompt,
    "messages": [],
    "argument": "",
    "counter_argument": "",
    "judgement": "",
    "feedback": "",
    "verdict": "",
})
print(state["judgement"])
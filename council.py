import os
import streamlit as st
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import Literal
from langgraph.pregel.debug import TypedDict
from pydantic import BaseModel, Field
from pydantic.v1.fields import FieldInfo as FieldInfoV1
from typing_extensions import Annotated

# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")

st.title("AI Council")

gemini_api_key = st.text_input(label="Enter your Gemini API key:", type="password")
model = st.text_input(label="Enter your Gemini model name:", value="gemini-2.5-flash")
user_prompt = st.text_input(label="Enter the user prompt:")

if st.button("Run Council"):
    if not gemini_api_key:
        st.warning("Please enter your API key to continue.")
    elif not model:
        st.warning("Please enter the model name.")
    elif not user_prompt:
        st.warning("Please enter the user prompt.")
    else:
        try:
            llm = ChatGoogleGenerativeAI(model=model, google_api_key=gemini_api_key)

            class State(TypedDict):
                user_prompt: str
                messages: Annotated[list[HumanMessage | AIMessage], add_messages]
                argument: str
                counter_argument: str
                judgement: str
                grade: str
                feedback: str
                FOF: str
                factual: str
                fiction: str
                verdict: str

            class feedback(BaseModel):
                grade: Literal["factual", "fiction"] = Field(
                    description="Decide If the Judgement is based in Fact or Fiction.",
                )
                feedback: str = Field(
                    description="If the Judgement is not funny, provide feedback to improve the Judgement.",
                )

            evaluator = llm.with_structured_output(feedback)

            # Nodes
            def Prosecutor(state: State):
                Prosecutor_prompt = "You are a Senior Prosecutor, Your objective is to find flaws with the person/idea/philosophy mentioned in the USER PROMPT logically and Prosecute the entity mentioned in the USER PROMPT no matter your personal beliefs. You will be presenting your case in front of a JUDGE, do away with the salutations. NO FLUFF"
                msg = llm.invoke(
                    Prosecutor_prompt + f" USER PROMPT: {state['user_prompt']}"
                )
                return {"argument": msg.content}

            def Defender(state: State):
                argument = state["argument"]
                Defender_prompt = "You are a Senior Defence Attorney, Your objective is to defend the person/idea/philosophy mentioned in the USER PROMPT logically and try to defend its integrity against all criticism despite your personal beliefs, You will be presenting your case in front of a JUDGE, do away with the salutations NO FLUFF"
                msg = llm.invoke(
                    f"{Defender_prompt}, USER PROMPT : {state['user_prompt']}, {argument}"
                )
                return {"counter_argument": msg.content}

            def Judge(state: State):
                argument = state["argument"]
                counter_argument = state["counter_argument"]
                judge_prompt = "Critically examine the Prosecution Argument and Defence Counter-Argument, and come to a unbiased conclusion based on nothing but the arguments provided. Fuck your personal beliefs."
                if state.get("feedback"):
                    msg = llm.invoke(
                        f"system prompt: {judge_prompt}+You Have been given some feedback work with that primarily., argument: {argument}, counter_argument: {counter_argument}, feedback: {state['feedback']}, Work on the Feedback given by the Jury and fix what ever mistakes are pointed out."
                    )
                else:
                    msg = llm.invoke(
                        f"system prompt: {judge_prompt}, argument: {argument}, counter_argument: {counter_argument}"
                    )
                return {"judgement": msg.content}

            def Jury(state: State):
                grade = evaluator.invoke(f"Grade the Judgement, {state['judgement']}")
                return {"FOF": grade.grade, "feedback": grade.feedback}

            def route_judgement(state: State):
                if state["FOF"] == "factual":
                    return "Accepted"
                elif state["FOF"] == "fiction":
                    return "Rejected + feedback"

            # WORKFLOW
            parallel_builder = StateGraph(State)

            parallel_builder.add_node("Prosecutor", Prosecutor)
            parallel_builder.add_node("Defender", Defender)
            parallel_builder.add_node("Judge", Judge)
            parallel_builder.add_node("Jury", Jury)

            parallel_builder.add_edge(START, "Prosecutor")
            parallel_builder.add_edge(START, "Defender")
            parallel_builder.add_edge("Prosecutor", "Judge")
            parallel_builder.add_edge("Defender", "Judge")
            parallel_builder.add_edge("Judge", "Jury")
            parallel_builder.add_conditional_edges(
                "Jury",
                route_judgement,
                {"Accepted": END, "Rejected + feedback": "Judge"},
            )
            parallel_workflow = parallel_builder.compile()

            # Graphically Depict the Nodes Connection
            # display(Image(parallel_workflow.get_graph().draw_mermaid_png()))

            # Invoke
            with st.spinner("The council is in session..."):
                state = parallel_workflow.invoke(
                    {
                        "user_prompt": user_prompt,
                        "messages": [],
                        "argument": "",
                        "counter_argument": "",
                        "grade": "",
                        "judgement": "",
                        "feedback": "",
                        "factual": "",
                        "fiction": "",
                        "FOF": "",
                        "verdict": "",
                    }
                )
                st.header("Judgement")
                st.write(state["judgement"])

        except Exception as e:
            st.error(f"An error occurred: {e}")
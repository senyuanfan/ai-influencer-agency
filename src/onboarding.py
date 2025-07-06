from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import asyncio
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the interview state
class InterviewState(TypedDict):
    messages: List[dict]
    question_count: int
    interview_complete: bool
    candidate_responses: List[str]
    interview_context: str
    current_question: str
    
# AI Interviewer System Prompt
from prompts import INTERVIEWER_PROMPT



def initialize_interview_node(state: InterviewState) -> InterviewState:
    """Initialize the interview with AI interviewer introduction"""
    print("\n" + "="*60)
    print("🤖 AI INTERVIEWER POWERED BY GPT-4.1")
    print("="*60)
    
    # Get AI introduction
    response = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {"role": "system", "content": "You are a professional AI interviewer. Introduce yourself briefly and explain that you'll be conducting an interview. Ask for the candidate's name and preferred role they're interviewing for. Keep it warm and professional."},
            {"role": "user", "content": "Please introduce yourself and start the interview."}
        ],
        max_tokens=200,
        temperature=0.7
    )
    
    ai_intro = response.choices[0].message.content
    print(f"\n🤖 AI Interviewer: {ai_intro}")
    
    state["messages"].append({"role": "assistant", "content": ai_intro})
    state["interview_context"] = "Interview starting - getting candidate information"
    
    return state

def get_candidate_info_node(state: InterviewState) -> InterviewState:
    """Get initial candidate information"""
    candidate_info = input("\n💬 Your response: ")
    
    state["messages"].append({"role": "user", "content": candidate_info})
    state["candidate_responses"].append(candidate_info)
    state["interview_context"] = f"Candidate info: {candidate_info}"
    
    print(f"\n✅ Thank you! Let's begin the interview.")
    
    return state

def ai_generate_question_node(state: InterviewState) -> InterviewState:
    """AI generates the next interview question"""
    
    # Build conversation context for the AI
    conversation_history = ""
    for msg in state["messages"]:
        role = "Interviewer" if msg["role"] == "assistant" else "Candidate"
        conversation_history += f"{role}: {msg['content']}\n"
    
    # Generate next question using GPT-4.1
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",  # Using GPT-4 Turbo as GPT-4.1 isn't available yet
        messages=[
            {"role": "system", "content": INTERVIEWER_PROMPT.format(
                context=state["interview_context"], 
                question_count=state["question_count"]
            )},
            {"role": "user", "content": f"Conversation so far:\n{conversation_history}\n\nGenerate the next interview question or indicate if complete."}
        ],
        max_tokens=300,
        temperature=0.8
    )
    
    ai_question = response.choices[0].message.content.strip()
    
    # Check if AI wants to complete the interview
    if "INTERVIEW_COMPLETE" in ai_question:
        state["interview_complete"] = True
        state["current_question"] = ""
        print(f"\n🎉 The AI interviewer has concluded the interview after {state['question_count']} questions.")
        return state
    
    state["current_question"] = ai_question
    state["question_count"] += 1
    
    print(f"\n📝 Question {state['question_count']}:")
    print(f"🤖 AI Interviewer: {ai_question}")
    print("\n⏳ Waiting for your response...")
    
    state["messages"].append({"role": "assistant", "content": ai_question})
    
    return state

def wait_for_response_node(state: InterviewState) -> InterviewState:
    """Human-in-the-loop: Wait for candidate response"""
    if state["interview_complete"]:
        return state
        
    response = input("\n💬 Your response: ")
    
    state["candidate_responses"].append(response)
    state["messages"].append({"role": "user", "content": response})
    
    # Update interview context with latest response
    state["interview_context"] += f" | Latest response: {response[:100]}..."
    
    print(f"\n✅ Response recorded!")
    
    return state

def ai_interview_summary_node(state: InterviewState) -> InterviewState:
    """AI generates interview summary and feedback"""
    
    # Build full conversation for summary
    full_conversation = ""
    for msg in state["messages"]:
        role = "Interviewer" if msg["role"] == "assistant" else "Candidate"
        full_conversation += f"{role}: {msg['content']}\n\n"
    
    # Get AI summary
    summary_response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": """You are an AI interviewer providing a professional interview summary. 

Analyze the conversation and provide:
1. Key strengths demonstrated by the candidate
2. Areas of interest or concern
3. Overall impression
4. Recommendation (Strong Yes, Yes, Maybe, No)

Be objective, constructive, and professional."""},
            {"role": "user", "content": f"Please analyze this interview conversation and provide a summary:\n\n{full_conversation}"}
        ],
        max_tokens=500,
        temperature=0.3
    )
    
    ai_summary = summary_response.choices[0].message.content
    
    print("\n" + "="*60)
    print("📊 AI INTERVIEW ANALYSIS")
    print("="*60)
    print(f"🤖 AI Interviewer Analysis:\n{ai_summary}")
    print("\n" + "="*60)
    print(f"📈 Interview Statistics:")
    print(f"   • Total Questions Asked: {state['question_count']}")
    print(f"   • Total Responses: {len(state['candidate_responses'])}")
    print(f"   • Interview Completion: {'✅ Complete' if state['interview_complete'] else '❌ Incomplete'}")
    print("="*60)
    print("Thank you for participating in the AI-powered interview!")
    print("="*60)
    
    return state

def should_continue(state: InterviewState) -> str:
    """Conditional edge to determine next step"""
    if state["interview_complete"]:
        return "summary"
    else:
        return "generate_question"

def create_ai_interview_graph():
    """Create the AI-powered LangGraph interview workflow"""
    
    # Initialize the state graph
    workflow = StateGraph(InterviewState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_interview_node)
    workflow.add_node("get_info", get_candidate_info_node)
    workflow.add_node("generate_question", ai_generate_question_node)
    workflow.add_node("wait_response", wait_for_response_node)
    workflow.add_node("summary", ai_interview_summary_node)
    
    # Define the flow
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "get_info")
    workflow.add_edge("get_info", "generate_question")
    workflow.add_edge("generate_question", "wait_response")
    
    # Conditional edge based on completion status
    workflow.add_conditional_edges(
        "wait_response",
        should_continue,
        {
            "generate_question": "generate_question",
            "summary": "summary"
        }
    )
    
    workflow.add_edge("summary", END)
    
    # Compile with memory for persistence
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory, interrupt_before=["wait_response"])
    
    return app

async def run_ai_interview():
    """Run the AI-powered interview agent"""
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key in a .env file or environment variable.")
        return
    
    app = create_ai_interview_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "question_count": 0,
        "interview_complete": False,
        "candidate_responses": [],
        "interview_context": "",
        "current_question": ""
    }
    
    # Configuration for the run
    config = {"configurable": {"thread_id": "ai_interview_session_1"}}
    
    # Run the interview with human-in-the-loop
    try:
        async for event in app.astream(initial_state, config=config):
            # Check if we need human input
            if app.get_state(config).next == ("wait_response",):
                # Resume after human input
                await app.ainvoke(None, config=config)
    except Exception as e:
        print(f"❌ Error during AI interview: {e}")

def main():
    """Main function to start the AI interview"""
    print("🚀 Starting AI-Powered Interview Agent with GPT-4.1...")
    print("🔑 Make sure your OPENAI_API_KEY is set in your .env file!")
    
    try:
        # Run the async AI interview
        asyncio.run(run_ai_interview())
    except KeyboardInterrupt:
        print("\n\n❌ Interview cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error during interview: {e}")

if __name__ == "__main__":
    main()

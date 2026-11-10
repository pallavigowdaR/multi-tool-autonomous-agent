"""
agent.py - Multi-Tool Autonomous AI Agent

Implements an autonomous agent using LangChain that selects and invokes tools:
- Real-time DuckDuckGo Web Search
- Python Math Calculator
- Local File Reader

Visualizes the explicit reasoning chain:
  [THOUGHT] -> [ACTION] -> [OBSERVATION] -> [FINAL ANSWER]
"""

import os
import sys
from typing import Dict, Any, List
from dotenv import load_dotenv

# LangChain imports
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# Local tools
from tools import get_all_tools

# Color formatting for terminal
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    CYAN = Fore.CYAN
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    MAGENTA = Fore.MAGENTA
    RED = Fore.RED
    RESET = Style.RESET_ALL
    BOLD = Style.BRIGHT
except ImportError:
    CYAN = GREEN = YELLOW = MAGENTA = RED = RESET = BOLD = ""

load_dotenv()


# ============================================================================
# Reasoning Chain Callback Handler: Displays Thought -> Action -> Observation
# ============================================================================
class ReasoningChainCallbackHandler(BaseCallbackHandler):
    """
    Real-time terminal callback to stream and display the agent's reasoning loop:
      - Thought / Planning
      - Action (Tool invocation & inputs)
      - Observation (Tool return value)
    """

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        pass

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        tool_name = serialized.get("name", "UnknownTool")
        print(f"\n{YELLOW}{BOLD}⚡ [ACTION / TOOL INVOKED]{RESET}")
        print(f"  {BOLD}Tool:{RESET} {CYAN}{tool_name}{RESET}")
        print(f"  {BOLD}Input:{RESET} {input_str}")

    def on_tool_end(self, output: str, **kwargs) -> None:
        # Display first 500 characters of observation to keep terminal readable
        cleaned_output = str(output).strip()
        display_output = (cleaned_output[:500] + "... [truncated]") if len(cleaned_output) > 500 else cleaned_output
        print(f"\n{MAGENTA}{BOLD}🔍 [OBSERVATION / TOOL RESULT]{RESET}")
        print(f"  {display_output}")

    def on_tool_error(self, error: BaseException, **kwargs) -> None:
        print(f"\n{RED}{BOLD}❌ [TOOL ERROR]{RESET} {error}")


def initialize_llm():
    """
    Select and initialize the configured LLM provider:
    Supports OpenAI, Google Gemini, and Groq.
    """
    provider = os.getenv("LLM_PROVIDER", "").lower()
    
    # 1. Check OpenAI
    if provider == "openai" or os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"{GREEN}✓ Using LLM Provider: OpenAI ({model_name}){RESET}")
        return ChatOpenAI(model=model_name, temperature=0, streaming=True)

    # 2. Check Google Gemini
    if provider == "gemini" or os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        print(f"{GREEN}✓ Using LLM Provider: Google Gemini ({model_name}){RESET}")
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)

    # 3. Check Groq
    if provider == "groq" or os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        print(f"{GREEN}✓ Using LLM Provider: Groq ({model_name}){RESET}")
        return ChatGroq(model=model_name, temperature=0)

    # Missing API keys
    print(f"\n{YELLOW}{BOLD}Notice: No LLM API key detected in environment!{RESET}")
    print(f"You can configure an API key in your `.env` file:")
    print(f"  - OPENAI_API_KEY=your_key (for OpenAI gpt-4o-mini)")
    print(f"  - GOOGLE_API_KEY=your_key (for Google Gemini)")
    print(f"  - GROQ_API_KEY=your_key (for Groq Llama 3.3)")
    print(f"\n{CYAN}Would you like to run the Offline Multi-Step Simulation Demo? (Y/n){RESET}: ", end="")
    choice = input().strip().lower()
    if choice in ["", "y", "yes"]:
        run_simulated_demo()
        sys.exit(0)
    else:
        print(f"\nPlease configure `.env` and rerun. Exiting.\n")
        sys.exit(0)


def run_simulated_demo():
    """
    Runs a live demonstration of the multi-tool reasoning chain without requiring an LLM API key.
    Executes actual tools (file reading, math calculation, real-time web search) sequentially.
    """
    import time
    from tools import local_file_reader, python_calculator, web_search

    sample_query = (
        "Please read 'data/company_sales.txt', compute the percentage revenue growth from Q1 to Q2, "
        "and search DuckDuckGo for SaaS industry average growth to compare."
    )

    print(f"\n{'=' * 75}")
    print(f"{CYAN}{BOLD}📥 RUNNING SIMULATED COMPOUND QUERY:{RESET}")
    print(f"  {sample_query}")
    print(f"{'=' * 75}\n")
    time.sleep(1)

    print(f"{CYAN}{BOLD}🧠 [THOUGHT 1]{RESET}: The user needs to analyze quarterly revenue growth. First, I need to read the local data file 'data/company_sales.txt' to retrieve the revenue values for Q1 and Q2.")
    time.sleep(1)

    print(f"\n{YELLOW}{BOLD}⚡ [ACTION 1 / TOOL INVOKED]{RESET}")
    print(f"  {BOLD}Tool:{RESET} {CYAN}local_file_reader{RESET}")
    print(f"  {BOLD}Input:{RESET} data/company_sales.txt")

    file_result = local_file_reader.invoke({"file_path": "data/company_sales.txt"})
    time.sleep(1)
    
    print(f"\n{MAGENTA}{BOLD}🔍 [OBSERVATION 1]{RESET}")
    lines = [line.strip() for line in file_result.splitlines() if "Revenue:" in line]
    print(f"  Extracted records: {', '.join(lines[:2])}")

    print(f"\n{CYAN}{BOLD}🧠 [THOUGHT 2]{RESET}: Q1 Revenue is $4,500,000 USD and Q2 Revenue is $6,300,000 USD. Now I will use the Python math calculator to compute the exact quarter-over-quarter percentage growth using the formula: ((Q2 - Q1) / Q1) * 100.")
    time.sleep(1)

    calc_expr = "((6300000 - 4500000) / 4500000) * 100"
    print(f"\n{YELLOW}{BOLD}⚡ [ACTION 2 / TOOL INVOKED]{RESET}")
    print(f"  {BOLD}Tool:{RESET} {CYAN}python_calculator{RESET}")
    print(f"  {BOLD}Input:{RESET} {calc_expr}")

    calc_result = python_calculator.invoke({"expression": calc_expr})
    time.sleep(1)

    print(f"\n{MAGENTA}{BOLD}🔍 [OBSERVATION 2]{RESET}")
    print(f"  {calc_result}")

    print(f"\n{CYAN}{BOLD}🧠 [THOUGHT 3]{RESET}: The computed growth is 40.0%. Next, I must query real-time DuckDuckGo web search to find typical SaaS industry quarterly growth benchmarks for comparison.")
    time.sleep(1)

    search_query = "average SaaS quarterly revenue growth rate benchmark"
    print(f"\n{YELLOW}{BOLD}⚡ [ACTION 3 / TOOL INVOKED]{RESET}")
    print(f"  {BOLD}Tool:{RESET} {CYAN}web_search{RESET}")
    print(f"  {BOLD}Input:{RESET} {search_query}")

    search_result = web_search.invoke({"query": search_query})
    preview_search = search_result.split("\n\n")[0] if "\n\n" in search_result else search_result[:300]
    
    print(f"\n{MAGENTA}{BOLD}🔍 [OBSERVATION 3]{RESET}")
    print(f"  {preview_search}")

    print(f"\n{CYAN}{BOLD}🧠 [THOUGHT 4]{RESET}: I have collected all necessary facts from the file, exact math calculations, and real-time market data. I will now synthesize the final response.")
    time.sleep(1)

    print(f"\n{'=' * 75}")
    print(f"{GREEN}{BOLD}🏁 [FINAL SYNTHESIZED ANSWER]{RESET}")
    print(f"{'=' * 75}")
    print(f"""
### Financial Performance & Market Comparison Summary

1. **Internal Data Extraction (`local_file_reader`)**:
   - **Q1 Revenue**: $4,500,000 USD
   - **Q2 Revenue**: $6,300,000 USD

2. **Growth Calculation (`python_calculator`)**:
   - Computed Quarter-over-Quarter Growth: **+40.0%**

3. **Market Benchmark Comparison (`web_search`)**:
   - Standard healthy SaaS companies typically experience quarterly growth between **5% to 15%**.

4. **Strategic Conclusion**:
   - ACME Tech Corp's 40.0% QoQ growth significantly outpaced general industry averages, driven by European market expansion.
""")
    print(f"{'=' * 75}\n")


def create_autonomous_agent():
    """Constructs the agent prompt, binds tools, and builds the AgentExecutor."""
    llm = initialize_llm()
    tools = get_all_tools()

    system_prompt = (
        "You are an autonomous AI research and computation agent capable of using tools "
        "to solve complex multi-step problems.\n\n"
        "Guidelines:\n"
        "1. Break compound questions down into logical sequential steps.\n"
        "2. When solving math or percentage questions, DO NOT guess: use the `python_calculator`.\n"
        "3. When requested to inspect local files, use the `local_file_reader`.\n"
        "4. When real-time, external, or current data is required, use `web_search`.\n"
        "5. Always articulate your reasoning before deciding which tool to invoke.\n"
        "6. Combine observations from all tools to produce a coherent, structured final answer."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )
    return executor


def run_query(agent_executor: AgentExecutor, user_query: str):
    """Executes a single compound query through the agent with reasoning traces."""
    print(f"\n{'=' * 75}")
    print(f"{CYAN}{BOLD}📥 USER QUERY:{RESET} {user_query}")
    print(f"{'=' * 75}\n")
    print(f"{CYAN}{BOLD}🧠 [THOUGHT & REASONING CHAIN INITIATED]{RESET}")

    callback_handler = ReasoningChainCallbackHandler()
    
    try:
        response = agent_executor.invoke(
            {"input": user_query},
            config={"callbacks": [callback_handler]}
        )

        final_output = response.get("output", "No response generated.")
        
        print(f"\n{'=' * 75}")
        print(f"{GREEN}{BOLD}🏁 [FINAL SYNTHESIZED ANSWER]{RESET}")
        print(f"{'=' * 75}")
        print(f"\n{final_output}\n")
        return final_output

    except Exception as e:
        print(f"\n{RED}{BOLD}Agent execution failed:{RESET} {str(e)}")
        return None


def interactive_console():
    """Starts the interactive console loop for autonomous multi-step problem solving."""
    print(f"{BOLD}{CYAN}==============================================================={RESET}")
    print(f"{BOLD}{CYAN}        🤖 MULTI-TOOL AUTONOMOUS AI AGENT CONSOLE             {RESET}")
    print(f"{BOLD}{CYAN}==============================================================={RESET}")
    print(f"Available Tools:")
    print(f"  1. 🌐 Web Search (DuckDuckGo)")
    print(f"  2. 🧮 Python Math Calculator")
    print(f"  3. 📂 Local File Reader\n")
    print(f"Commands:")
    print(f"  - Type 'demo' to run a compound multi-step test query (requires API credits).")
    print(f"  - Type 'simulate' to run the multi-step reasoning demo without API keys.")
    print(f"  - Type 'exit' or 'quit' to terminate.\n")

    agent_executor = create_autonomous_agent()

    while True:
        try:
            user_input = input(f"{BOLD}Agent Prompt > {RESET}").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print(f"\n{GREEN}Goodbye! 👋{RESET}")
                break

            if user_input.lower() in ["simulate", "sim", "offline"]:
                run_simulated_demo()
                continue

            if user_input.lower() == "demo":
                user_input = (
                    "Please read 'data/company_sales.txt', retrieve the revenue figures for "
                    "Q1 and Q2, use the calculator to compute the exact percentage growth "
                    "from Q1 to Q2, and then search the web for 'average SaaS quarterly growth rate' "
                    "to compare how our company performed against the market standard."
                )

            run_query(agent_executor, user_input)

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{YELLOW}Session ended by user.{RESET}")
            break


if __name__ == "__main__":
    interactive_console()


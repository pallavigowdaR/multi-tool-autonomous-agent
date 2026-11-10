"""
tools.py - Autonomous AI Agent Tool Definitions

This module provides the three core tools required by the autonomous agent:
1. web_search: Real-time search using DuckDuckGo (no API key required).
2. python_calculator: Safe Python math evaluator for compound calculations.
3. local_file_reader: Local file reader for reading text, csv, json, and markdown files.
"""

import os
import math
import warnings
from typing import List
from langchain_core.tools import tool, BaseTool

# Suppress deprecation/runtime warnings from third-party search libraries
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# TOOL 1: DuckDuckGo Web Search Tool
# ============================================================================
@tool
def web_search(query: str) -> str:
    """
    Search the public web in real-time for up-to-date facts, news, and external information.
    Input should be a targeted search query string.
    Example: 'current tech inflation rate 2025' or 'latest SpaceX launch date'
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
            
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return f"No search results found for query: '{query}'"
            
            formatted_results = []
            for idx, r in enumerate(results, 1):
                title = r.get("title", "No Title")
                snippet = r.get("body", "No Snippet")
                link = r.get("href", "")
                formatted_results.append(f"[{idx}] {title}\nSnippet: {snippet}\nURL: {link}")
            
            return "\n\n".join(formatted_results)
    except Exception as e:
        # Fallback to langchain_community DuckDuckGoSearchRun if available
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            search = DuckDuckGoSearchRun()
            return search.run(query)
        except Exception as fallback_err:
            return f"Web search error: {str(e)} (Fallback error: {str(fallback_err)})"


# ============================================================================
# TOOL 2: Python Math Calculator
# ============================================================================
@tool
def python_calculator(expression: str) -> str:
    """
    Evaluate mathematical expressions or Python calculations with high precision.
    Use this tool whenever you need exact calculations, percentages, ratios,
    square roots, powers, or statistical math.
    Input must be a valid Python math expression.
    Examples:
      - '((6300000 - 4500000) / 4500000) * 100' (Percentage increase)
      - 'math.sqrt(144) + 15 * 8'
      - 'math.log(100, 10)'
    """
    # Safe namespace restricted to math and built-in numerical functions
    safe_dict = {
        "math": math,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "int": int,
        "float": float,
    }
    
    clean_expr = expression.strip().replace("`", "")
    
    try:
        # Evaluate within isolated safe dictionary
        result = eval(clean_expr, {"__builtins__": {}}, safe_dict)
        return f"Calculation Result: {result}"
    except Exception as err:
        return f"Math calculation failed for '{clean_expr}'. Error: {str(err)}. Ensure valid Python syntax."


# ============================================================================
# TOOL 3: Local File Reader
# ============================================================================
@tool
def local_file_reader(file_path: str) -> str:
    """
    Read and extract the textual contents of a local file on the system.
    Supports .txt, .csv, .json, .md, and .log files.
    Input should be a relative or absolute file path.
    Example: 'data/company_sales.txt'
    """
    # Normalize path
    target_path = os.path.normpath(file_path.strip().strip("'\""))
    
    if not os.path.exists(target_path):
        # Check relative to project root or workspace
        workspace_path = os.path.join(os.getcwd(), target_path)
        if os.path.exists(workspace_path):
            target_path = workspace_path
        else:
            return f"Error: File '{file_path}' does not exist on disk."

    if os.path.isdir(target_path):
        entries = os.listdir(target_path)
        return f"Path '{file_path}' is a directory containing:\n" + "\n".join(f"- {e}" for e in entries)

    try:
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        
        char_count = len(content)
        lines = content.splitlines()
        preview_header = f"--- File: {os.path.basename(target_path)} ({len(lines)} lines, {char_count} chars) ---\n"
        
        # Guard against gigantic files (>50,000 characters) by truncating
        if char_count > 50000:
            content = content[:50000] + f"\n\n[TRUNCATED: File exceeds 50,000 characters]"
            
        return preview_header + content
    except Exception as err:
        return f"Failed to read file '{file_path}': {str(err)}"


def get_all_tools() -> List[BaseTool]:
    """Return list of all configured tools for agent binding."""
    return [web_search, python_calculator, local_file_reader]


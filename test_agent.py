"""
test_agent.py - Automated Unit & Integration Tests for Multi-Tool Agent

Tests individual tool functions:
1. Local file reading
2. Python math calculation
3. DuckDuckGo web search
4. Agent tool binding sanity check
"""

import os
import unittest
from tools import local_file_reader, python_calculator, web_search, get_all_tools


class TestAgentTools(unittest.TestCase):

    def test_file_reader_existing_file(self):
        """Verify local file reader successfully reads sample data."""
        sample_path = os.path.join("data", "company_sales.txt")
        result = local_file_reader.invoke({"file_path": sample_path})
        self.assertIn("ACME TECH CORP", result)
        self.assertIn("Revenue: $4,500,000 USD", result)
        print("  ✓ test_file_reader_existing_file passed")

    def test_file_reader_missing_file(self):
        """Verify proper error handling when reading a missing file."""
        result = local_file_reader.invoke({"file_path": "non_existent_file_12345.txt"})
        self.assertIn("does not exist", result.lower())
        print("  ✓ test_file_reader_missing_file passed")

    def test_python_calculator_arithmetic(self):
        """Verify calculator evaluates arithmetic and percentages accurately."""
        # Q1: 4.5M, Q2: 6.3M -> ((6.3 - 4.5) / 4.5) * 100 = 40.0%
        result = python_calculator.invoke({"expression": "((6300000 - 4500000) / 4500000) * 100"})
        self.assertIn("40.0", result)
        print("  ✓ test_python_calculator_arithmetic passed")

    def test_python_calculator_math_module(self):
        """Verify calculator handles math functions like sqrt and pow."""
        result = python_calculator.invoke({"expression": "math.sqrt(144) + pow(2, 3)"})
        self.assertIn("20.0", result)
        print("  ✓ test_python_calculator_math_module passed")

    def test_web_search_connectivity(self):
        """Verify DuckDuckGo search executes and returns structured results."""
        query = "Python programming language official website"
        result = web_search.invoke({"query": query})
        self.assertTrue(len(result) > 20)
        self.assertTrue("python" in result.lower())
        print("  ✓ test_web_search_connectivity passed")

    def test_tools_binding(self):
        """Verify all 3 tools are exposed in get_all_tools()."""
        tools = get_all_tools()
        self.assertEqual(len(tools), 3)
        tool_names = [t.name for t in tools]
        self.assertIn("web_search", tool_names)
        self.assertIn("python_calculator", tool_names)
        self.assertIn("local_file_reader", tool_names)
        print("  ✓ test_tools_binding passed")


if __name__ == "__main__":
    print("\n================ RUNNING TOOL SUITE TESTS ================")
    unittest.main(verbosity=2)


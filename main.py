#!/usr/bin/env python3
"""
Study Buddy AI Agent - Final Integrated System
"""

import os
import argparse
from dotenv import load_dotenv
from utils.config_loader import ConfigLoader
from agents.rag_agent import RAGAgent
from agents.formatter_agent import FormatterAgent

def setup_test_environment():
    """Set up test directories and sample data only if they don't exist"""
    os.makedirs("./test_courses", exist_ok=True)
    os.makedirs("./test_outputs", exist_ok=True)
    
    # Only create sample data if no courses exist
    if not os.listdir("./test_courses"):
        os.makedirs("./test_courses/CS101", exist_ok=True)
        
        # Create sample lecture file
        sample_lecture = """
        # CS101: Data Structures and Algorithms
        
        ## Graph Traversal Algorithms
        
        Depth-First Search (DFS) and Breadth-First Search (BFS) are fundamental graph traversal algorithms.
        
        DFS explores as far as possible along each branch before backtracking. It uses a stack data structure.
        
        BFS explores all neighbors at the present depth before moving deeper. It uses a queue data structure.
        
        Time Complexity: Both algorithms have time complexity O(V + E) where V is vertices and E is edges.
        """
        
        with open("./test_courses/CS101/lecture1.txt", "w") as f:
            f.write(sample_lecture)
        
        # Create sample homework solution
        sample_solution = """
        Problem 1: Explain DFS and BFS.
        
        Solution:
        DFS goes deep first using stack. BFS goes wide first using queue.
        
        Problem 2: Time complexity.
        
        Both are O(V + E) for vertices V and edges E.
        """
        
        with open("./test_courses/CS101/solution.txt", "w") as f:
            f.write(sample_solution)

def get_available_courses():
    """Get all available courses from the test_courses directory"""
    courses = []
    if os.path.exists("./test_courses"):
        for item in os.listdir("./test_courses"):
            course_path = os.path.join("./test_courses", item)
            if os.path.isdir(course_path):
                courses.append({
                    'name': item,
                    'path': course_path
                })
    return courses

def demo_rag_agent():
    """Demo the RAG agent with all available courses"""
    config = ConfigLoader()
    rag_agent = RAGAgent(config)
    
    courses = get_available_courses()
    if not courses:
        print(" No courses found in ./test_courses/")
        return
    
    print(" Study Buddy - RAG Agent Demo")
    print(f"Found {len(courses)} courses: {[c['name'] for c in courses]}")
    
    # Build knowledge base for all courses
    for course in courses:
        print(f"Building knowledge base for {course['name']}...")
        try:
            rag_agent.build_knowledge_base(course['path'], course['name'])
            print(f" {course['name']} knowledge base built successfully!")
        except Exception as e:
            print(f" Error building knowledge base for {course['name']}: {e}")
    
    # Interactive querying
    while True:
        question = input("\n Ask a question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        print("\n Thinking...")
        try:
            result = rag_agent.query(question)
            print(f"\nAnswer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except Exception as e:
            print(f" Error: {e}")

def demo_formatter_agent():
    """Demo the Formatter agent"""
    config = ConfigLoader()
    
    print("\n Study Buddy - Formatter Agent Demo")
    formatter_agent = FormatterAgent(config)
    
    # Read sample solution
    with open("./test_courses/CS101/solution.txt", "r") as f:
        sample_solution = f.read()
    
    print("Converting text solution to LaTeX...")
    result = formatter_agent.text_to_latex(sample_solution, "homework")
    
    if result["is_valid"]:
        print(" LaTeX conversion successful!")
        print(f"Model used: {result['model_used']}")
        print("\nGenerated LaTeX:")
        print("=" * 50)
        print(result["latex_code"])
        print("=" * 50)
        
        # Save to file
        with open("./test_outputs/solution.tex", "w") as f:
            f.write(result["latex_code"])
        print("Saved to ./test_outputs/solution.tex")
    else:
        print(" LaTeX conversion failed:")
        print(result.get("error", "Unknown error"))

def demo_integrated_workflow():
    """Demo integrated RAG + Formatter workflow"""
    config = ConfigLoader()
    
    print("\n Study Buddy - Integrated Workflow Demo")
    print("=" * 50)
    
    # Initialize both agents
    rag_agent = RAGAgent(config)
    formatter_agent = FormatterAgent(config)
    
    # Build knowledge base
    rag_agent.build_knowledge_base("./test_courses/CS101", "CS101")
    
    # Ask a question and format the answer
    question = "What is the time complexity of DFS and BFS?"
    
    print(f" Question: {question}")
    print("\n Getting answer from course materials...")
    
    rag_result = rag_agent.query(question)
    answer = rag_result["answer"]
    
    print(f" Answer: {answer}")
    
    print("\n Converting answer to LaTeX...")
    latex_result = formatter_agent.text_to_latex(answer, "explanation")
    
    if latex_result["is_valid"]:
        print(" Answer converted to LaTeX!")
        print("\n Formatted Explanation:")
        print("=" * 40)
        print(latex_result["latex_code"])
        print("=" * 40)
    else:
        print(" LaTeX conversion failed")

def main():
    parser = argparse.ArgumentParser(description="Study Buddy AI Agent")
    parser.add_argument("--mode", choices=["rag", "formatter", "integrated", "all"], default="all", 
                       help="Choose which component to demo")
    
    args = parser.parse_args()
    
    # Setup test environment
    setup_test_environment()
    
    # Check for GROQ API key if using RAG
    if args.mode in ["rag", "integrated", "all"] and not os.getenv("GROQ_API_KEY"):
        print("  GROQ_API_KEY not set. RAG agent will use Groq for inference.")
        print("   Set it with: export GROQ_API_KEY=your_key_here")
    
    # Run demos based on mode
    if args.mode in ["rag", "all"]:
        demo_rag_agent()
    
    if args.mode in ["formatter", "all"]:
        demo_formatter_agent()
    
    if args.mode in ["integrated", "all"]:
        demo_integrated_workflow()

if __name__ == "__main__":
    main()

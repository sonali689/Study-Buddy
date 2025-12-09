#!/usr/bin/env python3
"""
Study Buddy - Simple Gradio Interface
"""

import gradio as gr
import os
import sys
from pathlib import Path
import tempfile

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_loader import ConfigLoader
from agents.rag_agent import RAGAgent
from agents.formatter_agent import FormatterAgent

class StudyBuddyApp:
    def __init__(self):
        self.config = ConfigLoader()
        self.rag_agent = None
        self.formatter_agent = None
        self.current_course = None
        
    def initialize_agents(self):
        """Initialize AI agents"""
        try:
            self.rag_agent = RAGAgent(self.config)
            self.formatter_agent = FormatterAgent(self.config)
            return True
        except Exception as e:
            print(f"Error initializing agents: {e}")
            return False

# Create app instance
app = StudyBuddyApp()

def build_knowledge_base(course_name, files):
    """Build knowledge base from uploaded files"""
    if not course_name.strip():
        return " Please enter a course name"
    
    if not files:
        return " Please upload at least one PDF or text file"
    
    try:
        # Create course directory
        course_dir = Path("courses") / course_name
        course_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded files - FIXED: Use file content properly
        for file in files:
            file_path = course_dir / os.path.basename(file.name)
            # Copy the uploaded file to our directory
            with open(file_path, "wb") as f:
                with open(file.name, "rb") as source_file:
                    f.write(source_file.read())
        
        # Initialize agents if needed
        if not app.rag_agent:
            app.initialize_agents()
        
        # Build knowledge base
        app.rag_agent.build_knowledge_base(str(course_dir), course_name)
        app.current_course = course_name
        
        return f" Success! Built knowledge base for '{course_name}' with {len(files)} files"
        
    except Exception as e:
        return f" Error: {str(e)}"

def ask_question(question):
    """Ask a question about course materials"""
    if not app.current_course:
        return "Please build a knowledge base first", ""
    
    if not question.strip():
        return "Please enter a question", ""
    
    try:
        result = app.rag_agent.query(question)
        sources = "\n".join([f"• {source}" for source in result["sources"]])
        return result["answer"], sources
    except Exception as e:
        return f"Error: {str(e)}", ""

def format_to_latex(text, doc_type):
    """Convert text to LaTeX"""
    if not text.strip():
        return "Please enter some text to convert", ""
    
    try:
        if not app.formatter_agent:
            app.initialize_agents()
        
        result = app.formatter_agent.text_to_latex(text, doc_type)
        
        if result["is_valid"]:
            return result["latex_code"], " Successfully converted to LaTeX!"
        else:
            return "", f" Conversion failed: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return "", f" Error: {str(e)}"

def create_demo():
    """Create the simplified Gradio interface"""
    
    # Initialize agents
    app.initialize_agents()
    
    with gr.Blocks(
        title="Study Buddy",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1000px;
            margin: auto;
        }
        .success { color: green; }
        .error { color: red; }
        """
    ) as demo:
        gr.Markdown("#  Study Buddy")
        gr.Markdown("AI-powered study assistant for course materials and LaTeX formatting")
        
        with gr.Tabs():
            # Tab 1: Course Setup
            with gr.TabItem(" Setup Course"):
                gr.Markdown("### Upload your course materials")
                
                with gr.Row():
                    with gr.Column():
                        course_name = gr.Textbox(
                            label="Course Name",
                            placeholder="e.g., CS101-Algorithms",
                            info="Give your course a name"
                        )
                        
                        file_upload = gr.File(
                            label="Upload Course Materials",
                            file_types=[".pdf", ".txt"],
                            file_count="multiple",
                            type="filepath"  # This returns file paths instead of file objects
                        )
                        
                        build_btn = gr.Button(" Build Knowledge Base", variant="primary")
                    
                    with gr.Column():
                        build_status = gr.Markdown("### Status: No course built yet")
                
                build_btn.click(
                    fn=build_knowledge_base,
                    inputs=[course_name, file_upload],
                    outputs=build_status
                )
            
            # Tab 2: Ask Questions
            with gr.TabItem(" Ask Questions"):
                gr.Markdown("### Ask questions about your course materials")
                
                with gr.Row():
                    with gr.Column():
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., What is the time complexity of DFS?",
                            lines=3
                        )
                        ask_btn = gr.Button(" Get Answer", variant="primary")
                    
                    with gr.Column():
                        answer_output = gr.Textbox(
                            label="Answer",
                            lines=5,
                            interactive=False
                        )
                        
                        sources_output = gr.Textbox(
                            label="Sources",
                            lines=2,
                            interactive=False
                        )
                
                ask_btn.click(
                    fn=ask_question,
                    inputs=question_input,
                    outputs=[answer_output, sources_output]
                )
            
            # Tab 3: Format LaTeX
            with gr.TabItem(" Format LaTeX"):
                gr.Markdown("### Convert text solutions to LaTeX")
                
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Your Solution",
                            placeholder="e.g., Solve x^2 + 2x + 1 = 0. Solution: x = -1",
                            lines=5
                        )
                        
                        doc_type = gr.Dropdown(
                            choices=["homework", "algorithm", "proof", "general"],
                            label="Document Type",
                            value="homework"
                        )
                        
                        format_btn = gr.Button(" Convert to LaTeX", variant="primary")
                    
                    with gr.Column():
                        latex_output = gr.Code(
                            label="Generated LaTeX",
                            language="latex",
                            lines=8
                        )
                        
                        format_status = gr.Markdown()
                
                format_btn.click(
                    fn=format_to_latex,
                    inputs=[text_input, doc_type],
                    outputs=[latex_output, format_status]
                )
    
    return demo

def main():
    """Launch the application"""
    # Create necessary directories
    Path("courses").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    
    # Create and launch demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )

if __name__ == "__main__":
    main()

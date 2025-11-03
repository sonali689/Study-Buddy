```markdown
# Study Buddy - Setup Instructions

## Quick Start Guide

Follow these simple steps to run the Study Buddy application:

### 1. Clone the Repository
```bash
git clone <your-repo-address>
cd DS-IntershipTask
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Models
```bash
gdown models
unzip models.zip
```

### 4. Set API Key
```bash
export GROQ_API_KEY="your-api-key-here"
```

*Note: Replace "your-api-key-here" with your actual Groq API key*

### 5. Launch the App
```bash
python app_gradio.py
```

### 6. Access the Application
- The app will open at: `http://localhost:7860`
- You can share the public URL provided in the terminal

## What You'll Get
- Upload course materials (PDF/TXT)
- Ask questions about your course content
- Convert text to LaTeX formatting
```

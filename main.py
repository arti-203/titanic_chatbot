import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for FastAPI
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
import os

# --- 1. Data Loading ---
# Load Titanic dataset
TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic/train.csv"
df = pd.read_csv(TITANIC_URL)

# Data Cleaning
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna('S', inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# --- 2. Custom Tools for LangChain ---

def analyze_titanic(query: str) -> str:
    """Answers statistical questions using Pandas."""
    try:
        q = query.lower()
        if "percentage" in q and "male" in q:
            res = (df['Sex'] == 'male').sum() / len(df) * 100
            return f"{(res):.2f}%"
        elif "percentage" in q and "female" in q:
            res = (df['Sex'] == 'female').sum() / len(df) * 100
            return f"{(res):.2f}%"
        elif "average" in q and "fare" in q:
            return f"${df['Fare'].mean():.2f}"
        elif "average" in q and "age" in q:
            return f"{df['Age'].mean():.2f} years"
        elif "embark" in q:
            return df['Embarked'].value_counts().to_string()
        elif "survived" in q:
            return f"Total survived: {df['Survived'].sum()} out of {len(df)}"
        elif "count" in q and "passenger" in q:
            return f"Total passengers: {len(df)}"
        return "I can calculate that. Please ask a specific question like 'Average fare' or 'How many males'."
    except Exception as e:
        return f"Error: {str(e)}"

def visualize_titanic(query: str) -> str:
    """Generates charts and returns base64 string."""
    plt.figure(figsize=(10, 6))
    q = query.lower()
    
    try:
        if "histogram" in q and "age" in q:
            sns.histplot(df['Age'], kde=True, color="skyblue")
            plt.title("Distribution of Passenger Ages")
        elif "histogram" in q and "fare" in q:
            sns.histplot(df['Fare'], kde=True, color="green")
            plt.title("Distribution of Ticket Fares")
        elif "embark" in q:
            sns.countplot(x='Embarked', data=df, palette="Set2")
            plt.title("Passengers by Port of Embarkation")
        elif "surviv" in q and "gender" in q:
            sns.countplot(x='Sex', hue='Survived', data=df, palette="Set1")
            plt.title("Survival by Gender")
        else:
            sns.countplot(x='Pclass', data=df, palette="muted")
            plt.title("Passenger Count by Class")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(buf)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64
    except Exception as e:
        return f"Error creating chart: {str(e)}"

# --- 3. Initialize Agent ---
# IMPORTANT: Set your OpenAI Key in environment variable OPENAI_API_KEY
# Or replace the line below with: os.environ["OPENAI_API_KEY"] = "sk-..."
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY not set in environment.")

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

analyze_tool = Tool(
    name="Titanic Data Analyzer",
    func=analyze_titanic,
    description="Useful for answering questions about numbers, statistics, and counts."
)

viz_tool = Tool(
    name="Titanic Visualizer",
    func=visualize_titanic,
    description="Useful for generating charts when user asks for 'histogram', 'plot', or 'visual'."
)

agent = initialize_agent(
    [analyze_tool, viz_tool], 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    max_iterations=3
)

# --- 4. FastAPI Server ---
app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: QueryRequest):
    # The agent decides if it needs the Visualizer or Analyzer tool
    result = agent.run(request.question)
    
    # Simple check: If result is long, assume it's base64 image
    # In production, you'd have the agent return JSON metadata
    if len(result) > 100 and not "Error" in result:
        return {"type": "image", "content": result}
    else:
        return {"type": "text", "content": result}
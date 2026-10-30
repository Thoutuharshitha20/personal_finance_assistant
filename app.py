
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from google import genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


app = FastAPI(title="Personal Finance Assistant Agent")


# Request format
class FinanceRequest(BaseModel):
    mode: Literal["expense", "budget", "saving", "explain"]
    topic: str = Field(min_length=1, max_length=5000)


# Connect to Gemma
def generate_with_gemma(prompt_text):

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents=prompt_text
    )

    if not response.text:
        raise ValueError("The AI did not generate a response.")

    return response.text


# Home page
@app.get("/", response_class=HTMLResponse)
def home():

    return playground()


# Health check
@app.get("/health")
def health():

    return {"status": "Personal Finance Assistant is running"}


# AI Assistant interface
@app.get("/agent/playground/", response_class=HTMLResponse)
def playground():

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Personal Finance Assistant</title>

        <meta name="viewport" content="width=device-width, initial-scale=1.0">

        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f0f4f8;
                margin: 0;
                padding: 30px;
            }

            .container {
                max-width: 750px;
                margin: auto;
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }

            h1 {
                text-align: center;
                color: #176b55;
            }

            p {
                text-align: center;
                color: #555;
            }

            label {
                display: block;
                margin-top: 20px;
                font-weight: bold;
            }

            select, textarea {
                width: 100%;
                padding: 12px;
                margin-top: 8px;
                border: 1px solid #ccc;
                border-radius: 6px;
                font-size: 16px;
                box-sizing: border-box;
            }

            textarea {
                height: 140px;
                resize: vertical;
            }

            button {
                width: 100%;
                padding: 14px;
                margin-top: 20px;
                background: #176b55;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 17px;
                cursor: pointer;
            }

            button:hover {
                background: #10513f;
            }

            button:disabled {
                background: #888;
                cursor: not-allowed;
            }

            #result {
                margin-top: 25px;
                padding: 18px;
                background: #f4f8f6;
                border-radius: 8px;
                white-space: pre-wrap;
                line-height: 1.6;
                overflow-wrap: anywhere;
            }

            .disclaimer {
                font-size: 13px;
                margin-top: 20px;
                color: #777;
            }
        </style>
    </head>

    <body>

        <div class="container">

            <h1>Personal Finance Assistant</h1>

            <p>Your AI assistant for managing money and understanding finances.</p>

            <label for="mode">Choose an option:</label>

            <select id="mode">
                <option value="expense">Analyze Expenses</option>
                <option value="budget">Plan My Budget</option>
                <option value="saving">Get Saving Tips</option>
                <option value="explain">Explain Finance Concepts</option>
            </select>

            <label for="topic">Enter your details or question:</label>

            <textarea id="topic"
                placeholder="Example: My monthly income is ₹25,000 and my expenses are ₹18,000. Help me understand my spending.">
            </textarea>

            <button id="submitBtn" onclick="askAgent()">
                Generate Answer
            </button>

            <div id="result"></div>

            <p class="disclaimer">
                Note: This assistant provides general financial information
                and educational suggestions, not professional financial advice.
            </p>

        </div>

        <script>

            async function askAgent() {

                const mode = document.getElementById("mode").value;

                const topic = document.getElementById("topic").value.trim();

                const result = document.getElementById("result");

                const button = document.getElementById("submitBtn");

                if (!topic) {
                    result.innerText = "Please enter your details or question.";
                    return;
                }

                button.disabled = true;
                button.innerText = "Generating...";
                result.innerText = "Please wait. Your AI assistant is preparing an answer...";

                try {

                    const response = await fetch("/finance", {

                        method: "POST",

                        headers: {
                            "Content-Type": "application/json"
                        },

                        body: JSON.stringify({
                            mode: mode,
                            topic: topic
                        })

                    });

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.detail || "Something went wrong.");
                    }

                    result.innerText = data.answer;

                } catch (error) {

                    result.innerText = "Error: " + error.message;

                } finally {

                    button.disabled = false;
                    button.innerText = "Generate Answer";

                }

            }

        </script>

    </body>
    </html>
    """


# Finance AI Agent
@app.post("/finance")
def finance_agent(request: FinanceRequest):

    instructions = {

        "expense": """
        Analyze the user's income and expenses.
        Explain spending patterns, calculate the remaining amount
        when enough information is available, and suggest ways to manage expenses.
        """,

        "budget": """
        Create a simple monthly budget based on the user's income and expenses.
        Suggest reasonable spending categories and possible savings.
        Clearly state any assumptions you make.
        """,

        "saving": """
        Provide practical and realistic money-saving tips.
        Consider the user's income, expenses and financial goals
        if they are provided.
        """,

        "explain": """
        Explain the financial concept in simple language.
        Include its meaning, an easy example and its importance.
        """
    }

    try:

        prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful Personal Finance Assistant.

            Give clear, simple and beginner-friendly answers.

            Do not invent financial details or calculations.
            If important information is missing, explain what is needed.

            Do not guarantee investment returns or recommend specific
            financial products.

            Task:
            {instruction}

            User details or question:
            {topic}
            """
        )

        chain = prompt | RunnableLambda(
            lambda prompt_value: generate_with_gemma(
                prompt_value.to_string()
            )
        )

        answer = chain.invoke({
            "instruction": instructions[request.mode],
            "topic": request.topic
        })

        return {"answer": answer}

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

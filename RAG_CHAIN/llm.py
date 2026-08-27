import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.

    Please provide the most accurate response based on the question.
    Make sure to give a complete and easy-to-understand answer.

    <context>
    {context}
    </context>

    Question: {input}
    """
)


def get_groq_api_key() -> str:
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise EnvironmentError(
            "GROQ API key is missing! Please check your .env file."
        )

    os.environ["GROQ_API_KEY"] = groq_api_key
    return groq_api_key


def build_llm(model_name: str = "openai/gpt-oss-20b") -> ChatGroq:
    groq_api_key = get_groq_api_key()

    return ChatGroq(
        api_key=groq_api_key,
        model_name=model_name,
    )


from examples import get_example_selector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate # type: ignore
import os
from dotenv import load_dotenv # type: ignore

load_dotenv()  # Load environment variables from .env file

# Read the SQL prompt from a text file instead of .env
def load_prompt():
    with open("final_prompt.txt", "r", encoding="utf-8") as file:
        return file.read()

FINAL_PROMPT = load_prompt()
# Get the static part of the prompt
static_prompt = FINAL_PROMPT
example_prompt = ChatPromptTemplate.from_messages(
    [
        # ("human", "{input}\nSQLQuery:"),
         ("human", "{input}"),
        ("ai", "{query}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=get_example_selector(),
    input_variables=["input","top_k"],
)

final_prompt1 = ChatPromptTemplate.from_messages(
    [
        ("system", static_prompt.format(table_details="{table_info}")),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ]
)


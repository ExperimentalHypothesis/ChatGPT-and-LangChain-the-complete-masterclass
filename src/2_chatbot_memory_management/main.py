from langchain_openai import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()
chat = ChatOpenAI()

memory = ConversationBufferMemory(
    memory_key="messages",  # this is the key name that will be created inside the memory buffer
    return_messages=True  # dont just throw strings but wrap them into HumanMessage... for chat like models
)

# a bit CONFUSING here - as this should hold the whole history
prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        # this needs to match with the memory_key inside the ConversationBufferMemory
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory
)

res = None
while True:
    content = input(">> ")
    res = chain({"content": content})
    print(res["text"])

    if content == 'q':
        break
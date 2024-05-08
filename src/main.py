from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain_community.llms.openai import OpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()


llm = OpenAI()

code_prompt = PromptTemplate(
    input_variables=["task", "language"],
    template="Write a very short {language} function that will {task}",
)

test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test fot he following {language} code:<n{code}"
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test",
)

# main chain to bind the output of 1. as input to 2., generating answer
chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["test", "code"]
)

result = chain({
    "language": args.language,
    "task": args.task
})


print(">>>>>>>>> CODE: ")
print(result["code"])

print(">>>>>>>>> TEST: ")
print(result["test"])


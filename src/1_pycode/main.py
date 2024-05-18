# this is an example of using two chains combined together.
# forst chain produces some sample code and send it to second chain which should generate tests for it

from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()


llm = OpenAI()

# 1. chain
code_prompt = PromptTemplate(
    input_variables=["task", "language"],
    template="Write a very short {language} function that will {task}",
)
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

# 2. chain
test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for he following {language} code:<n{code}"
)
test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test",
)

# here is where the magic happens - I need to wire the chains together
# because it is a sequential action
chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["test", "code"]
)

result = chain.invoke({
    "language": args.language,
    "task": args.task
})


print(">>>>>>>>> CODE: ")
print(result["code"])

print(">>>>>>>>> TEST: ")
print(result["test"])


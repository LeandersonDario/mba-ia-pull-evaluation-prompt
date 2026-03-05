import os
from dotenv import load_dotenv
from langsmith import Client
from langchain import hub
from utils import get_llm

load_dotenv()

client = Client()
llm = get_llm()

# Puxar v2 (que agora é v2.1 no hub)
prompt_template = hub.pull("leandersondario/bug_to_user_story_v2")
dataset_name = "prompt-optimization-challenge-resolved-eval"
examples = list(client.list_examples(dataset_name=dataset_name))

example = examples[0]
inputs = example.inputs
reference = example.outputs["reference"]

chain = prompt_template | llm
response = chain.invoke(inputs)

print("--- INPUT ---")
print(inputs)
print("\n--- REFERENCE ---")
print(reference)
print("\n--- GENERATED ANSWER ---")
print(response.content)

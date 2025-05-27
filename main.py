from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from vector import retriever

model = OllamaLLM(model="llama3.2")

# This is what the model will do
template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template) # this gets passed to the model below
# Invokes entire chain to combine multiple components together
chain = prompt | model

while True:
    print("-" * 30)
    question = input("Enter your question about the pizza restaurant (or 'q' to quit): ")
    print("\n\n")

    if question.lower() == 'q':
        break

    reviews = retriever.invoke(question)  # Retrieve relevant reviews based on the question
    result = chain.invoke({
        "reviews": reviews,
        "question": question
    })

    print(result)  # Output the result from the model

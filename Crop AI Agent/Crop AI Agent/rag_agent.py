from langchain_community.llms import Ollama  # <- updated import
from langchain.chains import RetrievalQA
from pdf_loader import db

# Initialize Mistral via Ollama
llm = Ollama(model="mistral", temperature=0.1)

# Connect LLM to your PDF database
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3})
)

# Test it!
response = qa.run("What are meristems?")
print(response)

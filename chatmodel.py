

!pip install langchain

from llama_index  import SimpleDirectoryReader,GPTListIndex, GPTVectorStoreIndex,LLMPredictor,PromptHelper
from langchain import OpenAI
import sys
import os

os.environ["OPENAI_API_KEY"]="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"

from llama_index import ServiceContext,StorageContext,load_index_from_storage
def create_index(path):
  max_input=4096
  tokens=200
  chunk_size=600 #for LLM we need to define chunk size
  max_chunk_overlap=20


  prompt_helper=PromptHelper(max_input,tokens,chunk_size_limit=chunk_size)


  #define LLM
  llmPredictor=LLMPredictor(llm=OpenAI(temperature=0,model_name="text-ada-001",max_tokens=tokens))

  #load_data take all .txtx files if there are more than one
  docs=SimpleDirectoryReader(path).load_data()

  #create vector index
  service_context=ServiceContext.from_defaults(llm_predictor=llmPredictor,prompt_helper=prompt_helper)

  vectorIndex=GPTVectorStoreIndex.from_documents(documents=docs,service_context=service_context)
  vectorIndex.storage_context.persist(persist_dir='Store')
  return vectorIndex

create_index('chatmodel')

def answerMe(vectorIndex):
  storage_context=StorageContext.from_defaults(persist_dir='Store')
  index=load_index_from_storage(storage_context)
  query_engine=index.as_query_engine()
  response=query_engine.query(question)
  return response

response=answerMe("which was the recent release?")

response.response


import chromadb
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def query_papers(input):
    
    OPENAI_API_KEY="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"

    persistent_client = chromadb.PersistentClient()
    vectorstore = Chroma(
        client=persistent_client,
        collection_name="chunks",
        embedding_function=OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small"),
    )

    retriever = vectorstore.as_retriever()

    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise, also
    include the title of the sources you are extracting the response.
    Question: {question} 
    Context: {context} 
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0, openai_api_key=OPENAI_API_KEY)

    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()} 
        | prompt 
        | llm
        | StrOutputParser() 
    )

    return rag_chain.invoke(input)



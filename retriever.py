import chromadb
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain.schema.output_parser import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt 
        | llm
        | StrOutputParser() 
    )
    
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain)

    return rag_chain_with_source.invoke(input)

# result = query_papers("Los smartphones perjudican la calidad de sueño")
# print(result)
# print('------')
# print(result['answer'])
# print('----')
# for i in result['context']:
#     print(i.metadata['title'])
#     print(i.metadata['author'])


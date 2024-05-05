import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

OPENAI_API_KEY="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"

papers = [
    "1409-4258-ree-26-02-408.pdf",
    "2452-6053-andesped-andespediatr-v94i1-4203.pdf",
    "2308-0531-rfmh-23-04-62.pdf",
    "2223-7666-liber-27-01-e458.pdf",
    "2529-850X-jonnpr-05-12-1558.pdf",
    "1729-519X-rhcm-18-06-942.pdf",
    "2255-3517-enefro-22-04-361.pdf",
    "v79s3a08.pdf",
    "0213-1285-odonto-35-2-83.pdf",
    "1561-2961-enf-35-02-e1718.pdf",
    "2007-5057-iem-8-30-9.pdf",
    "1561-3119-ped-91-02-e518.pdf",
    "v117n2a04.pdf",
    "v79n1s1a10.pdf",
    "a05v82n1.pdf",
    "serIVn19a03.pdf",
    "1390-7697-rctu-5-02-00037.pdf",
    "2395-8421-eu-15-01-6.pdf"
]

for paper in papers:
    loader = PyMuPDFLoader('papers/' + paper)
    documents = loader.load()

    text_splitter = SemanticChunker(OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    chunks = text_splitter.split_documents(documents)
    print(len(chunks))

    persistent_client = chromadb.PersistentClient()
    vectorstore = Chroma.from_documents(chunks,
        OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small"),
        client=persistent_client,
        collection_name="chunks"
    )

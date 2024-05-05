import chromadb

persistent_client = chromadb.PersistentClient()
persistent_client.get_or_create_collection("papers")
persistent_client.get_or_create_collection("chunks")
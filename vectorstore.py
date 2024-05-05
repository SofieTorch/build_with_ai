import chromadb

persistent_client = chromadb.PersistentClient()
persistent_client.get_or_create_collection("papers")
collection = persistent_client.get_or_create_collection("chunks")
print(collection)
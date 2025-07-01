import chromadb

client = chromadb.PersistentClient(path = "./database")
client.delete_collection("embeddings")
client.create_collection("embeddings")
print("Worked!")


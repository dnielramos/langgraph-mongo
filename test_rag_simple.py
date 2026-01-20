# test_rag_simple.py
# Simplified diagnostic script

import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

print("=" * 60)
print("RAG DIAGNOSTIC")
print("=" * 60)

# Connect to MongoDB
mongodb_uri = os.getenv('MONGODB_URI')
db_name = os.getenv('MONGODB_DB_NAME', 'predisaber')

client = MongoClient(mongodb_uri)
db = client[db_name]
collection = db['vector_store']

# Count documents
doc_count = collection.count_documents({})
print(f"\n1. Documents in vector_store: {doc_count}")

# Show documents
print("\n2. Documents content:")
docs = list(collection.find({}, {"content": 1, "_id": 0}).limit(10))
for i, doc in enumerate(docs):
    content = doc.get('content', 'N/A')
    print(f"   [{i+1}] {content[:120]}...")

# Check if vector index exists
print("\n3. Collection indexes:")
indexes = list(collection.list_indexes())
for idx in indexes:
    print(f"   - {idx.get('name')}")

# Check embedding field
print("\n4. Checking embedding field:")
sample = collection.find_one({"embedding": {"$exists": True}})
if sample:
    emb = sample.get('embedding', [])
    print(f"   Embedding exists: True, Dimension: {len(emb)}")
else:
    print("   Embedding exists: False")

# Try simple text search as fallback
print("\n5. Text search for 'XYZ':")
text_results = list(collection.find({"content": {"$regex": "XYZ", "$options": "i"}}))
print(f"   Found {len(text_results)} documents containing 'XYZ'")
for doc in text_results:
    print(f"   -> {doc.get('content', '')[:100]}...")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

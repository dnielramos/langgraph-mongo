# test_vector_search.py
# Script de diagnóstico para probar la búsqueda vectorial

import os
import asyncio
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_voyageai import VoyageAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

load_dotenv()

def test_vector_search():
    print("=" * 60)
    print("🔍 DIAGNÓSTICO DE BÚSQUEDA VECTORIAL")
    print("=" * 60)
    
    # 1. Verificar conexión a MongoDB
    print("\n1️⃣ Conectando a MongoDB...")
    mongodb_uri = os.getenv('MONGODB_URI')
    db_name = os.getenv('MONGODB_DB_NAME', 'predisaber')
    voyage_api_key = os.getenv('VOYAGE_API_KEY')
    
    client = MongoClient(mongodb_uri)
    db = client[db_name]
    collection = db['vector_store']
    
    # 2. Contar documentos en la colección
    doc_count = collection.count_documents({})
    print(f"   ✅ Conectado. Documentos en vector_store: {doc_count}")
    
    # 3. Mostrar algunos documentos
    print("\n2️⃣ Documentos en la base de datos:")
    docs = list(collection.find({}, {"content": 1, "embedding": {"$slice": 3}}).limit(5))
    for i, doc in enumerate(docs):
        content = doc.get('content', 'N/A')[:100]
        has_embedding = 'embedding' in doc and len(doc.get('embedding', [])) > 0
        print(f"   [{i+1}] {content}...")
        print(f"       Has embedding: {has_embedding}, Length: {len(doc.get('embedding', []))}")
    
    # 4. Inicializar embeddings
    print("\n3️⃣ Inicializando VoyageAI embeddings...")
    embeddings = VoyageAIEmbeddings(
        voyage_api_key=voyage_api_key,
        model="voyage-4-large"
    )
    print("   ✅ VoyageAI inicializado")
    
    # 5. Crear vector store
    print("\n4️⃣ Creando vector store...")
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index",
        text_key="content",
        embedding_key="embedding"
    )
    print("   ✅ Vector store creado")
    
    # 6. Probar búsquedas
    test_queries = [
        "empresa XYZ",
        "¿En qué se especializa la empresa XYZ?",
        "XYZ software desarrollo",
        "inteligencia artificial machine learning",
        "Daniel Ramos",
        "¿Cuándo fue fundada la empresa?"
    ]
    
    print("\n5️⃣ Probando búsquedas semánticas:")
    for query in test_queries:
        print(f"\n   🔎 Query: '{query}'")
        try:
            results = vector_store.similarity_search(query, k=3)
            if results:
                for j, res in enumerate(results):
                    print(f"      [{j+1}] Score: N/A | Content: {res.page_content[:80]}...")
            else:
                print("      ❌ Sin resultados")
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    # 7. Probar búsqueda con scores
    print("\n6️⃣ Probando búsqueda con scores:")
    try:
        query = "empresa XYZ software"
        results_with_score = vector_store.similarity_search_with_score(query, k=5)
        print(f"   Query: '{query}'")
        for doc, score in results_with_score:
            print(f"      Score: {score:.4f} | {doc.page_content[:80]}...")
    except Exception as e:
        print(f"   ❌ Error en similarity_search_with_score: {e}")
    
    # 8. Verificar índice
    print("\n7️⃣ Verificando índices en colección:")
    try:
        indexes = list(collection.list_indexes())
        for idx in indexes:
            print(f"   - {idx.get('name')}: {idx.get('key')}")
    except Exception as e:
        print(f"   ❌ Error listando índices: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Diagnóstico completado")
    print("=" * 60)

if __name__ == "__main__":
    test_vector_search()

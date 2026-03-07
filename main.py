# super_agent_mongodb.py
import os
import asyncio
from typing import Dict, List, Any, TypedDict, Annotated, Literal, Optional
from datetime import datetime
import logging
import certifi
from dotenv import load_dotenv

# Cargar variables de entorno al inicio
load_dotenv()

# LangChain y LangGraph imports
from langchain_cerebras import ChatCerebras
from langchain.embeddings.base import Embeddings
import voyageai
import httpx
import certifi
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# MongoDB imports
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure

# FastAPI imports
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SuperAgentMongoDB")

class AgentState(TypedDict):
    """Estado completo del agente con memoria persistente"""
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    chat_history: list[dict]
    current_step: str
    tool_calls: list[dict]
    tool_results: list[dict]
    needs_tool: bool
    final_response: str
    context_retrieved: list[Document]
    search_query: str
    error: Optional[str]

class SuperAgentMongoDB:
    def __init__(self):
        """Inicializa el super agente con MongoDB Atlas como cerebro central"""
        self._load_env_vars()
        self._init_mongodb()
        self._init_vector_store()
        self._init_llm()
        self._init_tools()
        self._build_graph()
        self._init_web_app()
        
    def _load_env_vars(self):
        """Carga variables de entorno críticas"""
        self.cerebras_api_key = os.getenv('CEREBRAS_API_KEY')
        self.voyage_api_key = os.getenv('VOYAGE_API_KEY')
        self.voyage_base_url = os.getenv('VOYAGE_BASE_URL', 'https://api.voyageai.com/v1')
        self.mongodb_uri = os.getenv('MONGODB_URI')
        self.db_name = os.getenv('MONGODB_DB_NAME', 'super_agent_db')
        self.mock_mode = False
        
        if not self.cerebras_api_key or not self.voyage_api_key or not self.mongodb_uri:
            logger.warning("⚠️ Faltan variables de entorno. Iniciando en modo MOCK para diagnóstico.")
            self.mock_mode = True
            # Valores dummy para permitir arranque
            self.cerebras_api_key = self.cerebras_api_key or "dummy_key"
            self.voyage_api_key = self.voyage_api_key or "dummy_key"
            self.mongodb_uri = self.mongodb_uri or "mongodb://dummy:dummy@localhost:27017/dummy"
    
    def _init_mongodb(self):
        """Inicializa todas las colecciones de MongoDB necesarias"""
        try:
            # Configuración robusta de SSL con certifi para entornos Linux/Render
            self.client = MongoClient(
                self.mongodb_uri,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )

            if not self.mock_mode:
                # Verificar conexión real
                self.client.admin.command('ping')

            self.db = self.client[self.db_name]
            
            # Colecciones especializadas
            self.chat_history_col = self.db["chat_history"]
            self.documents_col = self.db["knowledge_base"]
            self.embeddings_col = self.db["vector_store"]
            self.sessions_col = self.db["sessions"]
            self.tools_col = self.db["tools_usage"]
            self.errors_col = self.db["errors"]
            
            # Índices para rendimiento
            self.chat_history_col.create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
            self.documents_col.create_index([("content", "text")])
            self.sessions_col.create_index([("last_active", DESCENDING)], expireAfterSeconds=86400)  # 24h TTL
            
            # Verificar conexión
            self.client.admin.command('ping')
            logger.info("✅ Conexión exitosa a MongoDB Atlas - ¡Tu cerebro central está activo!")
        except ConnectionFailure as e:
            logger.error(f"❌ Error conectando a MongoDB Atlas: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error inesperado en MongoDB: {e}")
            raise
    
    def _init_vector_store(self):
        """Inicializa el vector store para RAG con MongoDB Atlas y VoyageAI"""
        try:
            # Clase wrapper para ignorar las validaciones estrictas de pydantic de langchain_voyageai
            # y poder inyectar la base_url de MongoDB Atlas AI (las keys al-)
            voyage_api_key = self.voyage_api_key
            voyage_base_url = self.voyage_base_url

            class MongoVoyageEmbeddings(Embeddings):
                """Wrapper de VoyageAI usando peticiones HTTP directas al endpoint MongoDB Atlas AI."""
                EMBED_MODEL = "voyage-3-large"

                def __init__(self):
                    self.api_key = voyage_api_key
                    # Asegurar que la URL termine correctamente
                    base = voyage_base_url.rstrip("/")
                    if not base.endswith("/v1"):
                        base += "/v1"
                    self.base_url = base
                    self.headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }

                def embed_documents(self, texts: list[str]) -> list[list[float]]:
                    return self._get_embeddings(texts, "document")

                def embed_query(self, text: str) -> list[float]:
                    return self._get_embeddings([text], "query")[0]
                
                def _get_embeddings(self, texts: list[str], input_type: str) -> list[list[float]]:
                    payload = {
                        "input": texts,
                        "model": self.EMBED_MODEL,
                        "input_type": input_type
                    }
                    try:
                        response = httpx.post(
                            f"{self.base_url}/embeddings",
                            headers=self.headers,
                            json=payload,
                            timeout=30.0
                        )
                        response.raise_for_status()
                        data = response.json()
                        # VoyageAI devuelve data[i].embedding
                        return [item["embedding"] for item in data.get("data", [])]
                    except Exception as e:
                        logger.error(f"Error HTTP obteniendo embeddings de MongoDB Atlas AI: {e}")
                        if hasattr(e, "response") and e.response:
                            logger.error(f"Detalle API: {e.response.text}")
                        raise

            self.embeddings = MongoVoyageEmbeddings()

            self.vector_store = MongoDBAtlasVectorSearch(
                collection=self.embeddings_col,
                embedding=self.embeddings,
                index_name="vector_index",
                text_key="content",
                embedding_key="embedding"
            )

            logger.info("🧠 Vector Store con VoyageAI (MongoDB Atlas AI) inicializado - Listo para recuperación semántica!")
        except Exception as e:
            logger.error(f"❌ Error inicializando vector store: {e}")
            raise
    
    def _init_llm(self):
        """Inicializa el modelo de lenguaje Cerebras con capacidades avanzadas"""
        try:
            self.llm = ChatCerebras(
                model=os.getenv("CEREBRAS_MODEL"),
                api_key=self.cerebras_api_key,
                temperature=0.3,
                max_tokens=2000
            )
            
            logger.info("🤖 Cerebras " + os.getenv("CEREBRAS_MODEL") + " inicializado - ¡Potencia máxima activada!")    
        except Exception as e:
            logger.error(f"❌ Error inicializando LLM: {e}")
            raise
    
    def _init_tools(self):
        """Crea herramientas inteligentes que interactúan con MongoDB"""
        @tool
        async def search_knowledge_base(query: str) -> str:
            """Busca información en la base de conocimiento usando búsqueda semántica (sirve para personas, productos, contactos, etc.)"""
            try:
                # Se busca en todos los documentos ingestados sin pre_filter
                results = await self.vector_store.asimilarity_search(
                    query, 
                    k=5
                )
                
                # Guardar uso de herramienta en MongoDB
                self.tools_col.insert_one({
                    "tool_name": "search_knowledge_base",
                    "query": query,
                    "results_count": len(results),
                    "timestamp": datetime.utcnow(),
                    "type": "general_search"
                })
                
                if not results:
                    return "No encontré información relacionada con tu búsqueda en la base de conocimiento."
                
                formatted_results = []
                for doc in results:
                    metadata = doc.metadata
                    # Formato flexible dependiendo si es producto o información general
                    if metadata.get('type') == 'product':
                        formatted_results.append(
                            f"🔹 **{metadata.get('name', 'Producto')}**\n"
                            f"💰 Precio: ${metadata.get('price', 'N/A')}\n"
                            f"⭐ Rating: {metadata.get('rating', 'N/A')}/5\n"
                            f"📦 Stock: {metadata.get('stock', 'N/A')}\n"
                            f"📝 Descripción: {doc.page_content}"
                        )
                    else:
                        formatted_results.append(f"📄 **Información general**: {doc.page_content}")
                
                return "\n\n".join(formatted_results)
            
            except Exception as e:
                logger.error(f"Error en search_knowledge_base: {e}")
                return f"Error al buscar en la base de conocimiento: {str(e)}"
        
        @tool
        async def web_research(query: str) -> str:
            """Realiza investigación web en tiempo real para información actualizada"""
            try:
                search = DuckDuckGoSearchRun()
                results = search.run(query)
                
                # Guardar en base de conocimiento
                if results:
                    doc = Document(
                        page_content=results,
                        metadata={
                            "source": "web_research",
                            "query": query,
                            "timestamp": datetime.utcnow().isoformat(),
                            "type": "research"
                        }
                    )
                    await self.vector_store.aadd_documents([doc])
                
                return results[:500] + "..." if len(results) > 500 else results
            
            except Exception as e:
                logger.error(f"Error en web_research: {e}")
                return f"No pude realizar la investigación web: {str(e)}"
        
        @tool
        async def analyze_conversation_context(user_id: str) -> str:
            """Analiza el historial de conversación para contexto personalizado"""
            try:
                history = list(self.chat_history_col.find(
                    {"user_id": user_id},
                    {"_id": 0, "message": 1, "role": 1, "timestamp": 1}
                ).sort("timestamp", -1).limit(10))
                
                if not history:
                    return "No hay historial de conversación previo para este usuario."
                
                context_summary = []
                for msg in history:
                    role = "Usuario" if msg["role"] == "human" else "Asistente"
                    timestamp_str = msg['timestamp'].strftime('%Y-%m-%d %H:%M') if hasattr(msg['timestamp'], 'strftime') else str(msg['timestamp'])
                    context_summary.append(f"{role} ({timestamp_str}): {msg['message']}")
                
                return "\n".join(reversed(context_summary))
            
            except Exception as e:
                logger.error(f"Error en analyze_conversation_context: {e}")
                return "Error al analizar el contexto de conversación."
        
        self.tools = [search_knowledge_base, web_research, analyze_conversation_context]
        logger.info("🛠️ Herramientas inteligentes inicializadas - ¡Listas para acción!")
    
    def _build_graph(self):
        """Construye el grafo de estados con LangGraph para flujo de trabajo por pasos"""
        
        # Prompt del sistema con instrucciones para usar SIEMPRE el contexto
        system_prompt = """Eres Daniel, un asistente de IA empresarial con acceso a una base de conocimiento.

**REGLA CRÍTICA: Si se proporciona contexto de la base de conocimiento, DEBES usarlo para responder.**

Comportamiento:
1. Si el contexto contiene información relevante a la pregunta, USA ESA INFORMACIÓN.
2. Cita el contenido del contexto de manera precisa. Busca en el contexto la información que necesitas y úsala para responder.
3. Si el contexto NO tiene información relevante, indica que "no encontré información sobre esto en mi base de conocimiento".
4. NUNCA inventes información que no esté en el contexto.
5. Si el usuario pregunta sobre algo específico (persona, empresa, fecha), busca exactamente eso en el contexto.

{context}

Responde de manera útil, precisa y basándote en el contexto cuando esté disponible."""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Definir nodos del grafo
        def analyze_input(state: AgentState) -> AgentState:
            """Paso 1: Analizar la entrada del usuario y buscar contexto relevante"""
            latest_message = state["messages"][-1].content
            logger.info(f"🔍 Analizando: {latest_message}")
            
            context_results = []
            
            # 1. Intentar búsqueda vectorial
            try:
                vector_results = self.vector_store.similarity_search(latest_message, k=5)
                if vector_results:
                    logger.info(f"✅ Búsqueda vectorial: {len(vector_results)} resultados")
                    for r in vector_results:
                        logger.info(f"   -> {r.page_content[:80]}...")
                    context_results.extend(vector_results)
            except Exception as e:
                logger.warning(f"⚠️ Error búsqueda vectorial: {e}")
            
            # 2. Búsqueda de texto como respaldo/complemento
            try:
                # Extraer términos de búsqueda clave
                search_terms = [term.strip() for term in latest_message.split() if len(term) > 2]
                for term in search_terms[:5]:  # Limitar a 5 términos
                    # Buscar en la colección de documentos directamente para evitar problemas de índices incompletos
                    text_results = list(self.documents_col.find(
                        {"content": {"$regex": term, "$options": "i"}},
                        {"content": 1, "metadata": 1, "_id": 0} # Incluir metadata para Document
                    ).limit(3))
                    if text_results:
                        logger.info(f"📝 Búsqueda texto '{term}': {len(text_results)} resultados")
                        for doc in text_results:
                            from langchain_core.documents import Document
                            doc_obj = Document(page_content=doc.get('content', ''), metadata=doc.get('metadata', {}))
                            # Evitar duplicados
                            if not any(doc_obj.page_content == r.page_content for r in context_results):
                                context_results.append(doc_obj)
            except Exception as e:
                logger.warning(f"⚠️ Error búsqueda texto: {e}")
            
            # Limitar a máximo 10 resultados
            context_results = context_results[:10]
            logger.info(f"📚 Total contexto recuperado: {len(context_results)} documentos")
            
            state["context_retrieved"] = context_results
            state["needs_tool"] = any(
                keyword in latest_message.lower() 
                for keyword in ["buscar", "investigar", "web", "internet"]
            )
            state["current_step"] = "analyze_input"
            return state
        
        def use_tools(state: AgentState) -> AgentState:
            """Paso 2: Usar herramientas si es necesario"""
            if not state["needs_tool"]:
                state["current_step"] = "skipped_tools"
                return state
            
            logger.info("🔧 Usando herramientas inteligentes...")
            latest_message = state["messages"][-1].content
            
            # Como fallback por si el modelo no generó tool_calls nativos, 
            # forzamos la búsqueda en la base de conocimiento manualmente
            try:
                # Ejecutar búsqueda síncrona en la base de conocimiento
                tool_result = search_knowledge_base.invoke({"query": latest_message})
                state["tool_results"].append({
                    "tool": "search_knowledge_base",
                    "result": tool_result,
                    "timestamp": datetime.utcnow().isoformat()
                })
                logger.info("✅ Búsqueda en conocimiento completada")
            except Exception as e:
                logger.error(f"Error ejecutando search_knowledge_base: {e}")
                
            state["current_step"] = "tools_used"
            return state
        
        def generate_response(state: AgentState) -> AgentState:
            """Paso 3: Generar respuesta final con contexto completo"""
            logger.info("✍️ Generando respuesta final...")
            
            # Construir contexto completo para el LLM
            context_parts = []
            
            # Añadir contexto recuperado del vector store o text search
            if state["context_retrieved"]:
                context_parts.append("📚 **Contexto relevante de tu base de conocimiento:**")
                for i, doc in enumerate(state["context_retrieved"]):
                    # No truncar el contenido, necesitamos toda la información
                    context_parts.append(f"--- Documento {i+1} ---\n{doc.page_content}\n")
            
            # Añadir resultados de herramientas
            if state["tool_results"]:
                context_parts.append("\n🛠️ **Resultados de herramientas:**")
                for result in state["tool_results"]:
                    context_parts.append(f"- {result['result']}")
            
            # Crear contexto completo
            full_context = "\n".join(context_parts) if context_parts else "Sin contexto adicional disponible."
            
            logger.info(f"=== CONTEXTO PARA LLM ===\n{full_context}\n=========================")
            
            # Generar respuesta
            response_chain = self.prompt | self.llm | StrOutputParser()
            response = response_chain.invoke({
                "messages": state["messages"],
                "context": full_context
            })
            
            state["final_response"] = response
            state["current_step"] = "response_generated"
            
            # Guardar en historial de MongoDB
            self._save_to_history(state)
            
            logger.info("✅ Respuesta generada y guardada en historial")
            return state
        
        # Construir el grafo
        self.workflow = StateGraph(AgentState)
        
        # Añadir nodos
        self.workflow.add_node("analyze_input", analyze_input)
        self.workflow.add_node("use_tools", use_tools)
        self.workflow.add_node("generate_response", generate_response)
        
        # Definir rutas
        self.workflow.set_entry_point("analyze_input")
        self.workflow.add_edge("analyze_input", "use_tools")
        self.workflow.add_edge("use_tools", "generate_response")
        self.workflow.add_edge("generate_response", END)
        
        # Compilar el grafo con checkpoint para memoria persistente
        memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=memory)
        
        logger.info("📊 Grafo de estados construido - ¡Flujo de trabajo inteligente activado!")
    
    def _save_to_history(self, state: AgentState):
        """Guarda la conversación en MongoDB con contexto completo"""
        try:
            latest_message = state["messages"][-1]
            response_message = AIMessage(content=state["final_response"])
            
            # Guardar mensaje del usuario
            self.chat_history_col.insert_one({
                "user_id": state["user_id"],
                "session_id": state["session_id"],
                "role": "human",
                "message": latest_message.content,
                "timestamp": datetime.utcnow(),
                "context_used": [doc.metadata for doc in state["context_retrieved"]],
                "tools_used": state["tool_results"]
            })
            
            # Guardar respuesta del asistente
            self.chat_history_col.insert_one({
                "user_id": state["user_id"],
                "session_id": state["session_id"],
                "role": "ai",
                "message": state["final_response"],
                "timestamp": datetime.utcnow(),
                "step_completed": state["current_step"],
                "thinking_process": state["tool_results"]
            })
            
            # Actualizar sesión
            self.sessions_col.update_one(
                {"session_id": state["session_id"]},
                {
                    "$set": {
                        "last_active": datetime.utcnow(),
                        "user_id": state["user_id"],
                        "last_message": state["final_response"]
                    }
                },
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Error guardando historial: {e}")
            # Intentar guardar en colección de errores como fallback
            self.errors_col.insert_one({
                "error_type": "history_save_error",
                "error_message": str(e),
                "timestamp": datetime.utcnow(),
                "state_snapshot": str(state)
            })
    
    async def ingest_document(self, content: str, metadata: Dict[str, Any] = None):
        """Ingresa un documento en la base de conocimiento vectorial"""
        try:
            if metadata is None:
                metadata = {"source": "manual_ingestion", "timestamp": datetime.utcnow().isoformat()}
            
            doc = Document(page_content=content, metadata=metadata)
            
            # Dividir documento si es largo
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents([doc])
            
            # Añadir a vector store
            await self.vector_store.aadd_documents(chunks)
            
            # Guardar en colección de documentos
            self.documents_col.insert_many([
                {
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                    "timestamp": datetime.utcnow(),
                    "type": metadata.get("type", "general"),
                    "ingested_by": metadata.get("ingested_by", "system")
                }
                for chunk in chunks
            ])
            
            logger.info(f"✅ Documento ingresado exitosamente. {len(chunks)} chunks creados.")
            return {"status": "success", "chunks_created": len(chunks)}
        
        except Exception as e:
            logger.error(f"❌ Error ingresando documento: {e}")
            raise
    
    async def process_user_message(self, user_id: str, session_id: str, message: str) -> str:
        """Procesa un mensaje de usuario a través del grafo de estados"""
        try:
            # Obtener historial de esta sesión
            session_history = list(self.chat_history_col.find(
                {"session_id": session_id},
                {"_id": 0, "role": 1, "message": 1}
            ).sort("timestamp", 1))
            
            # Construir historial de mensajes
            messages = []
            for msg in session_history:
                if msg["role"] == "human":
                    messages.append(HumanMessage(content=msg["message"]))
                else:
                    messages.append(AIMessage(content=msg["message"]))
            
            # Añadir nuevo mensaje
            messages.append(HumanMessage(content=message))
            
            # Estado inicial
            initial_state: AgentState = {
                "messages": messages,
                "user_id": user_id,
                "session_id": session_id,
                "chat_history": session_history,
                "current_step": "initial",
                "tool_calls": [],
                "tool_results": [],
                "needs_tool": False,
                "final_response": "",
                "context_retrieved": [],
                "search_query": message,
                "error": None
            }
            
            # Ejecutar el grafo
            config = {"configurable": {"thread_id": f"{user_id}_{session_id}"}}
            final_state = await self.app.ainvoke(initial_state, config=config)
            
            return final_state["final_response"]
        
        except Exception as e:
            logger.error(f"❌ Error procesando mensaje: {e}")
            error_msg = "Lo siento, ocurrió un error al procesar tu solicitud. Por favor, inténtalo de nuevo."
            
            # Guardar error en MongoDB
            self.errors_col.insert_one({
                "user_id": user_id,
                "session_id": session_id,
                "error_type": "processing_error",
                "error_message": str(e),
                "timestamp": datetime.utcnow(),
                "original_message": message
            })
            
            return error_msg
    
    def _init_web_app(self):
        """Inicializa la aplicación FastAPI para streaming en tiempo real"""
        self.app_web = FastAPI(title="Super Agent NOVA", version="1.0.0")
        
        # Configurar CORS
        self.app_web.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @self.app_web.websocket("/ws/{user_id}/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, user_id: str, session_id: str):
            try:
                await websocket.accept()
                logger.info(f"🔌 Nueva conexión WebSocket ACEPTADA: user_id={user_id}, session_id={session_id}")
            except Exception as e:
                logger.error(f"❌ Error aceptando WebSocket: {e}")
                return

            try:
                if self.mock_mode:
                     await websocket.send_json({"type": "final_response", "data": {"response": "⚠️ MODO DIAGNÓSTICO: Backend en modo mock (faltan credenciales). La conexión WebSocket funciona.", "timestamp": datetime.utcnow().isoformat()}})

                while True:
                    data = await websocket.receive_text()
                    
                    if data.strip().lower() == "exit":
                        break
                    
                    # Enviar evento de inicio
                    await websocket.send_json({"type": "thinking_start", "data": {"message": "Analizando tu solicitud..."}})
                    
                    # Procesar mensaje
                    response = await self.process_user_message(user_id, session_id, data)
                    
                    # Enviar respuesta completa
                    await websocket.send_json({
                        "type": "final_response",
                        "data": {
                            "response": response,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    })
                    
                    # Enviar evento de finalización
                    await websocket.send_json({"type": "thinking_end", "data": {"status": "completed"}})
            
            except Exception as e:
                logger.error(f"❌ Error en WebSocket: {e}")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": f"Error en la conexión: {str(e)}"}
                    })
                except:
                    pass
            
            finally:
                try:
                    await websocket.close()
                except:
                    pass
                logger.info(f"🔌 Conexión WebSocket cerrada: user_id={user_id}, session_id={session_id}")
        
        # Endpoint REST para ingestar documentos
        @self.app_web.post("/ingest")
        async def ingest_endpoint(request: Request):
            """Ingresa un documento en la base de conocimiento"""
            try:
                data = await request.json()
                content = data.get("content")
                metadata = data.get("metadata", {})
                
                if not content:
                    return {"status": "error", "message": "El campo 'content' es requerido"}
                
                result = await self.ingest_document(content, metadata)
                return result
            except Exception as e:
                logger.error(f"❌ Error en endpoint ingest: {e}")
                return {"status": "error", "message": str(e)}
        
        # Endpoint de health check
        @self.app_web.get("/health")
        async def health_check():
            return {"status": "healthy", "agent": "NOVA", "version": "1.0.0"}
        
        # Servir frontend estático
        @self.app_web.get("/")
        async def serve_frontend():
            frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
            if os.path.exists(frontend_path):
                return FileResponse(frontend_path)
            return {"message": "Frontend no encontrado. Usa ws://localhost:8000/ws/{user_id}/{session_id} para conectarte."}
        
        # Servir archivos estáticos del frontend
        frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
        if os.path.exists(frontend_dir):
            self.app_web.mount("/static", StaticFiles(directory=frontend_dir), name="static")
        
        logger.info("🌐 FastAPI inicializado - Listo para conexiones WebSocket y REST!")
    
    async def start_local(self):
        """Inicia el servidor localmente (método legacy para desarrollo)"""
        import uvicorn
        logger.info("="*60)
        logger.info("🚀 ¡SUPER AGENTE ACTIVADO LOCALMENTE!")
        logger.info("="*60)
        # Ingesta de ejemplo en background para no bloquear
        asyncio.create_task(self._ingest_example_knowledge())
        
        # Iniciar servidor
        config = uvicorn.Config(self.app_web, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    
    async def _ingest_example_knowledge(self):
        """Ingresa conocimiento de ejemplo para demostración (opcional, no bloquea el servidor)"""
        # Verificar si ya hay documentos
        if self.documents_col.count_documents({}) > 0:
            logger.info("📚 La base de conocimiento ya tiene datos, saltando ingesta de ejemplo.")
            return
            
        logger.info("📚 Intentando ingresar conocimiento de ejemplo...")
        # Ejemplo de productos
        products = [
            {
                "content": "Dell Pro 15 - Procesador Intel i9, 32GB RAM, 1TB SSD, RTX 4080, pantalla 144Hz",
                "metadata": {"name": "Dell Pro 15", "price": 1899.99, "stock": 15, "rating": 4.8, "type": "product"}
            },
            {
                "content": "Monitor UltraWide 34 pulgadas - Resolución 3440x1440, 144Hz, HDR10, tiempo de respuesta 1ms",
                "metadata": {"name": "Monitor UltraWide 34\"", "price": 699.99, "stock": 23, "rating": 4.7, "type": "product"}
            },
            {
                "content": "Teclado Mecánico RGB - Switches Cherry MX Red, retroiluminación personalizable, reposamuñecas ergonómico",
                "metadata": {"name": "Teclado Mecánico RGB Pro", "price": 129.99, "stock": 42, "rating": 4.6, "type": "product"}
            }
        ]
        
        try:
            for product in products:
                await self.ingest_document(product["content"], product["metadata"])
            logger.info("✅ Conocimiento de ejemplo ingresado exitosamente!")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo ingresar conocimiento de ejemplo: {e}")

# -----------------------------------------------------------------------------
# Configuración para Producción (Render / Uvicorn)
# -----------------------------------------------------------------------------

# Instancia global del agente
try:
    nova_agent = SuperAgentMongoDB()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Gestiona el ciclo de vida de la aplicación (reemplaza on_event)"""
        # Startup
        logger.info("🚀 NOVA iniciando en entorno de producción")
        try:
            asyncio.create_task(nova_agent._ingest_example_knowledge())
        except Exception as e:
            logger.warning(f"⚠️ No se pudo programar ingesta de ejemplo: {e}")
        yield
        # Shutdown (aquí se puede cerrar la conexión MongoDB si es necesario)
        logger.info("🛑 NOVA cerrando...")

    # Reasignar lifespan a la app web existente
    nova_agent.app_web.router.lifespan_context = lifespan
    app = nova_agent.app_web  # Objeto ASGI expuesto para servidores de producción

except Exception as e:
    logger.critical(f"🔥 Error fatal iniciando NOVA: {e}")
    raise

if __name__ == "__main__":
    import uvicorn
    
    # Obtener puerto de Render o usar 8000 por defecto
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"  # Para Render y producción
    
    logger.info(f"📍 Iniciando servidor en {host}:{port}...")
    
    # Configuración optimizada para producción
    uvicorn.run(
        "main:app",  # El módulo es 'main' porque el archivo se llama main.py
        host=host,
        port=port,
        workers=1,  # Render maneja el escalado
        log_level="info",
        reload=False,  # Desactivar reload en producción
        timeout_keep_alive=60,
    )
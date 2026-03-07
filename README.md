# 🧠 NOVA - Super Agente RAG con Cerebras y MongoDB Atlas

¡Bienvenido a NOVA! Este es un agente de Inteligencia Artificial empresarial potenciado por **LangGraph**, el motor inferencial ultrarrápido de **Cerebras**, y búsqueda semántica avanzada usando **MongoDB Atlas Vector Search** con el modelo de incrustaciones de **VoyageAI**. 

## ✨ Características Especiales 
- **Respuestas Ultra-Rápidas:** Genera respuestas a una velocidad nunca antes vista usando los chips LPU de Cerebras.
- **Memoria Avanzada RAG:** Recupera no solo productos, también información general como contactos, reglas de negocio o descripciones usando búsqueda semántica y textual sobre tus documentos.
- **Búsqueda Web en Tiempo Real:** Cuando el modelo lo decide necesario, puede navegar usando DuckDuckGo internamente para ofrecer información al día.
- **Websocket Bidireccional:** Todo preparado para ser conectado con cualquier Frontend (React, Angular, Vanilla JS) en tiempo real.

---

## 🚀 Despliegue en Render (Paso a Paso)

El código ya está completamente refactorizado y optimizado (puertos dinámicos, disable reload, httpx routing) para ser subido a producción gratuitamente usando **Render**.

### 1. Conectar tu repositorio
Sube todo este código fuente actual a un repositorio nuevo en tu cuenta de GitHub de forma normal.

### 2. Crear el Web Service
- Ve a [Render.com](https://render.com/) e inicia sesión.
- Toca el botón **New +** y selecciona **Web Service**.
- Conecta tu cuenta de Github y selecciona el repositorio que acabas de subir.

### 3. Configurar el Despliegue
Llena los campos como se muestra a continuación:
- **Name:** *nova-agent-backend* (o el nombre que prefieras).
- **Region:** *Ohio (US East)* o la más cercana a ti.
- **Branch:** *main*
- **Runtime:** `Python 3`
- **Build Command:** 
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command:** 
  ```bash
  python main.py
  ```
  *(El script `main.py` arrancará Uvicorn internamente usando el puerto dinámico de Render).*

### 4. Configurar Variables de Entorno (Environment Variables)
Agrega estrictamente las siguientes llaves bajando hacia la sección "Environment Variables". Usa **los mismos valores que tienes actualmente en tu archivo local `.env`**:

| Key | Ej. Value | Descripción |
| :--- | :--- | :--- |
| `CEREBRAS_API_KEY` | `csk-xxxxxxxxx` | Tu llave del portal de Cerebras |
| `CEREBRAS_MODEL` | `llama3.3-70b` | El modelo que desees usar |
| `VOYAGE_API_KEY` | `al-xxxxxxxx` | Tu llave de MongoDB Atlas AI (prefijo al-) |
| `VOYAGE_BASE_URL` | `https://ai.mongodb.com/v1` | URL enrutadora directa de Atlas AI |
| `MONGODB_URI` | `mongodb+srv://...` | Connection String de tu clúster de MongoDB |
| `MONGODB_DB_NAME` | `super_agent_db` | El nombre donde se guardarán los documentos y el historial |

> ⚠️ **Nota Importante:** Render inyectará su propia variable reservada llamada `PORT` automáticamente en tiempo de ejecución. El archivo `main.py` ya está programado para detectar este puerto dinámico.

### 5. ¡A Deployar!
- Haz click en **Create Web Service**. 
- Verás unos logs instalando todo lo que pusimos en el _requirements.txt_.
- Cuando finalice, ¡tu agente estará vivo en la web! Podrás acceder y enviarle consultas de websocket usando en tu código frontend algo como: `wss://tu-app.onrender.com/ws/USUARIO_ID/SESION_ID`.

---

## 🧪 Pruebas Locales (Comandos)

**Crear y activar tu entorno virtual:**
```bash
python -m venv venv
# En Windows (Powershell):
.\venv\Scripts\Activate.ps1
# En Mac/Linux:
source venv/bin/activate
```

**Instalar dependencias:**
```bash
pip install -r requirements.txt
```

**Ejecutar el Agente:**
```bash
python main.py
```
> Arrancará e intentará inyectar el "Conocimiento de Ejemplo" de manera segura.

*¡Desarrollado con ❤️ para máxima escalabilidad corporativa!*

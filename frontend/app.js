/**
 * NOVA Chat Frontend - JavaScript Application
 * Handles WebSocket communication and document ingestion
 */

class NovaChat {
    constructor() {
        this.socket = null;
        this.userId = this.generateId();
        this.sessionId = this.generateId();
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        this.initElements();
        this.initEventListeners();
        this.connect();
    }
    
    generateId() {
        return 'xxxx-xxxx'.replace(/x/g, () => 
            Math.floor(Math.random() * 16).toString(16)
        );
    }
    
    initElements() {
        // Chat elements
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.chatForm = document.getElementById('chatForm');
        this.sendBtn = document.getElementById('sendBtn');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.thinkingText = document.getElementById('thinkingText');
        this.clearChat = document.getElementById('clearChat');
        
        // Connection elements
        this.connectionStatus = document.getElementById('connectionStatus');
        this.connectionText = document.getElementById('connectionText');
        this.sessionIdEl = document.getElementById('sessionId');
        
        // Document ingestion elements
        this.ingestForm = document.getElementById('ingestForm');
        this.documentContent = document.getElementById('documentContent');
        this.documentType = document.getElementById('documentType');
        this.documentName = document.getElementById('documentName');
        this.uploadStatus = document.getElementById('uploadStatus');
        this.statusIcon = document.getElementById('statusIcon');
        this.statusText = document.getElementById('statusText');
        this.statusDetail = document.getElementById('statusDetail');
        
        // Sidebar elements
        this.sidebar = document.getElementById('sidebar');
        this.menuBtn = document.getElementById('menuBtn');
        this.overlay = document.getElementById('overlay');
        
        // Update session ID display
        this.sessionIdEl.textContent = this.sessionId;
    }
    
    initEventListeners() {
        // Chat form submission
        this.chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });
        
        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 150) + 'px';
            this.sendBtn.disabled = !this.messageInput.value.trim() || !this.isConnected;
        });
        
        // Enter to send (Shift+Enter for new line)
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (this.messageInput.value.trim() && this.isConnected) {
                    this.sendMessage();
                }
            }
        });
        
        // Clear chat
        this.clearChat.addEventListener('click', () => {
            this.messagesContainer.innerHTML = '';
            this.addWelcomeMessage();
        });
        
        // Document ingestion form
        this.ingestForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.ingestDocument();
        });
        
        // Mobile sidebar toggle
        this.menuBtn.addEventListener('click', () => this.toggleSidebar());
        this.overlay.addEventListener('click', () => this.toggleSidebar());
    }
    
    toggleSidebar() {
        this.sidebar.classList.toggle('-translate-x-full');
        this.overlay.classList.toggle('hidden');
    }
    
    connect() {
        // Determine protocol based on current page protocol
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        const wsUrl = `${protocol}//${host}/ws/${this.userId}/${this.sessionId}`;
        
        this.updateConnectionStatus('connecting', 'Conectando...');
        
        try {
            this.socket = new WebSocket(wsUrl);
            
            this.socket.onopen = () => {
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('connected', 'Conectado');
                this.sendBtn.disabled = !this.messageInput.value.trim();
            };
            
            this.socket.onclose = () => {
                this.isConnected = false;
                this.updateConnectionStatus('disconnected', 'Desconectado');
                this.sendBtn.disabled = true;
                this.attemptReconnect();
            };
            
            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('error', 'Error de conexión');
            };
            
            this.socket.onmessage = (event) => {
                this.handleMessage(JSON.parse(event.data));
            };
        } catch (error) {
            console.error('Failed to connect:', error);
            this.updateConnectionStatus('error', 'Error de conexión');
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            this.updateConnectionStatus('connecting', `Reconectando (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            setTimeout(() => this.connect(), 3000);
        } else {
            this.updateConnectionStatus('error', 'No se pudo reconectar');
        }
    }
    
    updateConnectionStatus(status, text) {
        this.connectionText.textContent = text;
        
        const colors = {
            connecting: 'bg-yellow-500',
            connected: 'bg-green-500',
            disconnected: 'bg-red-500',
            error: 'bg-red-500'
        };
        
        this.connectionStatus.className = `absolute -bottom-1 -right-1 connection-dot ${colors[status] || 'bg-gray-500'}`;
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'thinking_start':
                this.showTypingIndicator(data.data.message || 'Analizando...');
                break;
                
            case 'final_response':
                this.hideTypingIndicator();
                this.addMessage('ai', data.data.response, data.data.timestamp);
                break;
                
            case 'thinking_end':
                this.hideTypingIndicator();
                break;
                
            case 'error':
                this.hideTypingIndicator();
                this.addMessage('ai', `⚠️ ${data.data.message}`, new Date().toISOString());
                break;
        }
    }
    
    showTypingIndicator(text = 'Analizando...') {
        this.thinkingText.textContent = text;
        this.typingIndicator.classList.remove('hidden');
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.typingIndicator.classList.add('hidden');
    }
    
    sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || !this.isConnected) return;
        
        // Add user message to chat
        this.addMessage('user', message);
        
        // Send via WebSocket
        this.socket.send(message);
        
        // Clear input
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.sendBtn.disabled = true;
    }
    
    addMessage(type, content, timestamp = new Date().toISOString()) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `flex items-start gap-3 ${type === 'user' ? 'flex-row-reverse' : ''}`;
        
        const time = new Date(timestamp).toLocaleTimeString('es-ES', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        
        if (type === 'user') {
            messageDiv.innerHTML = `
                <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center flex-shrink-0">
                    <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
                    </svg>
                </div>
                <div class="message-user rounded-2xl rounded-tr-md p-4 max-w-2xl">
                    <p class="text-white whitespace-pre-wrap">${this.escapeHtml(content)}</p>
                    <p class="text-xs text-indigo-200 mt-2 text-right">${time}</p>
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center flex-shrink-0">
                    <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
                    </svg>
                </div>
                <div class="message-ai rounded-2xl rounded-tl-md p-4 max-w-2xl">
                    <div class="text-slate-200 whitespace-pre-wrap">${this.formatMessage(content)}</div>
                    <p class="text-xs text-slate-500 mt-2">${time}</p>
                </div>
            `;
        }
        
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    formatMessage(content) {
        // Simple markdown-like formatting
        let formatted = this.escapeHtml(content);
        
        // Bold text **text**
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong class="text-indigo-400">$1</strong>');
        
        // Emoji support (already works in HTML)
        
        // Code blocks `code`
        formatted = formatted.replace(/`([^`]+)`/g, '<code class="bg-slate-700 px-1 rounded text-cyan-400">$1</code>');
        
        // Line breaks
        formatted = formatted.replace(/\n/g, '<br>');
        
        return formatted;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }, 100);
    }
    
    addWelcomeMessage() {
        const welcomeHtml = `
            <div class="flex items-start gap-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center flex-shrink-0">
                    <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
                    </svg>
                </div>
                <div class="message-ai rounded-2xl rounded-tl-md p-4 max-w-2xl">
                    <p class="text-slate-200">
                        ¡Hola! 👋 Soy <strong class="text-indigo-400">NOVA</strong>, tu Super Asistente Inteligente con memoria persistente.
                    </p>
                    <p class="text-slate-300 mt-2">
                        Puedo ayudarte a buscar productos, realizar investigaciones web en tiempo real, y responder tus preguntas usando mi base de conocimiento. 
                    </p>
                    <p class="text-slate-400 mt-2 text-sm">
                        💡 Tip: Usa el panel izquierdo para añadir documentos a mi base de conocimiento y podré usarlos para responder tus preguntas.
                    </p>
                </div>
            </div>
        `;
        this.messagesContainer.innerHTML = welcomeHtml;
    }
    
    async ingestDocument() {
        const content = this.documentContent.value.trim();
        const type = this.documentType.value;
        const name = this.documentName.value.trim();
        
        if (!content) {
            this.showUploadStatus('error', 'Error', 'El contenido es requerido');
            return;
        }
        
        this.showUploadStatus('loading', 'Subiendo...', 'Procesando documento');
        
        try {
            // Use relative path for API calls to adapt to any deployment
            const response = await fetch('/ingest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    content: content,
                    metadata: {
                        type: type,
                        name: name || undefined,
                        source: 'frontend_upload',
                        timestamp: new Date().toISOString()
                    }
                })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showUploadStatus('success', '¡Documento subido!', `${result.chunks_created} chunks creados`);
                this.documentContent.value = '';
                this.documentName.value = '';
                
                // Add a message to chat about the upload
                this.addMessage('ai', `📚 He añadido un nuevo documento a mi base de conocimiento (${result.chunks_created} chunks). ¡Ahora puedo responder preguntas sobre este contenido!`);
            } else {
                this.showUploadStatus('error', 'Error', result.message || 'Error desconocido');
            }
        } catch (error) {
            console.error('Ingest error:', error);
            this.showUploadStatus('error', 'Error de conexión', 'No se pudo conectar al servidor');
        }
    }
    
    showUploadStatus(type, text, detail) {
        this.uploadStatus.classList.remove('hidden');
        this.statusText.textContent = text;
        this.statusDetail.textContent = detail;
        
        const icons = {
            loading: `
                <svg class="w-5 h-5 text-indigo-400 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
            `,
            success: `
                <svg class="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                </svg>
            `,
            error: `
                <svg class="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                </svg>
            `
        };
        
        const bgColors = {
            loading: 'bg-indigo-500/20',
            success: 'bg-green-500/20',
            error: 'bg-red-500/20'
        };
        
        this.statusIcon.className = `w-8 h-8 rounded-full flex items-center justify-center ${bgColors[type] || 'bg-slate-500/20'}`;
        this.statusIcon.innerHTML = icons[type] || '';
        
        // Auto-hide success after 5 seconds
        if (type === 'success') {
            setTimeout(() => {
                this.uploadStatus.classList.add('hidden');
            }, 5000);
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.novaChat = new NovaChat();
});

// Global function for sidebar toggle (used by overlay onclick)
function toggleSidebar() {
    if (window.novaChat) {
        window.novaChat.toggleSidebar();
    }
}

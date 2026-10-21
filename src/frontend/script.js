const queryInput = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');
const chatMessages = document.getElementById('chat-messages');
const graphEvidence = document.getElementById('graph-evidence');
const vectorEvidence = document.getElementById('vector-evidence');

function appendMessage(sender, text) {
    const div = document.createElement('div');
    div.className = `message ${sender}-message`;
    
    const avatar = sender === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
    
    div.innerHTML = `
        <div class="avatar">${avatar}</div>
        <div class="bubble">${text}</div>
    `;
    
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function updateEvidence(element, data, isGraph) {
    if (!data || data.length === 0) {
        element.innerHTML = '<p class="placeholder">No evidence found.</p>';
        return;
    }
    
    let html = '<ul>';
    data.forEach(item => {
        if (isGraph) {
            html += `<li><code>${item}</code></li>`;
        } else {
            // Trim long text for vector chunk
            const snippet = item.substring(0, 150) + '...';
            html += `<li>"${snippet}"</li>`;
        }
    });
    html += '</ul>';
    element.innerHTML = html;
}

async function handleSend() {
    const query = queryInput.value.trim();
    if (!query) return;
    
    // UI Update: Add user message
    appendMessage('user', query);
    queryInput.value = '';
    
    // UI Update: Loading state
    const loadingId = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.className = `message ai-message`;
    loadingDiv.id = loadingId;
    loadingDiv.innerHTML = `
        <div class="avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="bubble"><i class="fa-solid fa-circle-notch fa-spin"></i> Searching Knowledge Base...</div>
    `;
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Reset Evidence Panels
    graphEvidence.innerHTML = '<p class="placeholder"><i class="fa-solid fa-circle-notch fa-spin"></i> Querying Neo4j...</p>';
    vectorEvidence.innerHTML = '<p class="placeholder"><i class="fa-solid fa-circle-notch fa-spin"></i> Querying Qdrant...</p>';
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || "Server error");
        }
        
        // Remove Loading
        document.getElementById(loadingId).remove();
        
        // Render AI Answer
        appendMessage('ai', data.answer);
        
        // Render Evidence
        updateEvidence(graphEvidence, data.graph_context, true);
        updateEvidence(vectorEvidence, data.vector_context, false);
        
    } catch (error) {
        document.getElementById(loadingId).remove();
        appendMessage('ai', `Error: ${error.message}`);
        graphEvidence.innerHTML = '<p class="placeholder" style="color: #ef4444;">Failed to load.</p>';
        vectorEvidence.innerHTML = '<p class="placeholder" style="color: #ef4444;">Failed to load.</p>';
    }
}

sendBtn.addEventListener('click', handleSend);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSend();
});

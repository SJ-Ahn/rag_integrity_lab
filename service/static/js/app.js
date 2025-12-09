/**
 * RAG Assistant - Clean Paper Logic
 * Handles Chat, History (LocalStorage), and Markdown Rendering
 */

// DOM Elements
const conversationEl = document.getElementById("conversation");
const scrollContainer = document.querySelector(".chat-container");
const chatForm = document.getElementById("chat-form");
const questionEl = document.getElementById("question");
const newChatBtn = document.getElementById("new-chat-btn");
const historyListEl = document.getElementById("history-list");

// State
let currentSessionId = null;
let conversationHistory = []; // Local in-memory history for context
let isProcesssing = false;

// --- History Management (LocalStorage) ---

const generateId = () => Date.now().toString(36) + Math.random().toString(36).substr(2);

const getSavedSessions = () => {
    const sessions = localStorage.getItem("rag_sessions");
    return sessions ? JSON.parse(sessions) : {};
};

const saveSession = (id, messages) => {
    const sessions = getSavedSessions();
    // Generate a title from the first user message if possible
    let title = "New Conversation";
    const firstUserMsg = messages.find(m => m.role === "user");
    if (firstUserMsg) {
        title = firstUserMsg.content.substring(0, 30);
    }

    sessions[id] = {
        id,
        title,
        updatedAt: Date.now(),
        messages
    };
    localStorage.setItem("rag_sessions", JSON.stringify(sessions));
    renderHistoryList();
};

const deleteSession = (id, event) => {
    event.stopPropagation();
    if (!confirm("대화를 삭제하시겠습니까?")) return;
    
    const sessions = getSavedSessions();
    delete sessions[id];
    localStorage.setItem("rag_sessions", JSON.stringify(sessions));
    
    if (currentSessionId === id) {
        startNewChat();
    } else {
        renderHistoryList();
    }
};

const loadSession = (id) => {
    const sessions = getSavedSessions();
    const session = sessions[id];
    if (!session) return;

    currentSessionId = id;
    conversationHistory = session.messages || [];
    
    // UI Reset
    conversationEl.innerHTML = "";
    
    // Render Messages
    if (conversationHistory.length === 0) {
        renderWelcomeMessage();
    } else {
        conversationHistory.forEach(msg => {
            renderMessage(msg.role, msg.content, msg.sources);
        });
    }
    
    // Scroll to bottom
    scrollToBottom();
    
    renderHistoryList();
};

const renderHistoryList = () => {
    const sessions = getSavedSessions();
    const sortedargs = Object.values(sessions).sort((a, b) => b.updatedAt - a.updatedAt);
    
    historyListEl.innerHTML = "";
    
    sortedargs.forEach(session => {
        const li = document.createElement("li");
        li.classList.add("history-item");
        if (session.id === currentSessionId) {
            li.classList.add("active");
        }
        li.textContent = session.title || "새 대화";
        li.onclick = () => loadSession(session.id);
        
        historyListEl.appendChild(li);
    });
};

const startNewChat = () => {
    currentSessionId = generateId();
    conversationHistory = [];
    conversationEl.innerHTML = "";
    renderWelcomeMessage();
    renderHistoryList();
};

// --- UI Rendering ---

const renderWelcomeMessage = () => {
    // Hardcoded HTML for Welcome (matches index.html template)
    // Note: index.html already has this as initial state, but we might wipe it.
    // We will re-inject it.
    const html = `
        <div class="message-row">
            <div class="avatar ai">
                <!-- Option 3 Icon -->
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
            </div>
            <div class="message-content">
                <div class="ai-text">
                    <p>안녕하세요! <strong>AWS 문서 기반 AI 어시스턴트</strong>입니다.<br>
                    EC2, RDS, Aurora 등 AWS 서비스에 대해 무엇이든 물어보세요.</p>
                </div>
            </div>
        </div>
    `;
    conversationEl.insertAdjacentHTML('beforeend', html);
};

const renderMessage = (role, content, sources = null) => {
    const row = document.createElement("div");
    row.classList.add("message-row");
    if (role === "user") {
        row.classList.add("user-row");
    }

    // Avatar (only for AI)
    if (role === "assistant") {
        const avatar = document.createElement("div");
        avatar.className = "avatar ai";
        avatar.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>`;
        row.appendChild(avatar);
    }

    // Message Content
    const msgContent = document.createElement("div");
    msgContent.className = "message-content";

    const textDiv = document.createElement("div");
    textDiv.className = role === "user" ? "user-bubble" : "ai-text";

    if (role === "assistant" && typeof marked !== "undefined") {
        try {
            textDiv.innerHTML = marked.parse(content);
        } catch (e) {
            textDiv.textContent = content;
        }
    } else {
        textDiv.textContent = content;
    }

    msgContent.appendChild(textDiv);
    
    // Render Sources if AI and exists
    if (role === "assistant" && sources && sources.length > 0) {
        msgContent.appendChild(createSourcesElement(sources));
    }

    row.appendChild(msgContent);
    
    // User Avatar (Optional, placeholder 'U')
    if (role === "user") {
        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.style.background = "#ddd";
        avatar.textContent = "U";
        row.appendChild(avatar);
    }

    conversationEl.appendChild(row);
    scrollToBottom();
    return row;
};

const scrollToBottom = () => {
    if (scrollContainer) {
        // Use requestAnimationFrame to ensure DOM is updated
        requestAnimationFrame(() => {
            scrollContainer.scrollTop = scrollContainer.scrollHeight;
        });
    }
};

const createSourcesElement = (sources) => {
    const section = document.createElement("div");
    section.className = "citation-section";
    
    const header = document.createElement("div");
    header.className = "citation-header";
    header.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/></svg>
        관련 문서 (${sources.length})
    `;
    section.appendChild(header);
    
    const grid = document.createElement("div");
    grid.className = "citation-grid";
    
    sources.slice(0, 5).forEach(src => { // Limit to 5 cards
        const card = document.createElement("a");
        card.className = "citation-card";
        
        // Auto-navigate to page if available
        let href = src.ref || "#";
        if (src.page) {
            href += `#page=${src.page}`;
        }
        card.href = href;
        
        card.target = "_blank";
        
        // Title (Doc ID)
        const titleDiv = document.createElement("div");
        titleDiv.className = "cit-title";
        titleDiv.textContent = (src.doc_id || "Document").replace(/^aws\//, "");
        card.appendChild(titleDiv);
        
        // Badge (Page)
        if (src.page) {
            const badge = document.createElement("div");
            badge.className = "cit-badge";
            badge.textContent = `Page ${src.page}`;
            card.appendChild(badge);
        }
        
        // Preview
        if (src.anchor) {
            const preview = document.createElement("div");
            preview.className = "cit-preview";
            preview.textContent = src.anchor.trim();
            card.appendChild(preview);
        }
        
        grid.appendChild(card);
    });
    
    section.appendChild(grid);
    return section;
};

const startThinking = () => {
    const row = document.createElement("div");
    row.classList.add("message-row", "thinking-row");
    row.innerHTML = `
        <div class="avatar ai">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
        </div>
        <div class="message-content">
            <div class="ai-text" style="color:#aaa;">
                <span class="thinking-text">생각 중... (0.0s)</span>
            </div>
        </div>
    `;
    conversationEl.appendChild(row);
    scrollToBottom();
    
    const startTime = Date.now();
    const textSpan = row.querySelector(".thinking-text");
    
    const intervalId = setInterval(() => {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        if (textSpan) {
            textSpan.textContent = `생각 중... (${elapsed}s)`;
        }
    }, 100);

    return { row, intervalId };
};

const stopThinking = (thinkingObj) => {
    if (thinkingObj) {
        if (thinkingObj.intervalId) clearInterval(thinkingObj.intervalId);
        if (thinkingObj.row) thinkingObj.row.remove();
    }
};


// --- Event Listeners ---

chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (isProcesssing) return;
    
    const question = questionEl.value.trim();
    if (!question) return;
    
    if (!currentSessionId) startNewChat();
    
    // User Message
    renderMessage("user", question);
    conversationHistory.push({ role: "user", content: question });
    saveSession(currentSessionId, conversationHistory);
    
    questionEl.value = "";
    isProcesssing = true;
    
    // Thinking
    const thinkingRow = startThinking();
    
    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                history: conversationHistory // Send full history for context
            })
        });
        
        if (!response.ok) throw new Error("Network response was not ok");
        
        const data = await response.json();
        
        stopThinking(thinkingRow);
        
        // AI Message
        renderMessage("assistant", data.answer, data.sources);
        
        conversationHistory.push({ 
            role: "assistant", 
            content: data.answer,
            sources: data.sources 
        });
        
        saveSession(currentSessionId, conversationHistory);
        
    } catch (err) {
        stopThinking(thinkingRow);
        renderMessage("assistant", "죄송합니다. 오류가 발생했습니다.");
        console.error(err);
    } finally {
        isProcesssing = false;
        // Scroll
        scrollToBottom();
    }
});

questionEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        chatForm.requestSubmit();
    }
});

newChatBtn.addEventListener("click", startNewChat);

// --- Initialization ---

window.addEventListener("DOMContentLoaded", () => {
    // Check if we should load last session? 
    // For now, start new or load most recent?
    // Let's start new by default to show Welcome, but populate Sidebar.
    
    const sessions = getSavedSessions();
    const sorted = Object.values(sessions).sort((a,b) => b.updatedAt - a.updatedAt);
    
    if (sorted.length > 0) {
        // Optional: Load most recent? Or just show new?
        // Let's Start New Chat but show history.
        // Actually, if I reload, I might want to see my last chat.
        // Let's load the most recent one.
        loadSession(sorted[0].id);
    } else {
        startNewChat();
    }
    
    renderHistoryList();
});

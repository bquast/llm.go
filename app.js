document.addEventListener('DOMContentLoaded', async () => {
    const sysStatus = document.getElementById('systemStatus');
    const loadBtn = document.getElementById('loadBtn');
    
    // UI Elements
    const loaderState = document.getElementById('loaderState');
    const chatState = document.getElementById('chatState');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const chatHistory = document.getElementById('chatHistory');
    const promptInput = document.getElementById('promptInput');
    const generateBtn = document.getElementById('generateBtn');

    // 1. Init WebAssembly
    const go = new Go();
    try {
        const result = await WebAssembly.instantiateStreaming(fetch("main.wasm"), go.importObject);
        go.run(result.instance);
        sysStatus.textContent = "WASM Ready";
        sysStatus.className = "chip status-published";
        loadBtn.disabled = false;
    } catch (err) {
        console.error("WASM load error:", err);
        sysStatus.textContent = "WASM Error";
        sysStatus.className = "chip status-error";
    }

    // 2. Fetch and Cache Model
    loadBtn.addEventListener('click', async () => {
        loadBtn.style.display = 'none';
        progressContainer.style.display = 'block';
        
        const MODEL_URL = 'https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors';
        
        try {
            const cache = await caches.open('llm-go-cache');
            let cachedResponse = await cache.match(MODEL_URL);
            let arrayBuffer;

            if (cachedResponse) {
                progressText.textContent = "Loading from local cache...";
                progressBar.style.width = "100%";
                arrayBuffer = await cachedResponse.arrayBuffer();
            } else {
                // Fetch with progress tracking
                const response = await fetch(MODEL_URL);
                const contentLength = response.headers.get('content-length');
                const total = parseInt(contentLength, 10);
                
                let loaded = 0;
                const reader = response.body.getReader();
                const chunks = [];

                while(true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    
                    chunks.push(value);
                    loaded += value.length;
                    
                    if (total) {
                        const percent = Math.round((loaded / total) * 100);
                        progressBar.style.width = `${percent}%`;
                        progressText.textContent = `Downloading: ${percent}% (${(loaded/1024/1024).toFixed(1)}MB / ${(total/1024/1024).toFixed(1)}MB)`;
                    }
                }
                
                // Concatenate chunks
                const uint8Array = new Uint8Array(loaded);
                let position = 0;
                for (let chunk of chunks) {
                    uint8Array.set(chunk, position);
                    position += chunk.length;
                }
                arrayBuffer = uint8Array.buffer;

                // Cache for next time
                cache.put(MODEL_URL, new Response(arrayBuffer));
            }

            // 3. Pass to Go WebAssembly
            progressText.textContent = "Transferring weights to WebAssembly memory...";
            const uint8View = new Uint8Array(arrayBuffer);
            
            // Call Go func
            const goOutput = window.initModel(uint8View);
            
            console.log(goOutput);
            loaderState.style.display = 'none';
            chatState.style.display = 'flex';

        } catch (err) {
            console.error("Model fetch error:", err);
            progressText.textContent = "Error: " + err.message;
            progressText.style.color = "var(--error)";
            progressBar.style.backgroundColor = "var(--error)";
        }
    });

    // 4. Handle Inference
    async function handleGenerate() {
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        addMessage("User", prompt);
        promptInput.value = '';
        generateBtn.disabled = true;
        generateBtn.textContent = 'Running...';

        const loadingId = addMessage("SmolLM2", "Generating...", false, true);

        try {
            // Call exported Go generation wrapper
            const response = await window.generateText(prompt);
            updateMessage(loadingId, response);
        } catch (err) {
            updateMessage(loadingId, "Error: " + err);
        } finally {
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate';
            promptInput.focus();
        }
    }

    generateBtn.addEventListener('click', handleGenerate);
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleGenerate();
        }
    });

    let msgCounter = 0;
    function addMessage(sender, text, isError = false, isLoading = false) {
        msgCounter++;
        const id = `msg-${msgCounter}`;
        const div = document.createElement('div');
        div.className = 'message';
        div.id = id;
        
        let chipClass = sender === "User" ? "" : "status-published";
        div.innerHTML = `<span class="chip ${chipClass}">${sender}</span><p></p>`;
        
        if (isLoading) div.querySelector('p').innerHTML = `<i>${escapeHtml(text)}</i>`;
        else div.querySelector('p').innerText = text;
        
        chatHistory.appendChild(div);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return id;
    }

    function updateMessage(id, newText) {
        const msgDiv = document.getElementById(id);
        if (msgDiv) {
            msgDiv.querySelector('p').innerHTML = `<pre><code>${escapeHtml(newText)}</code></pre>`;
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
    }

    function escapeHtml(unsafe) {
        return unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }
});
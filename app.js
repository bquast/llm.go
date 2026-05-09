document.addEventListener('DOMContentLoaded', async () => {
    const sysStatus = document.getElementById('systemStatus');
    const loadBtn = document.getElementById('loadBtn');
    const loaderState = document.getElementById('loaderState');
    const chatState = document.getElementById('chatState');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const chatHistory = document.getElementById('chatHistory');
    const promptInput = document.getElementById('promptInput');
    const generateBtn = document.getElementById('generateBtn');

    let currentResponseId = null;

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

    loadBtn.addEventListener('click', async () => {
        loadBtn.style.display = 'none';
        progressContainer.style.display = 'block';
        
        const MODEL_URL = 'https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors';
        const TOK_URL = 'https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json';
        
        try {
            // 1. Fetch Tokenizer Vocabulary
            progressText.textContent = "Loading Tokenizer...";
            const tokRes = await fetch(TOK_URL);
            const tokJson = await tokRes.json();
            const vocabString = JSON.stringify(tokJson.model.vocab);

            // 2. Fetch Model Cache
            const cache = await caches.open('llm-go-cache');
            let cachedResponse = await cache.match(MODEL_URL);
            let arrayBuffer;

            if (cachedResponse) {
                progressText.textContent = "Loading weights from local cache...";
                progressBar.style.width = "100%";
                arrayBuffer = await cachedResponse.arrayBuffer();
            } else {
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
                        progressText.textContent = `Downloading Weights: ${percent}%`;
                    }
                }
                
                const uint8Array = new Uint8Array(loaded);
                let position = 0;
                for (let chunk of chunks) {
                    uint8Array.set(chunk, position);
                    position += chunk.length;
                }
                arrayBuffer = uint8Array.buffer;
                cache.put(MODEL_URL, new Response(arrayBuffer));
            }

            // 3. Mount to WebAssembly
            progressText.textContent = "Unpacking bfloat16 Tensors in Go...";
            // Give UI a tick to render text
            setTimeout(() => {
                const uint8View = new Uint8Array(arrayBuffer);
                const goOutput = window.initModel(uint8View, vocabString);
                console.log(goOutput);
                
                loaderState.style.display = 'none';
                chatState.style.display = 'flex';
            }, 50);

        } catch (err) {
            console.error("Setup error:", err);
            progressText.textContent = "Error: " + err.message;
            progressText.style.color = "var(--error)";
        }
    });

    // Called dynamically by Go WebAssembly
    window.appendToken = (text) => {
        if (!currentResponseId) return;
        const msgDiv = document.getElementById(currentResponseId);
        if (msgDiv) {
            const p = msgDiv.querySelector('p');
            if (p.innerHTML.includes('<i>')) p.innerHTML = ''; // clear loading state
            p.innerHTML += escapeHtml(text).replace(/\n/g, '<br/>');
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
    };

    async function handleGenerate() {
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        addMessage("User", prompt);
        promptInput.value = '';
        generateBtn.disabled = true;
        generateBtn.textContent = 'Running...';

        currentResponseId = addMessage("SmolLM2", "Generating (pure Go)...", false, true);

        try {
            await window.generateText(prompt);
        } catch (err) {
            const p = document.getElementById(currentResponseId).querySelector('p');
            p.innerHTML = "Error: " + err;
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

    function escapeHtml(unsafe) {
        return unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }
});
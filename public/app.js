/**
 * Apple iOS Siri-Style Client Application
 */

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const chatStream = document.getElementById("chatStream");
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const siriOrbBtn = document.getElementById("siriOrbBtn");
    const siriListeningOverlay = document.getElementById("siriListeningOverlay");
    const siriLiveTranscript = document.getElementById("siriLiveTranscript");
    const stopVoiceBtn = document.getElementById("stopVoiceBtn");
    const clearChatBtn = document.getElementById("clearChatBtn");
    const autoVoiceToggle = document.getElementById("autoVoiceToggle");
    const iosStatus = document.getElementById("iosStatus");
    const suggestionChips = document.querySelectorAll(".siri-chip");

    // Modal
    const inspectorModal = document.getElementById("inspectorModal");
    const closeModalBtn = document.getElementById("closeModalBtn");
    const modalProbsList = document.getElementById("modalProbsList");

    let isRecording = false;
    let recognition = null;
    let finalTranscript = "";

    // 1. Initialize Web Speech API
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.lang = "en-US";

        recognition.onstart = () => {
            isRecording = true;
            siriOrbBtn.classList.add("recording");
            siriListeningOverlay.style.display = "flex";
            siriLiveTranscript.innerText = "Siri is listening... Speak now";
            iosStatus.innerText = "Listening...";
            iosStatus.style.color = "#FF375F";
        };

        recognition.onresult = (event) => {
            let interim = "";
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                } else {
                    interim += event.results[i][0].transcript;
                }
            }
            const current = finalTranscript || interim;
            siriLiveTranscript.innerText = `"${current}"`;
            userInput.value = current;
        };

        recognition.onerror = (event) => {
            console.error("Speech Recognition Error:", event.error);
            stopRecordingUI();
            let msg = event.error === "not-allowed" ? "Microphone permission denied" : "Speech error: " + event.error;
            iosStatus.innerText = msg;
            iosStatus.style.color = "#FF9F0A";
        };

        recognition.onend = () => {
            stopRecordingUI();
            const text = userInput.value.trim();
            if (text && text !== "Siri is listening... Speak now") {
                submitQuery(text);
            }
        };
    } else {
        siriOrbBtn.title = "Web Speech API not supported in this browser";
        iosStatus.innerText = "Type below (Use Chrome/Edge for voice)";
    }

    function toggleRecording() {
        if (!recognition) {
            alert("Web Speech API is not supported in this browser. Please use Chrome or Edge, or type your query.");
            return;
        }
        if (isRecording) {
            recognition.stop();
        } else {
            finalTranscript = "";
            userInput.value = "";
            try {
                recognition.start();
            } catch (e) {
                console.error(e);
            }
        }
    }

    function stopRecordingUI() {
        isRecording = false;
        siriOrbBtn.classList.remove("recording");
        siriListeningOverlay.style.display = "none";
        iosStatus.innerText = "Tap orb or speak";
        iosStatus.style.color = "var(--ios-subtext)";
    }

    siriOrbBtn.addEventListener("click", toggleRecording);
    stopVoiceBtn.addEventListener("click", () => {
        if (recognition && isRecording) {
            recognition.stop();
        }
    });

    // 2. Submit Query Form
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (text) {
            submitQuery(text);
        }
    });

    // 3. Quick Chips
    suggestionChips.forEach((chip) => {
        chip.addEventListener("click", () => {
            const q = chip.getAttribute("data-query");
            userInput.value = q;
            submitQuery(q);
        });
    });

    // 4. Submit Query Pipeline
    async function submitQuery(queryText) {
        if (!queryText) return;

        // Render User Bubble
        appendUserMessage(queryText);
        userInput.value = "";
        userInput.focus();

        // Thinking Indicator
        const typingId = showThinkingIndicator();
        iosStatus.innerText = "Siri Thinking...";
        iosStatus.style.color = "#BF5AF2";
        sendBtn.disabled = true;

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: queryText, threshold: 0.55 })
            });

            removeThinkingIndicator(typingId);

            if (!res.ok) throw new Error("HTTP error " + res.status);
            const data = await res.json();

            // Render Siri Response Card
            appendSiriCard(data.response, data.intent, data.confidence, data.source, data.top_predictions);

            // Auto-speak voice response
            if (autoVoiceToggle.checked) {
                speakText(data.response);
            }
        } catch (err) {
            removeThinkingIndicator(typingId);
            console.error("Query error:", err);
            appendSiriCard("I am having trouble connecting to the network right now. Please check your connection and try again.", "network_error", 0, "System");
        } finally {
            iosStatus.innerText = "Tap orb or speak";
            iosStatus.style.color = "var(--ios-subtext)";
            sendBtn.disabled = false;
        }
    }

    // 5. Render User Message
    function appendUserMessage(text) {
        const row = document.createElement("div");
        row.className = "siri-card-row user-row";

        const bubble = document.createElement("div");
        bubble.className = "user-bubble";
        bubble.innerText = text;

        row.appendChild(bubble);
        chatStream.appendChild(row);
        chatStream.scrollTop = chatStream.scrollHeight;
    }

    // 6. Render Siri Card
    function appendSiriCard(response, intent, confidence, source, topPreds = []) {
        const row = document.createElement("div");
        row.className = "siri-card-row bot-row";

        const card = document.createElement("div");
        card.className = "siri-card";

        const textP = document.createElement("div");
        textP.className = "siri-card-text";
        textP.innerHTML = escapeHtml(response);

        const footer = document.createElement("div");
        footer.className = "siri-card-footer";

        // Source & Intent Badge
        const confPct = Math.round((confidence || 0) * 100);
        const sourcePill = document.createElement("span");
        sourcePill.className = "siri-source-pill";
        sourcePill.innerText = `⚡ ${source || 'Bi-LSTM'} &bull; ${intent || 'intent'} (${confPct}%)`;
        sourcePill.title = "Tap to inspect neural probability distribution";

        if (topPreds && topPreds.length > 0) {
            sourcePill.addEventListener("click", () => {
                openInspector(intent, confPct, topPreds);
            });
        }

        // Play Voice Button
        const playBtn = document.createElement("button");
        playBtn.className = "btn-play-voice";
        playBtn.innerText = "🔊 Speak";
        playBtn.addEventListener("click", () => {
            speakText(response);
        });

        footer.appendChild(sourcePill);
        footer.appendChild(playBtn);

        card.appendChild(textP);
        card.appendChild(footer);
        row.appendChild(card);
        chatStream.appendChild(row);
        chatStream.scrollTop = chatStream.scrollHeight;
    }

    function showThinkingIndicator() {
        const id = "think-" + Date.now();
        const row = document.createElement("div");
        row.id = id;
        row.className = "siri-card-row bot-row";
        row.innerHTML = `
            <div class="siri-card" style="padding: 12px 18px; width: fit-content;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <div class="siri-mini-orb"></div>
                    <span style="font-size:0.9rem; color:var(--ios-subtext);">Siri is finding an answer...</span>
                </div>
            </div>
        `;
        chatStream.appendChild(row);
        chatStream.scrollTop = chatStream.scrollHeight;
        return id;
    }

    function removeThinkingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    // 7. Modal Inspector
    function openInspector(intent, confPct, topPreds) {
        let html = "";
        topPreds.forEach((p) => {
            const pct = Math.round(p.confidence * 100);
            html += `
            <div class="modal-prob-row">
                <span class="modal-prob-tag">${p.intent}</span>
                <div class="modal-prob-bar-box">
                    <div class="modal-prob-bar-bg">
                        <div class="modal-prob-bar-fill" style="width: ${pct}%"></div>
                    </div>
                    <span style="font-weight: 700; font-size: 0.8rem; color:#FFFFFF;">${pct}%</span>
                </div>
            </div>`;
        });
        modalProbsList.innerHTML = html;
        inspectorModal.style.display = "flex";
    }

    closeModalBtn.addEventListener("click", () => {
        inspectorModal.style.display = "none";
    });
    inspectorModal.addEventListener("click", (e) => {
        if (e.target === inspectorModal) {
            inspectorModal.style.display = "none";
        }
    });

    // 8. Clear Chat
    clearChatBtn.addEventListener("click", () => {
        chatStream.innerHTML = `
            <div class="siri-card-row bot-row">
                <div class="siri-card">
                    <div class="siri-intro-header">
                        <div class="siri-intro-orb"></div>
                        <div class="siri-intro-title">What can I help you with?</div>
                    </div>
                    <p class="siri-card-text">Dialogue cleared. Tap the Siri orb or type any question below!</p>
                </div>
            </div>
        `;
    });

    function escapeHtml(str) {
        return str.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, "<br>");
    }
});

// Global Siri Speech Synthesis
function speakText(text) {
    if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
        const clean = text.replace(/<[^>]*>?/gm, "").replace(/[^\w\s.,!?'"-]/g, "");
        const utterance = new SpeechSynthesisUtterance(clean);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;

        // Try to pick natural voice if available
        const voices = window.speechSynthesis.getVoices();
        const siriVoice = voices.find(v => v.name.includes("Samantha") || v.name.includes("Siri") || v.name.includes("Natural") || (v.lang === "en-US" && !v.name.includes("Zira")));
        if (siriVoice) {
            utterance.voice = siriVoice;
        }

        window.speechSynthesis.speak(utterance);
    }
}

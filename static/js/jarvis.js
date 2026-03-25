(function () {
    function speak(text) {
        if (!window.speechSynthesis) return;
        window.speechSynthesis.cancel();
        const speech = new SpeechSynthesisUtterance(text);
        speech.rate = 1;
        speech.pitch = 1;
        window.speechSynthesis.speak(speech);
    }

    let lastResponseIndex = -1;
    function randomResponse(list) {
        let index;
        do {
            index = Math.floor(Math.random() * list.length);
        } while (index === lastResponseIndex && list.length > 1);
        lastResponseIndex = index;
        return list[index];
    }

    function sayRandom(list) {
        speak(randomResponse(list));
    }

    function triggerAnalyze() {
        const analyzeBtn = document.getElementById("analyzeBtn");
        if (analyzeBtn && !analyzeBtn.disabled) {
            analyzeBtn.click();
            return true;
        }
        speak("Please upload both scene and target images first.");
        return false;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const jarvisBtn = document.createElement("button");
    jarvisBtn.type = "button";
    jarvisBtn.id = "jarvisToggle";
    jarvisBtn.textContent = "Jarvis: Off";
    jarvisBtn.style.position = "fixed";
    jarvisBtn.style.bottom = "20px";
    jarvisBtn.style.left = "20px";
    jarvisBtn.style.padding = "10px 14px";
    jarvisBtn.style.background = "linear-gradient(135deg, rgba(72,220,255,0.86), rgba(0,168,225,0.86))";
    jarvisBtn.style.color = "#06101f";
    jarvisBtn.style.border = "1px solid rgba(190,245,255,0.9)";
    jarvisBtn.style.borderRadius = "10px";
    jarvisBtn.style.cursor = "pointer";
    jarvisBtn.style.zIndex = "9999";
    jarvisBtn.style.fontWeight = "700";
    jarvisBtn.style.fontFamily = "Space Grotesk, sans-serif";
    jarvisBtn.style.userSelect = "none";
    jarvisBtn.style.boxShadow = "0 12px 26px rgba(0, 130, 190, 0.28)";
    document.body.appendChild(jarvisBtn);

    if (!SpeechRecognition) {
        jarvisBtn.textContent = "Jarvis: Unsupported";
        jarvisBtn.disabled = true;
        jarvisBtn.style.opacity = "0.7";
        jarvisBtn.style.cursor = "not-allowed";
        jarvisBtn.title = "SpeechRecognition is not supported in this browser.";
        console.warn("SpeechRecognition is not supported in this browser.");
        return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    let jarvisEnabled = false;
    let listeningForCommand = false;
    let isRecognizing = false;
    let isStarting = false;
    let shouldAutoRestart = false;
    let wakeUntil = 0;

    function setButtonState(text, active) {
        jarvisBtn.textContent = text;
        jarvisBtn.style.filter = active ? "drop-shadow(0 0 10px rgba(72,220,255,0.65))" : "none";
    }

    async function startJarvis() {
        if (isStarting || isRecognizing || jarvisEnabled) return;
        isStarting = true;
        try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
            shouldAutoRestart = true;
            jarvisEnabled = true;
            setButtonState("Jarvis: Starting...", true);
            recognition.start();
        } catch (err) {
            jarvisEnabled = false;
            shouldAutoRestart = false;
            setButtonState("Jarvis: Off", false);
            console.log("Microphone access denied:", err);
            alert("Please allow microphone access to use Jarvis.");
        } finally {
            isStarting = false;
        }
    }

    function stopJarvis() {
        shouldAutoRestart = false;
        jarvisEnabled = false;
        listeningForCommand = false;
        wakeUntil = 0;
        setButtonState("Jarvis: Off", false);
        if (isRecognizing) {
            try {
                recognition.stop();
            } catch (err) {
                console.log("Stop skipped:", err);
            }
        }
    }

    jarvisBtn.addEventListener("click", function () {
        if (jarvisEnabled || isRecognizing || isStarting) {
            stopJarvis();
        } else {
            startJarvis();
        }
    });

    recognition.onstart = function () {
        isRecognizing = true;
        jarvisEnabled = true;
        setButtonState("Jarvis: On", true);
        console.log("Jarvis is listening...");
        speak("Jarvis activated. Say Jarvis, then your command.");
    };

    recognition.onerror = function (event) {
        console.log("Speech error:", event.error);
        if (event.error === "not-allowed" || event.error === "service-not-allowed") {
            stopJarvis();
            alert("Microphone permission is blocked. Please enable it in browser settings.");
            return;
        }
        if (event.error === "aborted") {
            return;
        }
    };

    function handleCommand(transcript) {
        if (transcript.includes("run")) {
            sayRandom([
                "Initiating quantum search now.",
                "Starting Grover's search algorithm.",
                "Activating quantum pattern matching."
            ]);
            if (window.runGroverAnimation) {
                window.runGroverAnimation();
            }
            triggerAnalyze();
            return;
        }

        const quantumCommands = [
            "analyze quantum",
            "run quantum analysis",
            "start quantum search",
            "run grover",
            "run search"
        ];

        if (quantumCommands.some(function (cmd) { return transcript.includes(cmd); })) {
            sayRandom([
                "Starting quantum analysis.",
                "Running Grover search simulation.",
                "Launching quantum pattern matching.",
                "Activating search algorithm.",
                "Initiating quantum search visualization."
            ]);
            triggerAnalyze();
        } else if (transcript.includes("scroll down")) {
            window.scrollBy({ top: window.innerHeight, behavior: "smooth" });
            sayRandom(["Scrolling down.", "Moving down the page.", "Page scrolled."]);
        } else if (transcript.includes("scroll up")) {
            window.scrollBy({ top: -window.innerHeight, behavior: "smooth" });
            sayRandom(["Scrolling up.", "Moving up the page.", "Page scrolled up."]);
        } else if (transcript.includes("project")) {
            speak("This project is a quantum accelerated pattern matching for computer vision using Grovers algorithm. Upload a scene and target image, then run analysis to find the best match.");
        } else {
            sayRandom([
                "I did not understand the command.",
                "Please repeat your instruction.",
                "Command not recognized.",
                "Try saying run search or explain project."
            ]);
        }
    }

    recognition.onresult = function (event) {
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const result = event.results[i];
            if (!result.isFinal) continue;
            const transcript = String(result[0].transcript || "").toLowerCase().trim();
            if (!transcript) continue;
            console.log("Heard:", transcript);

            if (!listeningForCommand && transcript.includes("jarvis")) {
                const wakeResponses = [
                    "Jarvis online.",
                    "I am listening.",
                    "Yes, what can I do?",
                    "Voice interface active.",
                    "Quantum system ready.",
                    "Awaiting your command."
                ];
                sayRandom(wakeResponses);
                listeningForCommand = true;
                wakeUntil = Date.now() + 7000;
                continue;
            }

            if (listeningForCommand && Date.now() <= wakeUntil) {
                handleCommand(transcript);
                listeningForCommand = false;
                wakeUntil = 0;
            }
        }
    };

    recognition.onend = function () {
        isRecognizing = false;
        if (!shouldAutoRestart) {
            return;
        }
        window.setTimeout(function () {
            if (!shouldAutoRestart || isRecognizing || isStarting) {
                return;
            }
            try {
                recognition.start();
            } catch (err) {
                console.log("Jarvis restart skipped:", err);
            }
        }, 250);
    };
})();
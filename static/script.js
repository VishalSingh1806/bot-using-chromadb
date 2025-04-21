const isLocal = window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost";
const BACKEND_CHAT_URL = isLocal ? "http://127.0.0.1:8000/query" : "http://34.173.78.39:8000/query";
const BACKEND_FORM_URL = isLocal ? "http://127.0.0.1:8000/collect_user_data" : "http://34.173.78.39:8000/collect_user_data";

let isFormSubmitted = false; // Default state: form not submitted
let inactivityTimer;
let lastPingTime = 0;
let hasSentSoftPing = false;


const suggestedQuestions = [
    "What is EPR registration?",
    "How can I apply for a plastic waste certificate?",
    "What are the responsibilities of a brand owner?",
    "Do I need to submit monthly reports?",
    "How is EPR compliance verified?",
    "What documents are needed for CPCB registration?",
    "Who qualifies as a PIBO?",
    "How does ReCircle help with plastic credit?",
    "What is the penalty for non-compliance?",
    "Can you help me with recycling partners?"
];


// Initialize when DOM is loaded
document.addEventListener("DOMContentLoaded", function() {
    console.log("🚀 Initializing chat...");
    
    // Generate session ID if not exists
    if (!localStorage.getItem("session_id")) {
        const sessionId = generateSessionId();
        console.log("Generated new session ID:", sessionId);
        localStorage.setItem("session_id", sessionId);
    }
    // Load stored chat history Chat history from localstorage
    loadChatHistory();

    // Get chat elements
    const chatWindow = document.getElementById("chatWindow");
    const chatContent = document.getElementById("chatContent");
    
    if (!chatWindow || !chatContent) {
        console.error("❌ Chat elements not found!");
        return;
    }

    console.log("✅ Chat elements found");

    ["mousemove", "keydown", "click", "scroll"].forEach(event => {
        document.addEventListener(event, resetInactivityTimer);
    });
    resetInactivityTimer(); // start it once
    

    const sendBtn = document.getElementById("sendMessage");
    if (sendBtn) {
        // Disable send button until form is submitted
        const formSubmitted = localStorage.getItem("formSubmitted") === "true";
        sendBtn.disabled = !formSubmitted;
    }

    // Only trigger the form if no chat history exists
    const storedChatHistory = JSON.parse(localStorage.getItem(`chatHistory_${localStorage.getItem("session_id")}`) || "[]");
    if (storedChatHistory.length === 0) {
        console.log("Chat content empty, triggering form check...");
        triggerBackendForForm();
    } else{
        console.log("✅ Chat history found, skipping form check.")
    }
});


// Trigger Backend for Form
async function triggerBackendForForm() {
    console.log("🚀 Triggering Backend for Form...");
    let session_id = localStorage.getItem("session_id") || ""; // ✅ Ensure session_id is not null

    if (!session_id) {
        console.log("No session found. Displaying form....");
        displayForm();
        return;
    }

    const chatContent = document.getElementById("chatContent");
    if (!chatContent) {
        console.error("❌ Chat content container not found!");
        return;
    }

    // ✅ Show loading message
    const loadingMessage = document.createElement("div");
    loadingMessage.className = "bot-message fade-in";
    loadingMessage.innerText = "Bot is loading...";
    chatContent.appendChild(loadingMessage);
    chatContent.scrollTop = chatContent.scrollHeight;

    try {
        const response = await fetch(BACKEND_CHAT_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: "Hello",
                session_id: session_id,
            }),
        });

        // ✅ Remove loading message safely
        if (chatContent.contains(loadingMessage)) {
            chatContent.removeChild(loadingMessage);
        }

        if (!response.ok) {
            console.error(`❌ Server error: ${response.status} - ${response.statusText}`);
            addMessageToChat(`Server error: ${response.status}. Please try again.`, "bot-message");
            return;
        }

        const data = await response.json();
        console.log("✅ API Response:", data);

        if (data.redirect_to === "/collect_user_data") {
            console.log("✅ Backend requested form display.");
            displayForm();
        } else if (data.message) {
            addMessageToChat(data.message, "bot-message");
        }
    } catch (error) {
        console.error("❌ Fetch error:", error.message || error);
        
        // ✅ Remove loading message safely
        if (chatContent.contains(loadingMessage)) {
            chatContent.removeChild(loadingMessage);
        }

        addMessageToChat("Network error. Please check your connection.", "bot-message");
    }
}

function displayForm() {
    console.log("📝 Displaying form...");
    const chatContent = document.getElementById("chatContent");
    
    if (!chatContent) {
        console.error("❌ Chat content element not found!");
        return;
    }

    // ✅ Check if the form already exists to prevent duplication
    if (document.getElementById("userForm")) {
        console.warn("⚠️ Form already displayed. Skipping re-render.");
        return;
    }

    const formHtml = `
        <div class="bot-message fade-in form-center">
            <div class="form-container">
             <h3 id="formHeading">Let's get to know you!</h3>
                <form id="userForm">
                    <div class="form-group">
                        <label for="name">Name</label>
                        <input type="text" id="name" placeholder="Enter your full name" required>
                        <div class="error-message" id="nameError"></div>
                    </div>
                    <div class="form-group">
                        <label for="email">Email</label>
                        <input type="email" id="email" placeholder="Enter your email address" required>
                        <div class="error-message" id="emailError"></div>
                    </div>
                    <div class="form-group">
                        <label for="phone">Phone</label>
                        <input type="text" id="phone" placeholder="Enter your phone number" required>
                        <div class="error-message" id="phoneError"></div>
                    </div>
                    <div class="form-group">
                        <label for="organization">Organization</label>
                        <input type="text" id="organization" placeholder="Enter your organization name" required>
                        <div class="error-message" id="organizationError"></div>
                    </div>
                    <div class="form-actions">
                        <button type="button" class="submit-button" onclick="submitForm()" disabled>Submit</button>
                    </div>
                </form>
            </div>
        </div>`;

    // ✅ Append form safely without removing existing event listeners
    chatContent.insertAdjacentHTML("beforeend", formHtml);
    chatContent.scrollTop = chatContent.scrollHeight;

    // ✅ Get form inputs safely
    const nameInput = document.getElementById("name");
    const emailInput = document.getElementById("email");
    const phoneInput = document.getElementById("phone");
    const orgInput = document.getElementById("organization");

    if (nameInput && emailInput && phoneInput && orgInput) {
        // ✅ Attach event listeners safely
        nameInput.addEventListener("blur", validateName);
        emailInput.addEventListener("blur", validateEmail);
        phoneInput.addEventListener("blur", validatePhone);
        orgInput.addEventListener("blur", validateOrganization);

        nameInput.addEventListener("input", toggleSubmitButton);
        emailInput.addEventListener("input", toggleSubmitButton);
        phoneInput.addEventListener("input", toggleSubmitButton);
        orgInput.addEventListener("input", toggleSubmitButton);
    } else {
        console.error("❌ Form elements not found!");
    }

    console.log("✅ Form displayed successfully");
}


// Validation Helper Functions
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/; // Basic email format
    return emailRegex.test(email.trim());
}

function isValidPhone(phone) {
    phone = phone.replace(/\D/g, ""); // Remove all non-numeric characters
    return phone.length === 10 || (phone.length === 12 && phone.startsWith("91")); // Allow "919876543210" or "9876543210"
}

function isValidName(name) {
    const nameRegex = /^[a-zA-Z]+(?:\s[a-zA-Z]+)*$/; // Allow spaces between words but not multiple consecutive spaces
    name = name.trim().replace(/\s+/g, " "); // Normalize spaces
    return name.length > 0 && nameRegex.test(name);
}

function isValidOrganization(organization) {
    return organization.trim().length > 0 && organization.length <= 100; // Max 100 characters
}


// Validate Name
function validateName() {
    const nameInput = document.getElementById("name");
    const nameError = document.getElementById("nameError");

    if (!nameInput || !nameError) {
        console.error("❌ Name input or error message element not found!");
        return;
    }

    const name = nameInput.value.trim();
    if (!isValidName(name)) {
        nameError.textContent = "Please enter a valid name (letters and spaces only).";
        nameError.style.display = "block";
    } else {
        nameError.style.display = "none";
    }
}

// Validate Email
function validateEmail() {
    const emailInput = document.getElementById("email");
    const emailError = document.getElementById("emailError");

    if (!emailInput || !emailError) {
        console.error("❌ Email input or error message element not found!");
        return;
    }

    const email = emailInput.value.trim();
    if (!isValidEmail(email)) {
        emailError.textContent = "Oops! Your email address looks incomplete. Please check again.";
        emailError.style.display = "block";
    } else {
        emailError.style.display = "none";
    }
}

// Validate Phone
function validatePhone() {
    const phoneInput = document.getElementById("phone");
    const phoneError = document.getElementById("phoneError");

    if (!phoneInput || !phoneError) {
        console.error("❌ Phone input or error message element not found!");
        return;
    }

    const phone = phoneInput.value.trim();
    if (!isValidPhone(phone)) {
        phoneError.textContent = "Please enter a valid 10-digit phone number.";
        phoneError.style.display = "block";
    } else {
        phoneError.style.display = "none";
    }
}

function validateOrganization() {
    const organization = document.getElementById("organization").value.trim();
    const organizationError = document.getElementById("organizationError");
    if (!isValidOrganization(organization)) {
        organizationError.textContent = "Please enter a valid organization name (max 100 characters).";
        organizationError.style.display = "block";
    } else {
        organizationError.style.display = "none";
    }
}


// Toggle Submit Button
function toggleSubmitButton() {
    const submitButton = document.querySelector(".submit-button");

    // Check if the button exists
    if (!submitButton) {
        console.error("❌ Submit button not found!");
        return;
    }

    const nameInput = document.getElementById("name");
    const emailInput = document.getElementById("email");
    const phoneInput = document.getElementById("phone");
    const orgInput = document.getElementById("organization");

    // Check if all elements exist before accessing `.value.trim()`
    if (!nameInput || !emailInput || !phoneInput || !orgInput) {
        console.error("❌ One or more form elements not found!");
        return;
    }

    const isFormValid =
        isValidName(nameInput.value.trim()) &&
        isValidEmail(emailInput.value.trim()) &&
        isValidPhone(phoneInput.value.trim()) &&
        isValidOrganization(orgInput.value.trim());

    submitButton.disabled = !isFormValid;
}


function sanitizeInput(input) {
    return input.replace(/<\/?[^>]+(>|$)/g, ""); // Removes HTML tags
}

function showInitialSuggestedQuestions() {
    const shuffled = suggestedQuestions.sort(() => 0.5 - Math.random());
    const topQuestions = shuffled.slice(0, 3); // Show 3 at a time

    displaySimilarQuestions(topQuestions);
}


// Submit Form Data
async function submitForm() {
    const nameInput = document.getElementById("name");
    const emailInput = document.getElementById("email");
    const phoneInput = document.getElementById("phone");
    const orgInput = document.getElementById("organization");

    if (!nameInput || !emailInput || !phoneInput || !orgInput) {
        console.error("❌ One or more form elements not found!");
        addMessageToChat("Form error: missing fields.", "bot-message");
        return;
    }

    const name = nameInput.value.trim();
    const email = emailInput.value.trim();
    const phone = phoneInput.value.trim();
    const organization = orgInput.value.trim();

    try {
        const response = await fetch(BACKEND_FORM_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: localStorage.getItem("session_id") || null,
                name,
                email,
                phone,
                organization,
            }),
        });

        if (response.ok) {
            // ✅ Remove the form after successful submission
            const userForm = document.getElementById("userForm");
            if (userForm) userForm.remove();

            // ✅ Add a centered success message
            const formWrapper = document.querySelector(".bot-message.form-center");
            if (formWrapper) formWrapper.remove();
            
            const userName = nameInput.value.trim().split(" ")[0]; // First name only

            const successHtml = `
                <div class="bot-message-wrapper fade-in">
                    <div class="message-bubble bot-message">
                        <img class="avatar bot-avatar" src="/static/bot-chat-img.svg" alt="Bot">
                        <div class="message-text">
                            🌟 Great to meet you, <b>${userName}</b>!<br>
                            What would you like to know today.
                        </div>
                    </div>
                </div>
            `;

            const chatContent = document.getElementById("chatContent");
            chatContent.insertAdjacentHTML("beforeend", successHtml);
            chatContent.scrollTop = chatContent.scrollHeight;

            localStorage.setItem("formSubmitted", "true");

            // ✅ Show 3 shuffled suggested questions
            showInitialSuggestedQuestions();

            const sendBtn = document.getElementById("sendMessage");
            if (sendBtn) sendBtn.disabled = false;

            isFormSubmitted = true;
        } else {
            addMessageToChat("Error submitting your details. Please try again.", "bot-message");
        }
    } catch (error) {
        addMessageToChat("Network error while submitting your details.", "bot-message");
        console.error("Fetch error:", error);
    }
}


// Toggle Chat Window
function toggleChat() {
    const chatWindow = document.getElementById("chatWindow");
    const chatButton = document.querySelector(".chat-button")
    if (!chatWindow || !chatButton) {
        console.error("❌ Chat elements not found!");
        return;
    }

    const isVisible = chatWindow.style.display === "block";
    
    if (isVisible) {
        chatWindow.style.display = "none";
        chatWindow.classList.remove('show');
        chatWindow.classList.add('hidden');
        chatButton.style.display = "block";
        chatButton.style.display = "flex";
        chatButton.style.bottom = "20px";
    } else {
        chatWindow.style.display = "block";
        chatWindow.classList.remove('hidden');
        chatWindow.classList.add('show');
        chatButton.style.display = "none";
        
        const chatContent = document.getElementById("chatContent");
        if (chatContent && !chatContent.children.length) {
            triggerBackendForForm();
        }
    }
}

// ✅ Handle "Enter" Key Press to Send Message
function checkEnter(event) {
    const userMessageInput = document.getElementById("userMessage");
    
    if (event.key === "Enter" && event.target === userMessageInput) {
        event.preventDefault(); // Prevent unintended Enter key behavior
        sendMessage();
    }
}



function addMessageToChat(message, className) {
    const chatContent = document.getElementById("chatContent");
    if (!chatContent) {
        console.error("❌ Chat content container not found!");
        return;
    }

    if (className.includes("user")) {
        const wrapper = document.createElement("div");
        wrapper.className = "message-wrapper user-wrapper fade-in";

        const bubble = document.createElement("div");
        bubble.className = "message-bubble user-message";
        bubble.innerHTML = `
            <div class="message-text">${message}</div>
            <img class="avatar user-avatar" src="/static/user-img.svg" alt="User">
        `;

        wrapper.appendChild(bubble);
        chatContent.appendChild(wrapper);
    } else {
        const wrapper = document.createElement("div");
        wrapper.className = "message-wrapper bot-wrapper fade-in";

        const bubble = document.createElement("div");
        bubble.className = "message-bubble bot-message";
        bubble.innerHTML = `
            <img class="avatar bot-avatar" src="/static/bot-chat-img.svg" alt="Bot">
            <div class="message-text">${message}</div>
        `;

        wrapper.appendChild(bubble);
        chatContent.appendChild(wrapper);
    }

    chatContent.scrollTop = chatContent.scrollHeight;

    const sessionId = localStorage.getItem("session_id");
    if (sessionId) {
        const chatHistoryKey = `chatHistory_${sessionId}`;
        const chatHistory = JSON.parse(localStorage.getItem(chatHistoryKey) || "[]");

        chatHistory.push({
            message: message,
            type: className.includes("user") ? "user" : "bot",
            timestamp: new Date().toISOString(),
        });

        localStorage.setItem(chatHistoryKey, JSON.stringify(chatHistory));
    }
}

function resetInactivityTimer() {
    clearTimeout(inactivityTimer);
    inactivityTimer = setTimeout(triggerSoftPing, 60000); // 1 min
}

function triggerSoftPing() {
    const now = Date.now();
    if (hasSentSoftPing) return;
    if (now - lastPingTime < 1 * 60 * 1000) return; // 5 min cooldown
    if (!localStorage.getItem("formSubmitted")) return;

    const input = document.getElementById("userMessage");
    if (!input || input.disabled) return;

    const softPings = [
        "Still there? 😊 I’m here to help.",
        "Have more questions about EPR?",
        "Need help with certificates or compliance?",
        "👋 Just checking in. Ask away anytime!",
    ];
    const randomPing = softPings[Math.floor(Math.random() * softPings.length)];

    addMessageToChat(randomPing, "bot-message");
    lastPingTime = now;
    hasSentSoftPing = true; 
}

function saveChatMessage(sessionId, messageData) {
    try {
        // Get existing chat history
        const chatHistoryKey = `chatHistory_${sessionId}`;
        const chatHistory = JSON.parse(localStorage.getItem(chatHistoryKey) || '[]');

        // Add new message
        chatHistory.push({
            message: messageData.message,
            type: messageData.type,
            timestamp: new Date().toISOString()
        });

        // Save updated history
        localStorage.setItem(chatHistoryKey, JSON.stringify(chatHistory));
        console.log("✅ Chat message saved successfully");
    } catch (error) {
        console.error("❌ Error saving chat message:", error);
    }
}

// ✅ Show error screen and hide chat
function showErrorScreen() {
    const chatBody = document.querySelector('.chat-body');
    const chatFooter = document.querySelector('.chat-footer');
    const errorScreen = document.querySelector('.error-screen');
    const chatWindow = document.getElementById("chatWindow");

    if (!chatBody || !chatFooter || !errorScreen || !chatWindow) {
        console.error("❌ Error: Missing chat elements!");
        return;
    }

    // ✅ Hide chat content, keep header visible
    chatBody.classList.add('hidden');
    chatFooter.classList.add('hidden');

    // ✅ Show error screen
    errorScreen.classList.add('show');
    errorScreen.style.display = "flex";

    // ✅ Ensure chat window height is correct
    chatWindow.style.height = "400px"; // 🔹 Make sure it's large enough
}


// ✅ Hide error screen and restore chat content
function hideErrorScreen() {
    const chatBody = document.querySelector('.chat-body');
    const chatFooter = document.querySelector('.chat-footer');
    const errorScreen = document.querySelector('.error-screen');

    if (!chatBody || !chatFooter || !errorScreen) {
        console.error("❌ Error: Missing chat elements!");
        return;
    }

    // ✅ Restore chat content
    chatBody.classList.remove('hidden');
    chatFooter.classList.remove('hidden');

    // ✅ Hide error screen
    errorScreen.classList.remove('show');
    errorScreen.style.display = "none";
}

async function sendMessage(userQuery = null, isSuggested = false) {
    try {
        const userMessageInput = document.getElementById("userMessage");
        const chatContent = document.getElementById("chatContent");
        const chatFooter = document.querySelector(".chat-footer");
        const errorScreen = document.querySelector(".error-screen");

        if (!chatContent || !errorScreen || !chatFooter) {
            console.error("❌ Error: Missing chat elements!");
            return;
        }

        let userMessage = userQuery || userMessageInput?.value.trim();
        if (!userMessage) return; // Prevent empty messages

        const sessionId = localStorage.getItem("session_id") || generateSessionId();

        // ✅ Append user message using addMessageToChat
        addMessageToChat(userMessage, "user-message");
        hasSentSoftPing = false; // ✅ reset when user sends message

        if (userMessageInput) userMessageInput.value = ""; // Clear input field

        try {
            const response = await fetch(BACKEND_CHAT_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId, text: userMessage, n_results: 5 })
            });

            if (!response.ok) throw new Error(`Server Error: ${response.status}`);

            const data = await response.json();
            console.log("✅ API Response:", data);

            if (!data.results?.[0]?.answer) throw new Error("No valid response received.");

            // ✅ Append bot message with avatar
            addMessageToChat(data.results[0].answer, "bot-message");

            // ✅ Display similar questions if available
            if (data.similar_questions?.length > 0) displaySimilarQuestions(data.similar_questions);

            // ✅ Hide error screen if chat works fine
            hideErrorScreen();
        } catch (error) {
            console.error("❌ Fetch Error:", error);
            showErrorScreen();
        }
    } catch (error) {
        console.error("❌ Unexpected Error:", error);
        showErrorScreen();
    }
}



function saveChatHistory(message, type) {
    try {
        const sessionId = localStorage.getItem("session_id");
        if (!sessionId) return;

        const chatHistory = JSON.parse(localStorage.getItem(`chatHistory_${sessionId}`)) || [];
        chatHistory.push({
            message,
            type,
            timestamp: new Date().toISOString()
        });

        localStorage.setItem(`chatHistory_${sessionId}`, JSON.stringify(chatHistory));
    } catch (error) {
        console.error("❌ Error saving chat history:", error);
    }
}

function loadChatHistory() {
    try {
        const sessionId = localStorage.getItem("session_id");
        if (!sessionId) return;

        const chatContent = document.getElementById("chatContent");
        if (!chatContent) {
            console.error("❌ Chat content container not found!");
            return;
        }

        const chatHistory = JSON.parse(localStorage.getItem(`chatHistory_${sessionId}`)) || [];
        chatContent.innerHTML = '';

        chatHistory.forEach(item => {
            const messageType = item.type === 'user' ? 'user-message' : 'bot-message';
            addMessageToChat(item.message, messageType);
        });

        // Ensure scroll to bottom after loading history
        chatContent.scrollTop = chatContent.scrollHeight;
    } catch (error) {
        console.error("❌ Error loading chat history:", error);
    }
}



// ✅ Function to Display Similar Questions
function displaySimilarQuestions(similarQuestions) {
    console.log("✅ Displaying Similar Questions:", similarQuestions);

    const chatContent = document.getElementById("chatContent");
    if (!chatContent) {
        console.error("❌ Chat content container not found!");
        return;
    }

    // ✅ Remove previous similar questions if they exist
    let existingSimilarDiv = document.querySelector(".similar-questions");
    if (existingSimilarDiv) {
        existingSimilarDiv.remove();
    }

    // ✅ If no similar questions, show a placeholder
    if (!similarQuestions || similarQuestions.length === 0) {
        console.log("⚠ No similar questions found.");
        let placeholder = document.createElement("div");
        placeholder.className = "similar-questions-placeholder";
        placeholder.textContent = "No related questions available.";
        chatContent.appendChild(placeholder);
        return;
    }

    // ✅ Create a new similar questions container
    const similarDiv = document.createElement("div");
    similarDiv.className = "similar-questions";
    similarDiv.innerHTML = `<strong>Would you also like to know about?:</strong>`;

    similarQuestions.forEach(q => {
        if (!q) return; // ✅ Prevent empty or invalid strings
    
        let button = document.createElement("button");
        button.textContent = q.trim();
        button.className = "similar-question-item";
        button.setAttribute("data-question", q.trim());
    
        similarDiv.appendChild(button);
    });
    

    chatContent.appendChild(similarDiv);
}

// ✅ Event Delegation (No need for `onclick` inside `displaySimilarQuestions()`)
document.addEventListener("click", function(event) {
    if (event.target.classList.contains("similar-question-item")) {
        let clickedQuestion = event.target.getAttribute("data-question");
        if (clickedQuestion) {
            console.log("Clicked similar question:", clickedQuestion);
            sendMessage(clickedQuestion, true);
        }
    }
});



function generateSessionId() {
    let sessionId = localStorage.getItem("session_id");
    if (!sessionId) {
        sessionId = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
        localStorage.setItem("session_id", sessionId);
    }
    return sessionId;
}


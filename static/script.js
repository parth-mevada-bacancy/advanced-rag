const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const resetBtn = document.getElementById("reset-btn");

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Small, dependency-free markdown -> HTML for AI replies (bold, italic, inline
// code, bullet/numbered lists, paragraphs). Escapes HTML first so nothing the
// model outputs can inject markup.
function formatMarkdown(raw) {
  const escaped = escapeHtml(raw);
  const lines = escaped.split("\n");

  let html = "";
  let listType = null; // "ul" | "ol" | null
  let paragraph = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      html += `<p>${paragraph.join("<br>")}</p>`;
      paragraph = [];
    }
  };
  const closeList = () => {
    if (listType) {
      html += `</${listType}>`;
      listType = null;
    }
  };
  const inline = (text) =>
    text
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>");

  for (const line of lines) {
    const bullet = line.match(/^\s*[-*]\s+(.*)/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)/);

    if (bullet) {
      flushParagraph();
      if (listType !== "ul") { closeList(); html += "<ul>"; listType = "ul"; }
      html += `<li>${inline(bullet[1])}</li>`;
    } else if (numbered) {
      flushParagraph();
      if (listType !== "ol") { closeList(); html += "<ol>"; listType = "ol"; }
      html += `<li>${inline(numbered[1])}</li>`;
    } else if (line.trim() === "") {
      closeList();
      flushParagraph();
    } else {
      closeList();
      paragraph.push(inline(line));
    }
  }
  closeList();
  flushParagraph();
  return html;
}

function addMessage(role, text) {
  const isAi = role === "ai";

  const row = document.createElement("div");
  row.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${isAi ? "ai-avatar" : "user-avatar"}`;
  avatar.textContent = isAi ? "W" : "U";

  const content = document.createElement("div");
  content.className = "msg-content";

  const sender = document.createElement("div");
  sender.className = "sender";
  sender.textContent = isAi ? "Assistant" : "User";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (isAi) {
    bubble.innerHTML = formatMarkdown(text);
  } else {
    bubble.textContent = text;
  }

  content.appendChild(sender);
  content.appendChild(bubble);
  row.appendChild(avatar);
  row.appendChild(content);
  chatWindow.appendChild(row);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return bubble;
}

function setBubbleText(bubble, text) {
  bubble.innerHTML = formatMarkdown(text);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendMessage(message) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  const data = await response.json();
  return data.reply;
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  addMessage("user", text);
  chatInput.value = "";
  chatInput.disabled = true;
  const submitBtn = chatForm.querySelector("button[type='submit']");
  submitBtn.disabled = true;

  const pendingBubble = addMessage("ai", "Thinking...");
  pendingBubble.classList.add("pending");

  try {
    const reply = await sendMessage(text);
    setBubbleText(pendingBubble, reply);
    pendingBubble.classList.remove("pending");
  } catch (err) {
    setBubbleText(pendingBubble, "Something went wrong reaching the server. Please try again.");
    pendingBubble.classList.remove("pending");
  } finally {
    chatInput.disabled = false;
    submitBtn.disabled = false;
    chatInput.focus();
  }
});

resetBtn.addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  chatWindow.innerHTML = "";
  addMessage("ai", "Conversation cleared. Ask me anything about Global Economic Prospects, June 2026.");
});

chatInput.focus();

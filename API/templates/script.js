/* ---------- Global state ---------- */
let imageUrl = null,
    imagePath = null;

/* ---------- Image upload ---------- */
document.getElementById('imageInput').addEventListener('change', async function () {
  const file = this.files[0];
  if (!file) return;

  const fd = new FormData();
  fd.append('file', file);

  const res = await fetch('/upload', { method: 'POST', body: fd });
  const data = await res.json();

  if (data.error) { alert(data.error); return; }

  imageUrl  = data.url;
  imagePath = data.path;

  document.getElementById('preview').innerHTML =
    `<p>Image preview:</p><img src="${imageUrl}" alt="preview">`;
});

/* ---------- Send message ---------- */
document.getElementById('sendBtn').addEventListener('click', sendMessage);

function appendMessage(sender, html, isRaw = false) {
  const box  = document.getElementById('chatBox');
  const wrap = document.createElement('div');
  wrap.className = `message ${sender}`;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = isRaw ? html : marked.parse(html);

  wrap.appendChild(bubble);
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
  return bubble;   // return for later streaming update
}

async function sendMessage() {
  const input = document.getElementById('userInput');
  const txt   = input.value.trim();
  if (!txt && !imagePath) return;

  // show user message
  if (txt) appendMessage('user', txt);
  if (imageUrl) appendMessage('user', `<img src="${imageUrl}" alt="img">`, true);

  // reset composer
  input.value = '';
  document.getElementById('preview').innerHTML = '';

  // create empty robot bubble
  const robotBubble = appendMessage('robot', '');

  // SSE connection
  const es = new EventSourcePolyfill('/chat', {
    headers: { 'Content-Type': 'application/json' },
    payload: JSON.stringify({ message: txt, path: imagePath })
  });

  let buf = '';
  es.onmessage = e => {
    buf += e.data;
    robotBubble.innerHTML = marked.parse(buf);
  };
  es.onerror = () => es.close();

  imageUrl = imagePath = null;
}

/* ---------- SSE Polyfill ---------- */
class EventSourcePolyfill {
  constructor(url, opt = {}) {
    this.url     = url;
    this.payload = opt.payload;
    this.headers = opt.headers || {};
    this.ctrl    = new AbortController();
    this.onmessage = this.onerror = () => {};
    this._open();
  }
  async _open() {
    try {
      const res = await fetch(this.url, {
        method: 'POST',
        headers: this.headers,
        body: this.payload,
        signal: this.ctrl.signal
      });
      const rd = res.body.getReader();
      const td = new TextDecoder();
      let buf  = '';

      while (true) {
        const { value, done } = await rd.read();
        if (done) break;
        buf += td.decode(value, { stream: true });
        let parts = buf.split('\n\n');
        buf = parts.pop();
        parts.forEach(line => {
          if (line.startsWith('data: ')) this.onmessage({ data: line.slice(6) });
        });
      }
    } catch (e) { this.onerror(e); }
  }
  close() { this.ctrl.abort(); }
}

# app.py
import os
import uuid
from flask import (
    Flask, render_template, request, jsonify,
    Response, send_from_directory, url_for
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from API.llm_chat_api import get_client, encode_image_to_base64

load_dotenv()

app = Flask(__name__)

# 将 uploads 目录固定在项目根路径下，避免启动路径不同导致找不到文件
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 简易内存对话历史；生产场景建议持久化或做长度裁剪
chat_history = []


@app.route('/')
def index():
    return render_template('index.html')


# ---------- 图片上传 ----------
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    abs_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(abs_path)

    # url_for 保证不同部署路径也能正确访问
    return jsonify({
        'url': url_for('uploaded_file', filename=filename),  # 浏览器可访问的 URL
        'path': abs_path                                     # 服务器本地绝对路径
    })


# ---------- 聊天接口（SSE 流式） ----------
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    user_message   = data.get('message', '')
    image_abs_path = data.get('path')      # 前端只传绝对路径
    platform       = data.get('platform', 'aliyun')
    model          = data.get('model', 'qwen2.5-vl-32b-instruct')

    client = get_client(platform)

    # 拼装多模态 / 纯文本内容
    if image_abs_path:
        base64_image = encode_image_to_base64(image_abs_path)
        content = [
            {"type": "image_url", "image_url": {"url": base64_image}},
            {"type": "text",       "text": user_message}
        ]
    else:
        content = user_message

    chat_history.append({"role": "user", "content": content})

    def generate():
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system",
                           "content": "你是一个多模态聊天机器人，能够理解图像并进行对话。"}
                          ] + chat_history,
                stream=True
            )

            full_reply = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                text  = getattr(delta, "content", None)
                if text:
                    full_reply += text
                    yield f"data: {text}\n\n"

            chat_history.append({"role": "assistant", "content": full_reply})

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    # Server-Sent Events
    return Response(generate(), mimetype='text/event-stream')


@app.route('/assets/<path:filename>')
def assets(filename):
    # 让浏览器可以 GET /assets/style.css 或 script.js
    return send_from_directory(app.template_folder, filename)


# ---------- 访问上传图片 ----------
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, threaded=True)

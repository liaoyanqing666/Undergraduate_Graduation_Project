import base64
import io
from re import T
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def get_client(platform):
    """
    Function to get the client object for the specified Chinese platform.
    Notice that VPN should be closed when using the Chinese platforms.
    If you want to use the other platforms (like Openai), just change the base_url and api_key.

    :param platform: the platform name
    :return: a client of the specified platform
    """
    test_model = None
    client = None

    # 百度
    # 百度支持的模型列表： https://cloud.baidu.com/doc/qianfan-docs/s/7m95lyy43
    if platform in ('baidu', 'baiduyun', 'qianfan', '百度'):
        client = OpenAI(
            base_url='https://qianfan.baidubce.com/v2/',
            api_key=os.getenv('BAIDU_API')
        )

    # 阿里
    # 阿里支持的模型列表： https://help.aliyun.com/zh/model-studio/getting-started/models?spm=a2c4g.11186623.0.0.76ce2bdbdTZ5xk#850732b1aabs0
    elif platform in ('ali', 'alibaba', 'aliyun', 'bailian', '阿里'):
        client = OpenAI(
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1/',
            api_key=os.getenv('ALIYUN_API')
        )

    # DeepSeek
    # DeepSeek支持的模型列表： https://api-docs.deepseek.com/zh-cn/quick_start/pricing
    elif platform in ('deepseek', ):
        client = OpenAI(
            base_url='https://api.deepseek.com/v1',
            api_key=os.getenv('DEEPSEEK_API')
        )

    # 本地ollama（需要先启动ollama）
    # 本地ollama支持的模型列表：命令行输入ollama list
    elif platform in ('local', '本地', 'ollama'):
        client = OpenAI(
            base_url='http://localhost:11434/v1/',
            api_key='None',
        )

    return client


def address_chat_response(
    client,
    model,
    messages,
    stream=True,
    text_only=True
):
    """
    Output the responses of the chat model in the terminal.

    :param client: client object
    :param model: model name
    :param messages: json of messages
    :param stream: whether to stream the response
    :param text_only: whether to output content only or the response object
    """

    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream
    )
    if stream:
        reseaning_func = None # Only used to output the title
        for crunch in chat_completion:
            if reseaning_func is None and hasattr(crunch.choices[0].delta, "reasoning_content"):
                reseaning_func = True
                print('\033[33mReasoning Content:\033[0m')
            if reseaning_func and crunch.choices[0].delta.content is not None:
                reseaning_func = False
                print('\033[32m\nReply Content:\033[0m')

            if text_only:
                print(crunch.choices[0].delta.content if crunch.choices[0].delta.content is not None else crunch.choices[0].delta.reasoning_content, end='')
            else:
                print(crunch)
        print()

    else:
        if hasattr(chat_completion.choices[0].message, "reasoning_content"):
            reasoning_content = chat_completion.choices[0].message.reasoning_content
            print('\033[33mReasoning Content:\033[0m', reasoning_content, '\033[32mReply Content:\033[0m', chat_completion.choices[0].message.content, sep='\n') \
                if text_only else print(chat_completion)
        else:
            print(chat_completion.choices[0].message.content) if text_only else print(chat_completion)


def test_llm_api():
    """
    Test the LLM API with different platforms and models.

    Suggested Single-modal model (reasoning and non-reasoning):
        baidu:      'qwq-32b'/'ernie-4.0-8k'
        ali:        'qwq-plus'/'qwen-turbo-2024-11-01'
        deepseek:   'deepseek-reasoner'/'deepseek-chat'
        ollama:     'deepseek-r1:1.5b-qwen-distill-fp16'/'qwen2.5:3b'

    """
    messages = [
        {
            "role": "system",
            "content": "你是一个聊天机器人。"
            # Translation: "You are a chatbot."
        },
        {
            "role": "user",
            "content": '你好',
            # Translation: "Hello"
        }
    ]
    client = get_client('aliyun')
    model = 'deepseek-r1'
    address_chat_response(client, model, messages, stream=True, text_only=True)


def encode_image_to_base64(path, max_size_kb=1024):
    """
    :param path: local image path
    :param max_size_kb: maximum image volume (to ensure that the image is not too big)
    :return: compressed Base64 string
    """
    with Image.open(path) as img:
        img = img.convert("RGB")  # 确保无透明层
        buffer = io.BytesIO()
        quality = 85

        while True:
            buffer.seek(0)
            buffer.truncate()
            img.save(buffer, format="JPEG", quality=quality)
            size_kb = buffer.tell() / 1024
            if size_kb <= max_size_kb or quality <= 20:
                break
            quality -= 5

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"


def test_llm_api_with_image():
    """
    Test the multi-modal LLM API with an image input.

    Suggested Multi-modal model:
        ali: 'qwen2.5-vl-32b-instruct'

    """

    image_path = "test_img.png"
    base64_image = encode_image_to_base64(image_path)

    messages = [
        {
            "role": "system",
            "content": "你是一个多模态聊天机器人，能够理解图像并回答相关问题。",
            # Translation: "You are a multi-modal chatbot that can understand images and answer related questions."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image
                    }
                },
                {
                    "type": "text",
                    "text": "请描述这张图。"
                    # Translation: "Please describe this image."
                }
            ]
        }
    ]
    client = get_client('aliyun')
    model = 'qwen2.5-vl-32b-instruct'
    address_chat_response(client, model, messages, stream=True, text_only=True)


if __name__ == '__main__':

    # When using the Chinese platform, please close the VPN

    print('\033[34mTest LLM API:\033[0m')
    test_llm_api()

    print('\n\n')

    print('\033[34mTest Multimodal LLM API\033[0m')
    test_llm_api_with_image()
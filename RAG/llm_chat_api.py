from re import T
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


def get_client(platform, reasoning=False):
    test_model = None
    client = None

    # 百度
    # 百度支持的模型列表： https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Fm2vrveyu
    if platform in ('baidu', 'baiduyun', 'qianfan', '百度'):
        client = OpenAI(
            base_url='https://qianfan.baidubce.com/v2/',
            api_key=os.getenv('BAIDU_API')
        )
        test_model = 'qwq-32b' if reasoning else 'ernie-4.0-8k'

    # 阿里
    # 阿里支持的模型列表： https://help.aliyun.com/zh/model-studio/getting-started/models?spm=a2c4g.11186623.0.0.76ce2bdbdTZ5xk#850732b1aabs0
    elif platform in ('ali', 'alibaba', 'aliyun', 'bailian', '阿里'):
        client = OpenAI(
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1/',
            api_key=os.getenv('ALIYUN_API')
        )
        test_model = 'qwq-plus' if reasoning else 'qwen-turbo-2024-11-01'

    # DeepSeek
    # DeepSeek支持的模型列表： https://api-docs.deepseek.com/zh-cn/quick_start/pricing
    elif platform in ('deepseek', ):
        client = OpenAI(
            base_url='https://api.deepseek.com/v1',
            api_key=os.getenv('DEEPSEEK_API')
        )
        test_model = 'deepseek-reasoner' if reasoning else 'deepseek-chat'

    # 本地ollama（需要先启动ollama）
    # 本地ollama支持的模型列表：命令行输入ollama list
    elif platform in ('local', '本地', 'ollama'):
        client = OpenAI(
            base_url='http://localhost:11434/v1/',
            api_key='None',
        )
        test_model = 'deepseek-r1:1.5b-qwen-distill-fp16' if reasoning else 'qwen2.5:3b'

    return client, test_model


def get_chat_response(
    platform,
    model=None,
    reasoning=False,
    system=None,
    message='',
    stream=True,
    text_only=True
):
    client, test_model = get_client(platform, reasoning=reasoning)
    if model is None:
        model = test_model
    print('使用模型:', model, '使用平台:', platform)
    chat_completion = client.chat.completions.create(
        model=model,
        messages=([
                      {
                          "role": "system",
                          "content": system,
                      }
                  ]
                  if system is not None else []) + [
                     {
                         "role": "user",
                         "content": message,
                     }
                 ],
        stream=stream
    )
    if stream:
        reseaning_func = None # Only used to output the title
        for crunch in chat_completion:
            if reseaning_func is None and hasattr(crunch.choices[0].delta, "reasoning_content"):
                reseaning_func = True
                print('\033[33m推理内容:\033[0m')
            if reseaning_func and crunch.choices[0].delta.content is not None:
                reseaning_func = False
                print('\033[32m\n回答内容:\033[0m')

            if text_only:
                print(crunch.choices[0].delta.content if crunch.choices[0].delta.content is not None else crunch.choices[0].delta.reasoning_content, end='')
            else:
                print(crunch)

    else:
        if hasattr(chat_completion.choices[0].message, "reasoning_content"):
            reasoning_content = chat_completion.choices[0].message.reasoning_content
            print('\033[33m推理内容:\033[0m', reasoning_content, '\033[32m回答内容:\033[0m', chat_completion.choices[0].message.content, sep='\n') \
                if text_only else print(chat_completion)
        else:
            print(chat_completion.choices[0].message.content) if text_only else print(chat_completion)


if __name__ == '__main__':
    # 需要关闭vpn
    platform = 'aliyun' # 不推荐deepseek（因为要花钱），可以使用百度/阿里的'deepseek-r1'（有免费额度）
    model = None # 模型名，可以为空，为空时使用测试模型
    reasoning = False # model = None时，是否使用推理型测试模型
    test_system = None
    test_message = '你好'
    stream = True
    text_only = True

    get_chat_response(platform, model, reasoning, test_system, test_message, stream, text_only)
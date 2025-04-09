from RAG import RAG
from llm_chat_api import get_chat_response


if __name__ == "__main__":
    platform = 'aliyun'
    model = 'deepseek-r1'
    reasoning = False
    stream = True
    text_only = True
    
    # use RAG
    # system = """
    # 你是负责回答问题的助手。
    # 使用以下你自己通过RAG检索到的内容来回答问题。
    # 如果你不知道答案，就说你不知道。
    # 最多只用三句话，回答要简明扼要。
    # """
    # message = '怎么治疗复发性口腔溃疡'
    
    # rag = RAG()
    # rag.load('RAG\dataset\\vector_database')
    # context = rag.search(message, top_k=3)
    # message = f"问题：{message}\n\n检索到的片段：\n{context}"
    # print(message)

    # not use RAG
    system = """
    你是负责回答问题的中医助手。
    如果你不知道答案，就说你不知道。
    最多只用三句话，回答要简明扼要。
    """
    message = '怎么治疗复发性口腔溃疡'


    get_chat_response(platform, model, reasoning, system, message, stream, text_only)
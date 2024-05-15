from pyrogram import Client, filters
from pyrogram.errors import MessageTooLong
import logging
from openai import OpenAI
import asyncio
import io
import urllib.parse
import traceback
from aiohttp import ClientSession
from pydub import AudioSegment
from dataclasses import dataclass
import signal
import sys
from dotenv import load_dotenv
import os

# 加载 .env 文件中的环境变量
load_dotenv()


# 初始化日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 从环境变量中读取配置项
API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_ENGINE = os.getenv('OPENAI_ENGINE')

# TTS 配置项
TTS_API_PATH = os.getenv('TTS_API_PATH')
TTS_API_LANGUAGE = os.getenv('TTS_API_LANGUAGE')
TTS_C_NAME = os.getenv('TTS_C_NAME')
TTS_API_TOPK = int(os.getenv('TTS_API_TOPK'))
TTS_API_TOPP = float(os.getenv('TTS_API_TOPP'))
TTS_API_TEMPERATURE = float(os.getenv('TTS_API_TEMPERATURE'))

# 初始化 Userbot 客户端
userAccount = Client("my_account", api_id=API_ID, api_hash=API_HASH)

# 初始化 OpenAI 客户端
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

@dataclass
class TTSJob:
    chat_id: int
    text: str
    language: str
    reply_to_message_id: int = None
    command_message_id: int = None

# 创建一个队列（改异步？）
request_queue = asyncio.Queue(maxsize=65535)
shutdown_event = asyncio.Event()

class NamedBytesIO(io.BytesIO):
    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop('name', 'voice.ogg')
        super().__init__(*args, **kwargs)

async def ai_tts_text(chat_id: int, text: str, reply_to_message_id: int = None, command_message_id: int = None):
    if not text:
        logger.error("Text is empty, nothing to process")
        return
    
    # 尝试删除 !v 指令消息
    if command_message_id:
        try:
            await userAccount.delete_messages(chat_id, message_ids=[command_message_id])
            logger.info(f"Command message deleted: message_id={command_message_id}")
        except Exception as e:
            logger.error(f"Failed to delete command message: {e}")
    
    await request_queue.put(TTSJob(chat_id, text, TTS_API_LANGUAGE, reply_to_message_id, command_message_id))
    logger.info(f"Added TTS job to queue: chat_id={chat_id}, text={text}")

async def start_tts_task():
    # 创建异步会话
    async with ClientSession() as session:
        # 当未收到停止事件或请求队列不为空时持续进行
        while not shutdown_event.is_set() or not request_queue.empty():
            try:
                # 尝试从请求队列中获取任务，设置超时为1秒
                job = await asyncio.wait_for(request_queue.get(), timeout=1)
            except asyncio.TimeoutError:
                # 如果等待超时，则继续下一个循环
                continue

            # 获取聊天ID
            chat_id = job.chat_id
            logger.info(f"Processing TTS job: chat_id={chat_id}, text={job.text}")

            # 将环境变量值转换为浮点数，确保API可以正确解析
            try:
                top_k = int(TTS_API_TOPK)  # 确保 top_k 是整数
                top_p = float(TTS_API_TOPP)
                temperature = float(TTS_API_TEMPERATURE)
            except ValueError:
                logger.error("Failed to convert TTS parameters to float")
                continue  # 转换失败时跳过此任务

            # 使用转换后的浮点数构造请求体
            body = {
                "cha_name": TTS_C_NAME,
                "text": urllib.parse.quote(job.text),
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
            }

            # 设置请求头，声明内容类型为JSON
            headers = {"Content-Type": "application/json"}

            try:
                # 发送POST请求到TTS API，附上JSON数据和头信息
                async with session.post(TTS_API_PATH, json=body, headers=headers, timeout=60) as response:
                    if response.status == 200:
                        # 请求成功，读取响应内容
                        content = await response.read()
                        # 将响应内容转换为音频
                        audio = AudioSegment.from_file(io.BytesIO(content), format="wav")
                        buffer = NamedBytesIO(name="voice.ogg")
                        # 导出音频为OGG格式
                        audio.export(buffer, format="ogg", codec="libopus")
                        buffer.seek(0)
                        # 确保客户端连接
                        if not userAccount.is_connected:
                            logger.warning("Client not connected, reconnecting...")
                            await userAccount.connect()

                        # 发送语音消息给用户
                        if job.reply_to_message_id:
                            await userAccount.send_voice(chat_id, buffer, reply_to_message_id=job.reply_to_message_id)
                        else:
                            await userAccount.send_voice(chat_id, buffer)
                        logger.info(f"TTS job completed successfully: chat_id={chat_id}")
                    else:
                        logger.error(f"TTS request failed: status={response.status}")
            except Exception as e:
                logger.error(f"TTS request exception: {e}")
                traceback.print_exc()
            finally:
                # 任务完成，标记队列任务已处理
                request_queue.task_done()

async def ai_translate(chat_id: int, input_text: str, message):
    translation_prompt = {
        "role": "system",
        "content": "你是一个好用的翻译助手。请将我的中文翻译成英文，将所有非中文的翻译成中文。我发给你所有的话都是需要翻译的内容，你只需要回答翻译结果。翻译结果请符合中文的语言习惯。"
    }

    messages = [translation_prompt, {"role": "user", "content": input_text}]

    try:
        response = client.chat.completions.create(
            model=OPENAI_ENGINE,
            messages=messages,
            max_tokens=3000
        )
        output_text = response.choices[0].message.content.strip()
        await message.edit_text(output_text)
        logger.info(f"Translated message edited: {output_text}")
    except Exception as e:
        logger.error(f"Translation request exception: {e}")
        traceback.print_exc()

@userAccount.on_message(filters.text)
async def hello(client, message):
    try:
        # 检查消息的发送者是否为 userbot 账号本身
        if not message.from_user or not message.from_user.is_self:
            return

        if message.text.startswith('!fanyi'):
            # 检查是否有回复的消息
            if message.reply_to_message:
                # 获取指令后的文本内容
                input_text = message.text[len('!fanyi '):].strip()
                if input_text:
                    # 第三种情况：回复消息且指令后有内容
                    logger.info(f"Processing !fanyi command with reply and content: {input_text}")
                    await ai_translate(message.chat.id, input_text, message)
                else:
                    # 第二种情况：回复消息但指令后无内容
                    input_text = message.reply_to_message.text.strip()
                    logger.info(f"Processing !fanyi command with reply: {input_text}")
                    await ai_translate(message.chat.id, input_text, message)
            else:
                # 第一种情况：没有回复消息但指令后有内容
                input_text = message.text[len('!fanyi '):].strip()
                if input_text:
                    logger.info(f"Processing !fanyi command: {input_text}")
                    await ai_translate(message.chat.id, input_text, message)
        elif message.text.startswith('!v'):
            # 获取命令后的文本内容
            input_text = message.text[len('!v '):].strip()

            # 如果没有指定内容且是在回复消息，则使用被回复消息的内容
            if not input_text and message.reply_to_message:
                input_text = message.reply_to_message.text.strip()
            
            if input_text:
                logger.info(f"Processing !v command: {input_text}")
                await ai_tts_text(message.chat.id, input_text, reply_to_message_id=message.reply_to_message.id if message.reply_to_message else None, command_message_id=message.id)
            else:
                logger.error("No text provided for TTS")
        else:
            logger.info(f"Unhandled message: {message.text}")
    except MessageTooLong:
        if message and message.id:
            try:
                await userAccount.delete_messages(message.chat.id, message_ids=[message.id])
            except Exception as delete_error:
                logger.error(f"删除消息失败: {delete_error}")
    except Exception as e:
        if message and message.id:
            try:
                # 处理其他异常，删除对应的消息
                await userAccount.delete_messages(message.chat.id, message_ids=[message.id])
            except Exception as delete_error:
                logger.error(f"删除消息失败: {delete_error}")
        logger.error(f"Something else went wrong: {e}")

def signal_handler(sig, frame):
    logger.info("Received signal to terminate. Shutting down...")
    shutdown_event.set()

    # 停止 userAccount
    userAccount.stop()
    
    # 停止事件循环
    loop = asyncio.get_event_loop()
    loop.stop()

# 注册信号处理程序
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print('==> Login UserAccount...')
userAccount.start()

# 启动 TTS 任务
loop = asyncio.get_event_loop()
loop.run_until_complete(start_tts_task())

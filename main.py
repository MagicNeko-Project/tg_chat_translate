from telethon import TelegramClient, events, errors
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
API_ID = int(os.getenv('API_ID')) # Telethon 需要 int
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

# TTS 功能开关
TTS_ENABLED = os.getenv('TTS_ENABLED', 'true').lower() == 'true'

# 初始化 Telegram 客户端 (Telethon)
# 使用 "my_account.session" 作为会话文件名，与 Pyrogram 默认行为类似
client_telegram = TelegramClient("my_account", API_ID, API_HASH)

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
            await client_telegram.delete_messages(entity=chat_id, message_ids=[command_message_id])
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

            chat_id = job.chat_id
            logger.info(f"Processing TTS job: chat_id={chat_id}, text={job.text}")

            try:
                top_k = int(TTS_API_TOPK)
                top_p = float(TTS_API_TOPP)
                temperature = float(TTS_API_TEMPERATURE)
            except ValueError:
                logger.error("Failed to convert TTS parameters to float")
                request_queue.task_done() # Ensure task is marked done
                continue

            body = {
                "cha_name": TTS_C_NAME,
                "text": urllib.parse.quote(job.text),
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
            }
            headers = {"Content-Type": "application/json"}

            try:
                async with session.post(TTS_API_PATH, json=body, headers=headers, timeout=60) as response:
                    if response.status == 200:
                        content = await response.read()
                        audio = AudioSegment.from_file(io.BytesIO(content), format="wav")
                        buffer = NamedBytesIO(name="voice.ogg")
                        audio.export(buffer, format="ogg", codec="libopus")
                        buffer.seek(0)

                        # Telethon's client.is_connected() and connect() are different.
                        # Usually, Telethon handles reconnections automatically if client.run_until_disconnected() is used.
                        # For sending, we assume the client is connected if the event handler was triggered.

                        await client_telegram.send_file(
                            entity=chat_id,
                            file=buffer,
                            voice_note=True,
                            reply_to=job.reply_to_message_id
                        )
                        logger.info(f"TTS job completed successfully: chat_id={chat_id}")
                    else:
                        logger.error(f"TTS request failed: status={response.status}")
            except Exception as e:
                logger.error(f"TTS request exception: {e}")
                traceback.print_exc()
            finally:
                request_queue.task_done()

async def ai_translate(chat_id: int, input_text: str, event_message): # event_message is Telethon's Message object
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
        await event_message.edit(output_text) # Use event_message.edit for Telethon
        logger.info(f"Translated message edited: {output_text}")
    except errors.rpcerrorlist.MessageTooLongError as e: # Specific Telethon error
        logger.error(f"Message too long to edit: {e}")
        # Optionally delete original message if edit fails due to length
        try:
            await client_telegram.delete_messages(entity=chat_id, message_ids=[event_message.id])
            logger.info(f"Original message {event_message.id} deleted due to being too long for translation edit.")
        except Exception as del_e:
            logger.error(f"Failed to delete message {event_message.id} after edit failed: {del_e}")
    except Exception as e:
        logger.error(f"Translation request exception: {e}")
        traceback.print_exc()

@client_telegram.on(events.NewMessage)
async def message_handler(event): # event is NewMessage.Event
    message = event.message # This is Telethon's Message object
    chat_id = event.chat_id

    try:
        # 检查消息的发送者是否为 userbot 账号本身 和 消息是否为文本
        if not message.out or not message.text:
            return

        text_content = message.text # Use message.text

        if text_content.startswith('!fanyi'):
            replied_msg = await message.get_reply_message()
            input_text_fanyi = text_content[len('!fanyi '):].strip()

            if replied_msg:
                if input_text_fanyi: # Reply with content
                    logger.info(f"Processing !fanyi command with reply and content: {input_text_fanyi}")
                    await ai_translate(chat_id, input_text_fanyi, message)
                else: # Reply without content, use replied message's text
                    if replied_msg.text:
                        logger.info(f"Processing !fanyi command with reply: {replied_msg.text}")
                        await ai_translate(chat_id, replied_msg.text, message)
                    else:
                        logger.info("Replied message has no text for !fanyi.")
            elif input_text_fanyi: # No reply, but has content
                logger.info(f"Processing !fanyi command: {input_text_fanyi}")
                await ai_translate(chat_id, input_text_fanyi, message)

        elif text_content.startswith('!v'):
            if not TTS_ENABLED:
                logger.info("TTS功能已禁用。跳过 !v 命令。")
                # Optional: await event.edit("TTS 功能当前已禁用。") # This would edit the original !v message
                return

            input_text_tts = text_content[len('!v '):].strip()
            reply_to_msg_id_tts = message.reply_to_msg_id
            
            replied_msg_tts = None
            if not input_text_tts and reply_to_msg_id_tts:
                 replied_msg_tts = await message.get_reply_message()
                 if replied_msg_tts and replied_msg_tts.text:
                    input_text_tts = replied_msg_tts.text.strip()
                 else: # Replied message has no text or no replied message for text extraction
                    logger.error("No text provided for TTS, and replied message has no text.")
                    return # or send an error message back

            if input_text_tts:
                logger.info(f"Processing !v command: {input_text_tts}")
                # For !v, we delete the command message, so pass message.id
                await ai_tts_text(chat_id, input_text_tts,
                                  reply_to_message_id=reply_to_msg_id_tts,
                                  command_message_id=message.id)
            else:
                logger.error("No text provided for TTS")
        # else:
            # logger.info(f"Unhandled message from self: {text_content}")

    except errors.rpcerrorlist.MessageTooLongError: # Catching general MessageTooLong from Telethon
        if message and message.id:
            try:
                # Attempt to delete the problematic outgoing message if it was too long
                await client_telegram.delete_messages(entity=chat_id, message_ids=[message.id])
                logger.info(f"Original message {message.id} deleted as it might have been too long.")
            except Exception as delete_error:
                logger.error(f"删除消息失败: {delete_error}")
    except Exception as e:
        logger.error(f"Something else went wrong in message_handler: {e}")
        traceback.print_exc()
        if message and message.id: # Attempt to delete original message on other errors too
            try:
                await client_telegram.delete_messages(entity=chat_id, message_ids=[message.id])
            except Exception as delete_error:
                logger.error(f"删除消息失败 on general error: {delete_error}")


async def main():
    # 启动TTS任务处理队列
    # asyncio.create_task(start_tts_task()) # Run TTS task processor in background

    # 注册信号处理
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(signal_handler_async(s)))

    try:
        print('==> Connecting Telegram Client...')
        await client_telegram.connect()
        if not await client_telegram.is_user_authorized():
            print("Client not authorized. Please run interactively to login.")
            # Optionally, could add client_telegram.start() here to trigger login if not authorized
            # await client_telegram.send_code_request(PHONE_NUMBER)
            # await client_telegram.sign_in(phone=PHONE_NUMBER, code=input('Enter code: '))
            return # Exit if not authorized and not handling login here

        print('==> Userbot Connected. Starting TTS task processor...')
        # Start the TTS task processor as a background task
        # Make sure it's started after client is connected if it uses the client directly for checks like is_connected
        # but in this design, send_file is called, which should be fine.
        tts_processor_task = asyncio.create_task(start_tts_task())

        print("==> Userbot is running. Listening for messages...")
        await client_telegram.run_until_disconnected()
    finally:
        print("==> Userbot is shutting down...")
        shutdown_event.set() # Signal TTS task to stop
        if 'tts_processor_task' in locals() and not tts_processor_task.done():
            await asyncio.wait_for(tts_processor_task, timeout=5.0) # Wait for TTS task to finish
        await client_telegram.disconnect()
        print("==> Userbot disconnected.")

async def signal_handler_async(sig):
    logger.info(f"Received signal {sig}. Shutting down...")
    shutdown_event.set()
    # No need to stop loop explicitly here, run_until_disconnected will handle it on client disconnect
    # If client_telegram.disconnect() is not called by run_until_disconnected handler, call it here.
    # However, run_until_disconnected should exit on Ctrl+C.
    # Forcing a disconnect if not already happening:
    if client_telegram.is_connected():
        await client_telegram.disconnect()


if __name__ == '__main__':
    asyncio.run(main())

```markdown
# Telegram Userbot 中文使用文档

## 简介
本项目是一个基于 Python 开发的 Telegram Userbot，它通过集成 OpenAI API 和一个自定义的文本转语音 (TTS) 服务，为用户提供便捷的文本翻译和语音转换功能。Userbot 会监听用户自身账户在 Telegram 中发送的特定命令，并执行相应操作。

## 功能特性
- **文本翻译**:
    - 利用 OpenAI 的强大语言模型，实现中文与英文之间的双向智能翻译。
    - 翻译结果力求自然流畅，符合目标语言的表达习惯。
- **文本转语音 (TTS)**:
    - 将指定的文本内容转换为自然的人声语音。
    - 生成的语音以 OGG Opus 格式的音频消息形式在 Telegram 聊天中发送。
    - 支持配置不同的语音角色和语言（当前默认配置为中文）。

## 环境准备
在开始使用前，请确保您已准备好以下环境和信息：

- **Python**: 版本 3.7 或更高。
- **Telegram API 凭证**:
    - `API_ID`: 您的 Telegram 应用 API ID。
    - `API_HASH`: 您的 Telegram 应用 API Hash。
    (您可以从 [my.telegram.org](https://my.telegram.org/apps) 获取这些凭证。)
- **OpenAI API 密钥**:
    - `OPENAI_API_KEY`: 您的 OpenAI 账户 API 密钥。
- **TTS 服务配置** (如果您使用自定义的 TTS 服务):
    - `TTS_API_PATH`: TTS 服务的 API 访问路径。
    - 其他 TTS 相关参数如 `TTS_C_NAME`, `TTS_API_TOPK`, `TTS_API_TOPP`, `TTS_API_TEMPERATURE`。

## 安装与配置

请按照以下步骤安装和配置 Userbot：

1.  **克隆代码仓库**:
    ```bash
    git clone <仓库地址> # 例如：https://github.com/yourusername/telegram-userbot-openai.git
    cd <仓库目录> # 例如：telegram-userbot-openai
    ```

2.  **创建并激活 Python 虚拟环境**:
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **安装项目依赖**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **配置环境变量**:
    在项目的根目录下创建一个名为 `.env` 的文件。该文件用于存储您的 API 凭证和相关配置，确保不会将敏感信息直接写入代码中。
    文件内容格式如下，请将 `your_...` 部分替换为您的实际信息：

    ```env
    # Telegram API 凭证
    API_ID=your_api_id
    API_HASH=your_api_hash

    # OpenAI API 配置
    OPENAI_API_KEY=your_openai_api_key
    OPENAI_API_BASE=https://api.openai.com/v1 # OpenAI官方API地址，如果使用代理或自定义服务请修改
    OPENAI_ENGINE=gpt-3.5-turbo # 使用的OpenAI模型

    # TTS 服务配置
    TTS_API_PATH=your_tts_api_url # 例如：http://localhost:5000/tts
    TTS_API_LANGUAGE=zh # TTS 输出语言，例如 "zh" 代表中文
    TTS_C_NAME=your_character_name # TTS 角色名
    TTS_API_TOPK=50 # TTS top_k 参数
    TTS_API_TOPP=0.95 # TTS top_p 参数
    TTS_API_TEMPERATURE=0.7 # TTS temperature 参数
    ```

5.  **运行机器人**:
    完成上述配置后，在激活虚拟环境的终端中运行以下命令启动 Userbot：
    ```bash
    python main.py
    ```
    您应该会看到类似 `==> Login UserAccount...` 的输出，表示 Userbot 开始尝试登录。

## 使用方法

Userbot 启动并成功登录后，您可以在任何 Telegram 聊天中通过您自己的账户发送以下命令来使用其功能。

**重要提示**: Userbot **仅处理由您自己账户发送的、以指定命令开头的消息**。它不会响应来自其他用户或群聊中其他成员的消息。

### 文本翻译 (`!fanyi`)

使用 `!fanyi` 命令进行文本翻译。Userbot 会自动识别源语言（中文或英文）并翻译成另一种语言。

-   **直接翻译指定文本**:
    ```
    !fanyi 你好世界
    ```
    Userbot 会将 "你好世界" 翻译成英文。

    ```
    !fanyi Hello world
    ```
    Userbot 会将 "Hello world" 翻译成中文。

-   **翻译回复的消息**:
    如果您想翻译聊天中的某条消息，可以先回复该消息，然后发送 `!fanyi` 命令（命令后不带任何文本）。
    ```
    (回复某条包含 "Good morning" 的消息)
    !fanyi
    ```
    Userbot 会将 "Good morning" 翻译成中文。

-   **回复消息并指定翻译文本 (优先)**:
    如果您回复了一条消息，但在 `!fanyi` 命令后也提供了文本，那么 Userbot 将优先翻译您在命令后提供的文本，而不是被回复的消息内容。
    ```
    (回复某条消息)
    !fanyi 这是要翻译的新内容
    ```
    Userbot 会翻译 "这是要翻译的新内容"。

### 文本转语音 (`!v`)

使用 `!v` 命令将文本转换为语音消息。

-   **转换指定文本为语音**:
    ```
    !v 今天天气真好
    ```
    Userbot 会生成一条包含 "今天天气真好" 的语音消息。

-   **转换回复的消息为语音**:
    回复您想转换成语音的消息，然后发送 `!v` 命令（命令后不带任何文本）。
    ```
    (回复某条包含 "会议纪要如下" 的消息)
    !v
    ```
    Userbot 会将被回复消息的文本内容转换为语音。

-   **回复消息并指定转换文本 (优先)**:
    如果您回复了一条消息，但在 `!v` 命令后也提供了文本，Userbot 将优先转换您在命令后提供的文本。
    ```
    (回复某条消息)
    !v 这是要转语音的新内容
    ```
    Userbot 会将 "这是要转语音的新内容" 转换为语音。
    发送 `!v` 命令后，该命令消息本身会被自动删除，然后发送转换后的语音消息。

## 注意事项
- **日志**: Userbot 会在控制台输出运行日志，包括处理的命令、API 请求状态以及可能发生的错误。检查日志有助于排查问题。
- **TTS 队列**: TTS 请求会加入一个队列中进行处理，以避免并发问题。
- **消息过长**: 如果 OpenAI 返回的翻译结果过长，导致超出 Telegram 消息长度限制，原命令消息可能会被删除，并且无法发送翻译结果。
- **API 限制**: 请注意您所使用的 OpenAI API 和 TTS 服务的调用频率限制和配额，避免超出限制导致服务不可用。

## 贡献
欢迎对本项目进行贡献！如果您有任何改进建议或发现了 Bug，请随时提交 Issue 或 Pull Request。

## 许可证
本项目基于 MIT 许可证。详细信息请参阅项目根目录下的 `LICENSE` 文件（如果存在）。

## 致谢
- [Pyrogram](https://docs.pyrogram.org/) - 强大的 Telegram MTProto API 客户端库。
- [OpenAI](https://openai.com/) - 提供先进的自然语言处理 API。
- [Python-dotenv](https://pypi.org/project/python-dotenv/) - 方便地从 `.env` 文件加载环境变量。
- 以及其他所有依赖库的开发者。
```

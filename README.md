# Local QQ Ollama Bot

一个运行在 Windows 本机的 QQ 群聊机器人。NapCatQQ 负责接入普通 QQ 号，NoneBot2 通过 OneBot 11 反向 WebSocket 接收群消息，回答由本机 Ollama 的 `qwen3.5:9b` 生成。

机器人会静默记录白名单群内每位成员的文本消息、发送时间和 `@` 目标，但只在明确 `@机器人` 或引用回复机器人时回答。它支持文字和图片，对话记忆按“群号 + 用户 QQ + 人格”隔离，群友资料按“群号 + 用户 QQ”保存。

> NapCatQQ 是非官方 QQ 接入方式，无法消除封号和协议失效风险。请使用专门的小号，不要用于批量发送、主动加群或其他高频自动化行为。

## 环境要求

- Windows 10/11
- Python 3.11 或更高版本
- 已运行的 Ollama，默认地址为 `http://127.0.0.1:11434`
- 已安装模型：`qwen3.5:9b`
- 已登录测试小号的 NapCatQQ

本机当前 Ollama 已可访问，但当前终端检测到的 Python 是 3.9。请先安装 Python 3.11+，再创建虚拟环境。

## 安装

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

不要把真实令牌提交到 Git。可以参考 `.env.example`，在启动进程的环境变量中设置配置；如自行创建本地 `.env`，它已被 `.gitignore` 排除。

```powershell
$env:ONEBOT_ACCESS_TOKEN = "your-long-random-token"
$env:ALLOWED_GROUP_IDS = "123456789,987654321"
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
$env:OLLAMA_MODEL = "qwen3.5:9b"
$env:PROFILE_MESSAGE_LIMIT = "50"
$env:PROFILE_CHARACTER_BUDGET = "6000"
$env:PROFILE_TARGET_LIMIT = "5"
```

## NapCatQQ 配置

在 NapCatQQ 的 OneBot 11 网络配置中新增反向 WebSocket：

- URL：`ws://127.0.0.1:8080/onebot/v11/ws`
- Access Token：必须与 `ONEBOT_ACCESS_TOKEN` 完全一致
- 消息格式：消息段数组
- 心跳：启用

不要把 NapCat WebUI、OneBot WebSocket 或 Ollama 端口暴露到局域网或公网。

## 启动

先确认 Ollama 正常，再启动机器人，最后启用 NapCatQQ 的反向 WebSocket：

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/version
python -m qq_ollama_bot
```

群内使用方式：

```text
@机器人 你好
@机器人 /help
@机器人 /reset
@机器人 /reset all
@机器人 /status
@机器人 /persona list
@机器人 /persona assistant
@机器人 /persona
@机器人 /persona default
@机器人 @某位群友 他是谁
```

`/reset` 只清除发起人在当前群当前人格的对话记忆，`/reset all` 清除发起人在当前群全部人格的对话记忆。图片最多 4 张，单张最多 8 MB，总计最多 20 MB；图片内容不会写入数据库。

`/persona` 按“群号 + 用户”保存人格选择，重启后仍有效。`default` 是原有猫娘提示词，`assistant` 是内置的普通助手提示词。各人格使用独立的对话记忆，切回时继续该人格之前的对话。升级后的首次启动会自动为旧数据库添加人格字段，已有记录全部保留并归入 `default`；当前的 `assistant` 选择不变，但从空历史开始。

要增加人格，在 `qq_ollama_bot/prompts/` 中新增 UTF-8 编码的 `.md` 文件，文件名用小写英文字母、数字、`_` 或 `-`（最多 32 个字符），例如 `writer.md`；然后发送 `/persona writer`。修改文件后下次请求即生效。提示词文件不可用时机器人会明确报错，请切换到可用人格。切换仅影响之后的新请求；已经开始的请求沿用原人格，若期间发生切换，回复会标明来自切换前的请求，并只写入原人格的记忆。

请求日志会记录实际人格、历史消息条数和系统提示词的 SHA-256 前 12 位指纹，便于核对模型请求；日志不会输出完整提示词。

群友身份以 QQ 号作为内部键，每条白名单群消息都会更新群名片、昵称和群角色快照，并永久保存文本、发送时间及 `@` 关系。图片、文件和语音内容不会进入群聊归档。QQ 号不会发送给模型，`/reset` 也不会删除群聊归档或身份快照。

当一条提问同时 `@机器人` 和其他群友时，机器人会在当前群内检索每位目标最近 50 条发言，并在总计 6000 字符的预算内生成基于证据的概括；一次最多查询 5 人。模型只应总结可观察的常聊话题和表达习惯，不应推断敏感属性或未经证实的真实身份。

## 测试

```powershell
python -m pytest
```

测试完成后先在专用群前台观察 24 到 48 小时，再考虑使用 Windows 任务计划程序在登录后启动。生产运行时仍应保持 Ollama、机器人和 NapCatQQ 仅监听本机地址。

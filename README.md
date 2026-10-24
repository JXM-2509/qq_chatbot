# Local QQ Ollama Bot

一个运行在 Windows 本机的 QQ 群聊机器人。NapCatQQ 负责接入普通 QQ 号，NoneBot2 通过 OneBot 11 反向 WebSocket 接收群消息，模型后端可选择 DeepSeek API，或本机 Ollama 的 `qwen3.5:9b`。未设置 `MODEL_PROVIDER` 时继续使用 Ollama。

机器人会静默记录白名单群内每位成员的文本消息、发送时间和 `@` 目标。普通消息只在明确 `@机器人` 或引用回复机器人时回答；合并转发聊天记录会自动展开阅读并回应，无需 `@`。它支持文字和图片，对话记忆按“群号 + 用户 QQ + 人格”隔离，群友资料按“群号 + 用户 QQ”保存。

> NapCatQQ 是非官方 QQ 接入方式，无法消除封号和协议失效风险。请使用专门的小号，不要用于批量发送、主动加群或其他高频自动化行为。

## 项目目录

源码按功能划分，测试使用相同的功能分区：

| 目录 | 职责 |
| --- | --- |
| `qq_ollama_bot/chat/` | 消息提取、限流去重、图片读取、合并转发和转发队列 |
| `qq_ollama_bot/models/` | 统一模型接口、Ollama、DeepSeek、请求构建及错误处理 |
| `qq_ollama_bot/conversation/` | 对话记忆、人格选择、群成员身份和群聊归档 |
| `qq_ollama_bot/runtime/` | 进程管理、健康检查、故障恢复和 Windows 进程控制 |
| `qq_ollama_bot/plugins/` | NoneBot 聊天插件入口与业务编排 |
| `qq_ollama_bot/prompts/` | 可选人格提示词 |
| `tests/` | `chat/`、`models/`、`conversation/`、`runtime/` 分区测试与根目录配置测试 |
| `tools/runtime/` | 独立运行环境部署和便携 QQ 检查工具 |
| `tools/diagnostics/` | 图片、模型和合并转发诊断工具 |

`config.py` 保留在包根目录，`__main__.py` 是机器人入口，根目录的 `supervisor.py` 转调运行管理器。原有启动命令和 `start-project.cmd` 继续有效。

源码包中的 `qq_ollama_bot/runtime/` 存放 Python 代码；项目上一级的 `runtime/` 存放 QQ/NapCat 独立运行环境，两者用途不同。现有 `.env`、`data/` 和 `logs/` 的位置保持不变。

## 环境要求

- Windows 10/11
- Python 3.11 或更高版本
- DeepSeek 模式：可访问 DeepSeek API，并持有有效且有余额的 API Key
- Ollama 模式：本机 Ollama 和已安装的 `qwen3.5:9b`，默认地址为 `http://127.0.0.1:11434`
- 已登录测试小号的 NapCatQQ

请使用 Python 3.11+ 创建虚拟环境。

## 使用 DeepSeek API

本机 `.env` 已设置 `MODEL_PROVIDER=deepseek`，双击 `start-project.cmd` 后继续使用 DeepSeek，不会因为关闭终端而切回 Ollama。程序自身在没有任何后端配置时仍默认 Ollama，其他安装可在自己的 `.env` 中添加 `MODEL_PROVIDER=deepseek`。

启动时优先读取已有配置中的 `DEEPSEEK_API_KEY`。如果没有密钥，会提示在本机隐藏输入；输入过程中不显示字符，按回车确认。密钥仅保存在本次进程环境中，并传给机器人子进程，不写入文件或日志。

**每次关闭项目后重新启动，需要重新输入一次；同一次管理程序运行中的机器人自动重启不需要重输。** 如果启动环境已经提供有效的 `DEEPSEEK_API_KEY`，则不会提示输入。下面的 PowerShell 示例只对当前终端及其子进程有效，关闭终端后不会保留密钥。当前启动方式不提供记住密钥功能。

直接运行 `python -m qq_ollama_bot` 时也支持交互隐藏输入；只读检查 `--check` 和没有交互终端的启动不会询问密钥，缺少时明确报错。诊断工具仍需先通过环境变量提供密钥。

如需临时覆盖配置或提前提供密钥，可在项目目录打开 PowerShell，设置以下变量后从同一个窗口启动项目。进程环境变量优先于本地 `.env`；也可以直接双击启动脚本，按提示隐藏输入密钥。不要把真实密钥写进示例或提交到 Git。

```powershell
$env:MODEL_PROVIDER = "deepseek"
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
$env:DEEPSEEK_MODEL = "deepseek-flash"
$env:DEEPSEEK_MAX_TOKENS = "800"
$deepseekCredential = Read-Host "DeepSeek API Key" -AsSecureString
$env:DEEPSEEK_API_KEY = [System.Net.NetworkCredential]::new("", $deepseekCredential).Password
.\start-project.cmd
```

该模式统一处理文字、图片和合并转发，不需要安装或启动 Ollama。首次切换前先正常停止旧项目，避免端口冲突。人格、数据库、群白名单、触发条件、限流和记忆隔离继续沿用。切换回本地模型时，停止项目，在同一窗口执行 `$env:MODEL_PROVIDER = "ollama"`，再运行 `.\start-project.cmd`。

默认关闭模型思考，使用非流式回复、温度 `0.7`、最多 `800` 输出 token，并保持一次只生成一个回复。文字沿用 `REQUEST_TIMEOUT_SECONDS`（默认 120 秒），图片与转发分别沿用 `IMAGE_MODEL_TIMEOUT_SECONDS` 和 `FORWARD_MODEL_TIMEOUT_SECONDS`（默认 600 秒）；云端不会自动延长超时、重试生成或切回本地。

图片通过当前请求内的 Base64 数据发送，不向 DeepSeek 提供 QQ 图片链接。仍限制最多 4 张、单张 8 MiB、合计 20 MiB；最终请求体另限制为 48 MiB。不上传历史图片，只保留已有的图片占位文字。配置模型应支持图片；默认 `deepseek-flash`，具体可用模型和能力以官方 `/models` 为准。

DeepSeek 按 token 计费。参与本次推理的当前消息、相关对话历史、人格提示、群友发言证据、转发材料及本次图片会发送至配置的 API 服务。其他群消息不会仅因被本地记录就自动上传。诊断日志不保存 API Key、请求正文、图片或推理内容。

`/status` 和 `/healthz` 只通过 `/models` 检查接口、鉴权和模型信息，云端结果缓存 60 秒，不调用生成接口，也不保证推理余额充足。实际生成发生鉴权失败、余额不足或限流时，机器人会给出对应提示。云端异常不会触发 QQ 或机器人的重启；健康接口中的 `ollama_reachable` 在云端模式为 `null`。

配置密钥后，可从项目目录运行以下诊断；会调用当前后端、发送指定测试图片并产生 API 费用，但不会向 QQ 发送消息：

```powershell
.\.venv\Scripts\python.exe tools/diagnostics/check_image_model.py C:/path/to/test.png
.\.venv\Scripts\python.exe tools/diagnostics/check_forward_model.py C:/path/to/test.png --with-history
```

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

### 项目统一启动（Windows）

运行项目根目录的 `start-project.cmd`。本机默认读取 `.env` 中的 DeepSeek 选择，缺少 API Key 时先隐藏输入。它按顺序启动机器人和项目独立的 QQ/NapCat，并每 15 秒检查运行状态；仅 Ollama 模式会先启动或检查本地模型。如果通过窗口内环境变量覆盖配置，须从该窗口启动。请先关闭原来单独启动的机器人和 NapCat 终端；如果端口被它们占用，管理程序会提示并退出，不会关闭日常 QQ。

保持这个管理终端打开即可，锁屏不会停止项目。按 `Q` 或 `Ctrl+C` 停止；直接关闭终端也会结束本次启动的全部项目进程。原本已经运行并被复用的 Ollama 保持运行。按 `R` 可在修复故障原因后重新尝试自动恢复。

QQ 登录失效但仍可快速登录时，NapCat 优先自行恢复；持续异常时管理程序只重启故障组件。遇到二维码或安全验证会暂停 QQ 重启、弹出一次本机提醒，打开 `http://127.0.0.1:6099/webui` 使用原有 WebUI 密码完成登录，随后自动恢复。程序不能绕过 QQ 的扫码或安全验证。

同一组件 15 分钟内恢复失败 3 次，会停止自动重启并提示检查。缺少模型、配置不一致或外部进程占用端口时，不循环重启。模型生成期间和没有群消息时不会据此判定掉线。

只读健康接口为 `http://127.0.0.1:8080/healthz`（若修改了 `BOT_PORT`，使用对应端口）。返回机器人初始化、OneBot 连接、QQ 在线、当前模型后端和配置模型状态；不返回令牌或聊天内容。诊断日志存放在 `logs/supervisor-*.log`，按大小分卷，不自动删除旧日志；磁盘空间需自行管理。

运行版本固定为 NapCat `4.18.33`、QQ `9.9.33-52230`，位于上一级 `runtime/`。原来的 `napcat/` 和日常 QQ 安装保留，可停止新管理终端后按旧方法回退。程序读取原 `.env`，数据库继续使用其中配置的路径；不创建新的聊天数据库。安装准备脚本是 `tools/runtime/deploy_runtime.py`，默认仅校验本地附件和准备运行目录，显式添加 `--download` 才下载固定版本。下载包校验 SHA-256，不执行 QQ 系统安装程序。

可用 `.venv\Scripts\python.exe -m qq_ollama_bot.supervisor --check` 检查依赖和端口，不启动项目。独立 QQ 的无登录检查工具为 `tools/runtime/check_portable_qq.py`：使用隔离端口、禁用 OneBot 连接，只检查二维码登录服务，测试后结束自己的进程，不登录账号。

没有更改 Windows 睡眠、合盖或开机启动设置。电脑睡眠或休眠期间无法处理消息，唤醒后重新检查并恢复。进程重启会中断生成中的回复、丢失内存转发队列；已保存的聊天记录和人格选择保留，离线期间的消息不保证补收。

首次使用时在白名单测试群验证文字、图片、合并转发、人格和记忆；在网络与登录凭据正常时，目标是故障或断网恢复后 3 分钟内恢复运行。连续观察至少 24 小时，遇到无法恢复时保留诊断日志。锁屏收发、真实账号扫码和长期运行必须在实际使用中完成验收。

Ollama 模式下，先确认 Ollama 正常，再启动机器人，最后启用 NapCatQQ 的反向 WebSocket：

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

普通图片提问使用独立的模型等待时间：`IMAGE_MODEL_TIMEOUT_SECONDS` 默认 600 秒；纯文字仍使用 `REQUEST_TIMEOUT_SECONDS`，默认 120 秒。CPU 推理时图片编码和生成可能明显慢于文字，等待过程中不重启模型或 QQ。图片超时会提示“图片分析超时”；管理日志仅保存图片数量、读取及模型耗时等诊断数字，不保存图片内容或下载地址。可通过启动环境变量覆盖，无需修改 `.env`。新增的图片诊断日志在下次启动管理程序后生效。

模型输入容量默认 8192 tokens。如果模型明确报告文字、历史和附图合计超限，程序按实际输入长度预留回复空间，扩大本次请求容量后自动重试一次，最多 32768 tokens；不会为此删除历史或省略已读取的附图。仍然超限时明确提示拆分记录或减少图片，后续队列继续处理，不重启 QQ 或模型。更长记录在 CPU 上可能需要更多处理时间。

合并转发的扩容重试还会按实际输入长度增加本次等待时间，默认基础时限为 600 秒，最多延长至 1200 秒；如果用户配置的基础时限更长，则保留该配置。普通图片和文字提问的时限不受这一调整影响。

## 合并转发聊天记录

在白名单群直接发送合并转发聊天记录，机器人会读取内部文字和图片，并按发送者当前选择的人格自然简短回应。有附带问题时优先回答问题，没有问题时根据记录中的话题、情绪或疑问接话。一条消息中的多份转发记录合并为一次回应，回复会引用原消息。

发送者显示名称、可用的发送时间和发言顺序会保留，图片编号对应具体发言。转发内部的命令和 `@` 只是引用内容，不会执行命令、触发群友查询或更新当前群的成员资料。语音、视频、文件和卡片等仅保留占位说明。

连续发送的记录会按入队顺序逐条处理，不受普通请求的用户冷却时间限制；模型忙时等待。最多等待 20 条，队列满时明确提示重新发送。人格和提示词在入队时固定；排队或生成期间切换人格，回复会标明来自切换前的人格，并写入原人格的记忆。待处理任务只保存在内存中，重启后不恢复。

本次实际送给模型的文字记录、图片占位说明和回复进入“群号 + 用户 QQ + 人格”的对话记忆。之后可 `@机器人` 继续追问，沿用现有记忆轮数、字符预算和保留期限。图片原文件和编码不会写入数据库，后续追问不能重新查看之前的图片。

默认限制如下，可通过启动进程的环境变量覆盖，无需修改 NapCat 配置：

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `FORWARD_MAX_DEPTH` | `3` | 最多展开的转发层数，外层算第 1 层 |
| `FORWARD_MESSAGE_LIMIT` | `100` | 一次请求最多读取的内部消息数，同时限制转发展开次数 |
| `FORWARD_CHARACTER_BUDGET` | `6000` | 引用材料字符预算，包含格式、发送者标签和读取说明 |
| `FORWARD_TIMEOUT_SECONDS` | `30` | 每次获取转发详情、引用节点或图片的超时秒数 |
| `FORWARD_MODEL_TIMEOUT_SECONDS` | `600` | 转发记录的模型回复等待秒数；普通聊天仍使用 `REQUEST_TIMEOUT_SECONDS`，默认 120 秒 |
| `FORWARD_QUEUE_SIZE` | `20` | 最多等待的消息数，不含正在处理的一条 |

图片沿用 `IMAGE_MAX_COUNT`、`IMAGE_MAX_BYTES`、`IMAGE_TOTAL_MAX_BYTES`：默认按顺序尝试前 4 张，单张最多 8 MB，总计最多 20 MB。图片失败或超限时会标记未读取，继续处理其他可用内容。超过文字、消息数或嵌套层数限制时，按原始顺序保留可读部分，并明确说明截断或缺失；记录过期、为空或全部不可读时直接说明原因。

图片优先通过 NapCat 获取本地缓存；缓存不可用时尝试接口返回的图片编码，再通过图片消息附带的 QQ 图片 CDN 地址回退下载。只接受受支持的 QQ 图片域名，并校验重定向、实际大小和图片格式；不下载任意外部网址。仍仅支持 JPEG、PNG、WebP，GIF 动图暂不识别。下载地址过期、缓存不可读、格式不支持和大小超限会分别记录错误原因，日志不输出带签名的图片地址。

真实群验收：在专用白名单测试群发送一份带多位发送者、文字和图片的合并转发记录，确认无需 `@` 即回应且引用原消息；随后 `@机器人` 提问其中的文字内容，再连续发送两份不同记录确认逐条回复。

## 测试

```powershell
python -m pytest
```

测试完成后先在专用群前台观察 24 到 48 小时，再考虑使用 Windows 任务计划程序在登录后启动。生产运行时仍应保持 Ollama、机器人和 NapCatQQ 仅监听本机地址。

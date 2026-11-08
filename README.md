# MaskFlow

[中文](#中文) | [English](#english)

---

## 中文

MaskFlow 是一个本地运行的视频隐私打码演示项目。上传视频，用一句话描述打码规则，例如“遮挡所有人脸，但穿红色衣服的人不打码”，几分钟后即可对比查看处理结果。

Ling-3.2-flash 只负责把文字规则转换成代码，检测、跟踪、打码和视频编码都在本机完成，**视频内容不会发送给 Ling API**。

### 工作流程

```text
浏览器 → FastAPI → Ling-3.2-flash：把规则生成为 decide(obj) 函数
                 → 单任务 GPU 队列：
                     1. YOLO11n 检测并跟踪物体，YOLO11n-pose 估算人脸位置
                     2. 在独立子进程中对每个检测结果运行 decide()
                     3. 对选中的区域打码，并用 NVENC 编码
```

同一视频的检测结果会缓存，换规则时无需重新检测。

### 可识别的目标

人脸、人、汽车、公交车、卡车、摩托车、自行车、笔记本电脑、电视、手机。规则还可以使用出现时间、画面位置、目标大小、颜色和跟踪 ID。

### 环境要求

- 支持 CUDA 和 NVENC 的 NVIDIA 显卡
- Python 3.10（已在此版本上测试）
- 已安装 CUDA 版 [PyTorch](https://pytorch.org/get-started/locally/)
- `PATH` 中有支持 `h264_nvenc` 的 `ffmpeg`
- Ling API Key

### 快速开始

以下命令适用于 Windows PowerShell。创建虚拟环境时使用 `--system-site-packages`，可以复用已安装的 PyTorch。

```powershell
git clone https://github.com/fat1/maskflow.git
cd maskflow
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
"LING_API_KEY=你的密钥" | Out-File -Encoding ascii .env
.\.venv\Scripts\python.exe app.py
```

打开 <http://127.0.0.1:8000>。局域网内的其他设备可访问 `http://<本机内网IP>:8000`，可能需要在防火墙中放行 8000 端口。首次运行时会自动下载 YOLO 模型权重。

### 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/jobs` | 表单字段 `video` 和 `rule`，返回任务 ID |
| `GET` | `/api/jobs/{id}` | 查询状态、阶段、生成的代码和统计信息 |
| `GET` | `/api/jobs/{id}/input` | 获取原视频 |
| `GET` | `/api/jobs/{id}/output` | 下载处理后的视频 |

### 已知限制

- 这是演示项目，请勿用于合规场景，打码结果需要人工复核。
- 人脸位置由人体关键点估算，侧脸、远处或贴边的人脸可能漏检。
- 不支持检测车牌，提到车牌的规则会改为遮挡整辆车。
- 视频上限为 3600 帧、200MB。
- 每次提交都会重新调用 Ling 生成代码，任务状态只保存在内存中，重启服务后会丢失。

### 安全提示

**请勿将此服务暴露到公网。** 服务默认监听所有网络接口，且没有登录验证。同一网络内的人都可以提交任务、消耗你的 API 额度，持有任务链接的人还可以查看对应视频。生成的代码经过 AST 白名单检查，并在独立子进程中运行，但没有操作系统级隔离。请只在可信网络中使用，并且不要提交 `.env`。

### 许可证

本项目尚未选择开源许可证。依赖项有各自的许可证，其中 Ultralytics YOLO 采用 AGPL-3.0 或商业授权。

---

## English

MaskFlow is a locally hosted demo for masking private details in video. Upload a video, describe the rule in one sentence, such as "Mask all faces, except people wearing red", and compare the result a few minutes later.

Ling-3.2-flash only turns the written rule into code. Detection, tracking, masking and encoding all run on your machine, so **video content is never sent to the Ling API**.

### How it works

```text
Browser → FastAPI → Ling-3.2-flash: turns the rule into a decide(obj) function
                  → Single-job GPU queue:
                      1. YOLO11n detects and tracks objects; YOLO11n-pose estimates face positions
                      2. decide() runs on every detection in a separate subprocess
                      3. Selected regions are pixelated and encoded with NVENC
```

Detections are cached per video, so trying a new rule does not repeat detection.

### Supported targets

Faces, people, cars, buses, trucks, motorcycles, bicycles, laptops, TVs and phones. Rules can also use time, position in the frame, object size, color and track ID.

### Requirements

- An NVIDIA GPU with CUDA and NVENC
- Python 3.10 (the tested version)
- A CUDA build of [PyTorch](https://pytorch.org/get-started/locally/)
- `ffmpeg` with `h264_nvenc` on your `PATH`
- A Ling API key

### Quick start

These commands are for Windows PowerShell. `--system-site-packages` lets the virtual environment reuse your installed PyTorch.

```powershell
git clone https://github.com/fat1/maskflow.git
cd maskflow
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
"LING_API_KEY=your-key" | Out-File -Encoding ascii .env
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:8000>. Other devices on your network can use `http://<your-LAN-IP>:8000`, which may require allowing port 8000 through your firewall. The YOLO weights download automatically on first run.

### API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/jobs` | Form fields `video` and `rule`; returns a job ID |
| `GET` | `/api/jobs/{id}` | Status, stage, generated code and statistics |
| `GET` | `/api/jobs/{id}/input` | The original video |
| `GET` | `/api/jobs/{id}/output` | Download the masked video |

### Known limitations

- This is a demo. Do not rely on it for compliance, and review masked videos by hand.
- Face positions are estimated from body keypoints, so profile, distant or edge-of-frame faces can be missed.
- Licence plates are not detected. Rules that mention them mask the whole vehicle instead.
- Videos are limited to 3600 frames and 200 MB.
- Every submission calls Ling to generate code again. Job state is held in memory and is lost on restart.

### Security

**Do not expose this service to the internet.** It listens on all network interfaces and has no authentication. Anyone on the same network can submit jobs and spend your API quota, and anyone with a job link can view that video. Generated code passes an AST allowlist and runs in a separate subprocess, but it has no OS-level isolation. Use it only on trusted networks, and never commit `.env`.

### License

No open-source license has been chosen yet. Dependencies have their own licenses; Ultralytics YOLO is available under AGPL-3.0 or a commercial license.

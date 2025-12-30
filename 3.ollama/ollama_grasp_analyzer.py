"""
Ollama 抓取指令分析器
调用本地Ollama API，分析用户输入的自然语言指令，判断是否为抓取任务
"""

import requests
import json
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class GraspResult:
    """抓取分析结果"""
    is_capture: bool
    target: Optional[str]
    destination: Optional[str]
    raw_response: str = ""
    inference_time: float = 0.0  # 推理耗时(秒)


class OllamaGraspAnalyzer:
    """Ollama抓取指令分析器"""

    SYSTEM_PROMPT = """你是机器人指令解析器。分析用户指令，提取抓取任务信息。

## 输出字段
- IsCapture: 是否为抓取/移动物体任务 (true/false)
- Target: 待抓取物体 (物体名或None)
- Destination: 放置位置 (位置名或None)

## 判断规则
IsCapture=true: 含抓/拿/取/移/放/递/搬/捡/拾等动作词且涉及物体
IsCapture=false: 询问、闲聊、无物体操作

## 输出格式
仅输出JSON，无需任何解释：
{"IsCapture": bool, "Target": "string", "Destination": "string"}

## 示例
输入: 帮我拿苹果 → {"IsCapture": true, "Target": "苹果", "Destination": "None"}
输入: 把杯子放桌上 → {"IsCapture": true, "Target": "杯子", "Destination": "桌上"}
输入: 把香蕉移到左边 → {"IsCapture": true, "Target": "香蕉", "Destination": "左边"}
输入: 今天天气如何 → {"IsCapture": false, "Target": "None", "Destination": "None"}
输入: 桌上有什么 → {"IsCapture": false, "Target": "None", "Destination": "None"}"""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 30
    ):
        """
        初始化分析器
        Args:
            model: Ollama模型名称
            base_url: Ollama API地址
            timeout: 请求超时时间(秒)
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._running = False
        self._input_queue = queue.Queue()
        self._thread = None

    def _call_ollama(self, text: str) -> str:
        """调用Ollama API"""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            "stream": False,
            "keep_alive": "5m",
            "options": {
                "temperature": 0.1
            }
        }

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        result = response.json()
        return result.get("message", {}).get("content", "")

    def _parse_response(self, response: str, inference_time: float = 0.0) -> GraspResult:
        """解析模型响应"""
        try:
            # 尝试提取JSON
            response = response.strip()
            # 处理可能的markdown代码块
            if "```" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                response = response[start:end]

            data = json.loads(response)

            is_capture = data.get("IsCapture", False)
            target = data.get("Target", "None")
            destination = data.get("Destination", "None")

            # 处理None字符串
            if target in ["None", "none", "", None]:
                target = None
            if destination in ["None", "none", "", None]:
                destination = None

            return GraspResult(
                is_capture=is_capture,
                target=target,
                destination=destination,
                raw_response=response,
                inference_time=inference_time
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[解析错误] {e}, 原始响应: {response}")
            return GraspResult(
                is_capture=False,
                target=None,
                destination=None,
                raw_response=response,
                inference_time=inference_time
            )

    def analyze(self, text: str) -> Optional[GraspResult]:
        """
        分析单条指令（同步方法）
        Args:
            text: 用户输入的文本指令
        Returns:
            GraspResult: 分析结果
        """
        if not text or not text.strip():
            return None

        try:
            start_time = time.perf_counter()
            response = self._call_ollama(text.strip())
            inference_time = time.perf_counter() - start_time
            return self._parse_response(response, inference_time)
        except requests.RequestException as e:
            print(f"[API错误] {e}")
            return None

    def _run_loop(self, callback):
        """内部运行循环"""
        while self._running:
            try:
                # 非阻塞获取，超时0.1秒
                text = self._input_queue.get(timeout=0.1)
                if text is None:
                    continue
                result = self.analyze(text)
                if result and callback:
                    callback(result)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[运行错误] {e}")

    def run(self, callback=None):
        """
        启动分析器（异步模式）
        Args:
            callback: 结果回调函数，接收GraspResult参数
        """
        if self._running:
            print("分析器已在运行中")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(callback,),
            daemon=True
        )
        self._thread.start()
        print(f"[Ollama分析器] 已启动，模型: {self.model}")

    def input(self, text: str):
        """
        输入文本进行分析（异步模式下使用）
        Args:
            text: 用户输入的文本
        """
        if text and text.strip():
            self._input_queue.put(text)

    def stop(self):
        """停止分析器"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None
        print("[Ollama分析器] 已停止")


# 使用示例
if __name__ == "__main__":
    def on_result(result: GraspResult):
        """结果回调"""
        print(f"\n{'='*50}")
        print(f"IsCapture: {result.is_capture}")
        print(f"Target: {result.target}")
        print(f"Destination: {result.destination}")
        print(f"推理耗时: {result.inference_time*1000:.1f}ms")
        print(f"{'='*50}\n")

    # 创建分析器
    analyzer = OllamaGraspAnalyzer(model="qwen2.5:3b")

    # 启动异步模式
    analyzer.run(callback=on_result)

    print("输入指令进行分析（输入 'quit' 退出）：")
    try:
        while True:
            user_input = input("> ")
            if user_input.lower() == "quit":
                break
            analyzer.input(user_input)
    except KeyboardInterrupt:
        pass
    finally:
        analyzer.stop()

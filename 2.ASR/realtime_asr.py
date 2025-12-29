"""
实时语音识别脚本
使用 FunASR (阿里开源ASR模型) + PyAudio 实现麦克风实时语音转文字
"""

import pyaudio
import numpy as np
from funasr import AutoModel
import torch

class RealtimeASR:
    """实时语音识别类"""


    # 音频参数配置
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1600
    CHANNELS = 1
    FORMAT = pyaudio.paInt16

    # 静音检测参数
    SILENCE_THRESHOLD = 500
    SILENCE_CHUNKS = 10

    def __init__(self, model_path="paraformer-zh", vad_model_path="fsmn-vad", punc_model_path=None, device="cuda:0"):
        """
        初始化ASR
        Args:
            model_path: ASR模型本地路径
            vad_model_path: VAD模型本地路径
            punc_model_path: 标点模型本地路径
            device: 运行设备，"cuda:0" 或 "cpu"
        """
        self.model_path = model_path
        self.vad_model_path = vad_model_path
        self.punc_model_path = punc_model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.asr_model = None
        self.pyaudio = None
        self.stream = None
        self._running = False

        # 状态变量
        self._audio_buffer = []
        self._silence_count = 0
        self._is_speaking = False

    def _init_model(self):
        """加载本地预训练模型"""
        if self.asr_model is None:
            # print("正在加载本地ASR模型，请稍候...")
            # print(f"  ASR模型: {self.model_path}")
            # print(f"  VAD模型: {self.vad_model_path}")
            # print(f"  标点模型: {self.punc_model_path}")
            self.asr_model = AutoModel(
                model=self.model_path,
                vad_model=self.vad_model_path,
                punc_model=self.punc_model_path,
                device=self.device,
                disable_update=True  # 禁止自动下载更新
            )
            print("模型加载完成！")

    def _init_audio(self):
        """初始化音频流"""
        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SIZE
        )

    def _is_silence(self, audio_data):
        """判断是否为静音"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        return np.abs(audio_array).mean() < self.SILENCE_THRESHOLD

    def _recognize(self, audio_buffer):
        """执行语音识别"""
        audio_data = b''.join(audio_buffer)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0

        result = self.asr_model.generate(input=audio_float, batch_size_s=300)

        if result and len(result) > 0:
            text = result[0].get("text", "")
            if text.strip():
                return text
        return None

    def run(self):
        """
        运行实时语音识别（生成器）
        Yields:
            str or None: 识别到的文字，未识别到返回None
        """
        self._init_model()
        self._init_audio()
        self._running = True

        # 重置状态
        self._audio_buffer = []
        self._silence_count = 0
        self._is_speaking = False

        print("\n实时语音识别已启动...")

        while self._running:
            try:
                audio_chunk = self.stream.read(
                    self.CHUNK_SIZE,
                    exception_on_overflow=False
                )
            except Exception:
                yield None
                continue

            if self._is_silence(audio_chunk):
                self._silence_count += 1
                if self._is_speaking and self._silence_count > self.SILENCE_CHUNKS:
                    # 说话结束，进行识别
                    if len(self._audio_buffer) > 0:
                        text = self._recognize(self._audio_buffer)
                        # 重置状态
                        self._audio_buffer = []
                        self._is_speaking = False
                        self._silence_count = 0
                        yield text
                    else:
                        yield None
                else:
                    yield None
            else:
                # 检测到声音
                self._silence_count = 0
                self._is_speaking = True
                self._audio_buffer.append(audio_chunk)
                yield None

    def stop(self):
        self._running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.pyaudio:
            self.pyaudio.terminate()
            self.pyaudio = None
        print("语音识别已停止")


# 使用示例
if __name__ == "__main__":
    model_path = r"./iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    vad_model_path = r"./iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    punc_model_path = r"./iic/punc_ct-transformer_cn-en-common-vocab471067-large"

    asr = RealtimeASR(
        model_path=model_path,
        vad_model_path=vad_model_path,
        punc_model_path=None  # 不使用标点模型可设为None
    )
    try:
        for text in asr.run():
            if text is not None:
                print(f"识别结果: {text}")
    except KeyboardInterrupt:
        asr.stop()

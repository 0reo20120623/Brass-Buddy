import streamlit as st
import librosa
import numpy as np
import base64
import matplotlib.pyplot as plt
import librosa.display
from pydub import AudioSegment
from pydub.generators import Sine
import tempfile
import time
from io import BytesIO
from PIL import Image

# =========================
# 🎨 カラーテーマ & ロゴ
# =========================
theme = st.radio("テーマを選択してください", ["ホワイト", "ブラック"], horizontal=True)

if theme == "ブラック":
    st.markdown(
        """
        <style>
            body {
                background-color: #111111;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
            body {
                background-color: white;
                color: black;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# ロゴ表示（ファイル名を合わせて同じフォルダに置いてね）
# 変更後（Streamlit Cloud対応！）
st.image("brassbuddy_logo.png", use_column_width=True)


st.markdown(
    """
    <h1 style='text-align: center; color: #4A90E2;'>🎺 BrassBuddy</h1>
    <h3 style='text-align: center; color: gray;'>AIで上達をサポートする金管楽器練習アプリ</h3>
    <hr style='border-top: 3px solid #F5A623;'>
    """,
    unsafe_allow_html=True
)

# =========================
# 🎼 課題選択 & 譜面表示
# =========================
task = st.selectbox("今日の練習課題を選んでください", [
    "ロングトーンF",
    "スケール練習（C-Dur）",
    "タンギング練習（4分音符）",
    "課題曲：Fly Me to the Moon"
])
st.write(f"選択中の課題：{task}")

score_files = {
    "ロングトーンF": "long_tone_a.pdf",
    "スケール練習（C-Dur）": "scale_c.pdf",
    "タンギング練習（4分音符）": "tonguing_qn.pdf",
    "課題曲：Fly Me to the Moon": "fly_me_to_the_moon.pdf",
}

def display_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("譜面ファイルが見つかりません。PDFファイルを同じフォルダに置いてください。")

score_file = score_files.get(task)
if score_file:
    st.subheader("譜面表示")
    display_pdf(score_file)

# =========================
# 🎧 音声アップロード & ピッチ・リズム解析
# =========================
uploaded_file = st.file_uploader("演奏音声ファイルをアップロード（WAV/MP3/M4A）", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # M4A変換
    if uploaded_file.name.endswith(".m4a"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            audio = AudioSegment.from_file(uploaded_file, format="m4a")
            audio.export(tmpfile.name, format="wav")
            file_to_load = tmpfile.name
    else:
        file_to_load = uploaded_file

    y, sr = librosa.load(file_to_load, sr=None)

    # 🎯 ピッチ解析（A3=220Hz 中心の2オクターブ）
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('A2'), fmax=librosa.note_to_hz('A4'))
    valid_f0 = f0[~np.isnan(f0)]

    if len(valid_f0) > 0:
        avg_pitch = np.mean(valid_f0)
        std_pitch = np.std(valid_f0)
        pitch_score = max(0, 100 - std_pitch * 500)

        st.write(f"平均ピッチ: {avg_pitch:.2f} Hz")
        st.write(f"ピッチ安定スコア：{pitch_score:.1f} / 100")
        st.write(f"ピッチの標準偏差: {std_pitch:.4f} Hz")

        # 🎵 ピッチ評価コメント（基準 442Hz）
        target_pitch = 442
        tolerance = 10
        if avg_pitch < target_pitch - tolerance:
            st.info("🎵 音程が少し低めです。息の圧やアンブシュアを意識してみましょう。")
        elif avg_pitch > target_pitch + tolerance:
            st.info("🎵 音程がやや高めです。リラックスして吹くよう心がけてみましょう。")
        else:
            st.success("🎯 ピッチは良好です！この調子で続けましょう✨")

        # 🥁 リズム解析（onset検出）
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        st.subheader("リズム解析（波形と音の入り）")
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.6, ax=ax)
        ax.vlines(onset_times, -1, 1, color='r', linestyle='dashed', label='Onsets')
        ax.set_title("Waveform with Detected Onsets")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("音程が検出できませんでした。録音を確認してください。")

# =========================
# 🕒 メトロノーム
# =========================
st.subheader("🕒 練習用メトロノーム")

bpm = st.slider("テンポ (BPM)", min_value=40, max_value=240, value=120, step=1)

def generate_click_sound(duration_ms=100, freq=1000):
    tone = Sine(freq).to_audio_segment(duration=duration_ms).apply_gain(-3)
    buffer = BytesIO()
    tone.export(buffer, format="wav")
    return buffer.getvalue()

if st.button("メトロノーム再生（8回）"):
    interval = 60.0 / bpm
    st.write("🟢 メトロノーム再生中...")
    for _ in range(8):
        st.audio(generate_click_sound(), format="audio/wav")
        time.sleep(interval)

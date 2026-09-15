"""
Simple Streamlit UI for the Log Analyst Agent.

Run with:
    streamlit run streamlit_app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

import agent
import tools
from data_loader import SUPPORTED_UPLOAD_EXTENSIONS, load_uploaded_file

st.set_page_config(page_title="Log Analyst Agent", page_icon="🛠️", layout="wide")

st.title("🛠️ Log Analyst Agent")
st.caption(
    "AI4I 2020 Predictive Maintenance Dataset (UCI, CC BY 4.0) üzerinde "
    "doğal dilde soru sorun; agent gerekli analiz araçlarını çağırıp size "
    "sonucu açıklar."
)

with st.sidebar:
    st.header("Veri Seti")
    upload_types = [ext.lstrip(".") for ext in SUPPORTED_UPLOAD_EXTENSIONS]
    uploaded_file = st.file_uploader(
        f"Farklı bir log dosyası yükle (opsiyonel, AI4I kolon formatında) - {', '.join(upload_types)}",
        type=upload_types,
    )
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)
        try:
            tools._df = load_uploaded_file(tmp_path, uploaded_file.name)
            st.success(f"'{uploaded_file.name}' yüklendi ve kullanılıyor.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Dosya okunamadı: {exc}")
        finally:
            tmp_path.unlink(missing_ok=True)

    df_preview = tools._get_df()
    st.metric("Kayıt sayısı", len(df_preview))
    st.metric("Arıza sayısı", int(df_preview["machine_failure"].sum()))
    with st.expander("İlk 5 satır"):
        st.dataframe(df_preview.head())

    st.divider()
    st.caption(f"Model: `{agent.MODEL}`")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Örn: Tool wear kolonunda anormallik var mı?")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        for call in turn["tool_calls"]:
            if call["name"] == "plot_trend" and "image_path" in call["result"]:
                image_path = call["result"]["image_path"]
                if Path(image_path).exists():
                    st.image(image_path, caption=call["result"]["column"])
        with st.expander("Tool çağrıları (detay)"):
            st.json(turn["tool_calls"])

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analiz ediliyor..."):
            result = agent.run_agent(question)

        st.markdown(result["answer"])
        for call in result["tool_calls"]:
            if call["name"] == "plot_trend" and "image_path" in call["result"]:
                image_path = call["result"]["image_path"]
                if Path(image_path).exists():
                    st.image(image_path, caption=call["result"]["column"])
        with st.expander("Tool çağrıları (detay)"):
            st.json(result["tool_calls"])

    st.session_state.history.append(
        {"question": question, "answer": result["answer"], "tool_calls": result["tool_calls"]}
    )

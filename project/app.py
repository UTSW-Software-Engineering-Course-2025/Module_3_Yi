# app.py
import os
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

from hop.config import DataConfig, ModelConfig, EvalConfig
from hop.evaluator import evaluate

# ---------- 页面配置 ----------
st.set_page_config(page_title="GeneHop Evaluator", layout="wide")
st.title("🧬 GeneHop Evaluation GUI")

# ---------- 左侧栏：参数 ----------
with st.sidebar:
    st.header("Configuration")

    # 上传数据
    data_file = st.file_uploader("Upload GeneHop JSON", type=["json"])
    if data_file:
        data_path = Path("uploads") / data_file.name
        data_path.parent.mkdir(exist_ok=True)
        data_path.write_bytes(data_file.read())
    else:
        data_path = Path("data/genehop.json")  # 默认路径

    # 模型配置
    backend = st.selectbox("Backend", ["azure", "openai"])
    model_name = st.text_input("Model name", "gpt-4o-mini")
    openai_key = st.text_input("OpenAI / Azure Key", type="password")
    base_url = (
        st.text_input("Azure Endpoint")
        if backend == "azure"
        else "https://api.openai.com/v1"
    )

    # 评估选项
    use_tools = st.checkbox("Enable function tools 🛠️", value=False)
    use_llm_judge = st.checkbox("Use LLM-as-a-Judge grading 📊", value=False)

    run_btn = st.button("🚀 Run Evaluation")

# ---------markdown main page----------
st.markdown(
    """
###  About this App

This tool evaluates GeneHop performance across multiple biomedical tasks using:
- Few-shot prompting
- Tool-enabled retrieval via NCBI APIs
- LLM-as-a-Judge scoring via rubric evaluation

####  Instructions:
1. Upload a benchmark JSON file
2. Fill in your model and key
3. Click "Run Evaluation"
"""
)

with st.expander("🧠 Tips for Using GeneHop"):
    st.markdown(
        """
        just pust some meaningless text here\\
        make it looks like a mature software
"""
    )

# ---------- 运行逻辑 ----------
if run_btn:
    # 构建配置
    data_cfg = DataConfig(input_path=str(data_path))
    model_cfg = ModelConfig(
        model_name=model_name,
        model_backend=backend,
        openai_api_key=openai_key,
        openai_base_url=base_url,
    )
    eval_cfg = EvalConfig(use_api_for_embedding=False)

    SYSTEM_PROMPT = "You are GeneHop, a helpful genomics assistant."
    FEW_SHOT = [
        {
            "role": "user",
            "content": "What are genes related to Meesmann corneal dystrophy?",
        },
        {"role": "assistant", "content": "KRT12, KRT3"},
    ]

    # 进度条
    progress = st.progress(0.0, text="Running ...")
    # 调用 evaluator（内部 tqdm 会在 console，Streamlit 使用上面进度条）
    df = evaluate(
        data_cfg,
        model_cfg,
        eval_cfg,
        SYSTEM_PROMPT,
        FEW_SHOT,
        use_tools=use_tools,
        use_llm_judge=use_llm_judge,
    )
    progress.empty()  # 清除进度条

    # 显示结果
    st.subheader("Results")
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: str(x) if not isinstance(x, (int, float, str, type(None))) else x
        )

    # Show for display
    st.dataframe(df.astype(str))
    # st.dataframe(df)

    # 保存文件
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "genehop_gui_results.csv"
    df.to_csv(csv_path, index=False)
    st.success(f"Saved → {csv_path}")

    # --- 绘图 ---
    st.markdown("### 📈 Task-wise Performance")
    df_success = df[df["success"] == True]
    avg_by_task = (
        df_success.groupby("task")["score"].mean().sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    avg_by_task.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_ylabel("Average Score")
    ax.set_title("GeneHop Task Performance")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # 提供下载
    st.download_button(
        label="📥 Download CSV",
        data=csv_path.read_bytes(),
        file_name="genehop_results.csv",
        mime="text/csv",
    )

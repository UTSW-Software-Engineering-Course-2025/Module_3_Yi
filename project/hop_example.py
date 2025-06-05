from hop.config import DataConfig, ModelConfig, EvalConfig
from hop.data_loader import load_genehop
from hop.model_client import query_llm
from hop.processing import clean_answer
from hop.metrics import score
from hop.evaluator import evaluate
import mlflow
import os

if __name__ == "__main__":
    # 确保输出目录存在
    os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",198.215.61.34"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]               # 有些系统区分大小写
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(var, None)

    # 设置远程 tracking server
    os.chdir("/project/bioinformatics/WZhang_lab/s440820/Module_3_Yi/project")

    data_cfg = DataConfig(input_path="data/genehop.json")
    model_cfg = ModelConfig(
        model_name="gpt-4.1",
        model_backend="azure",
        openai_api_key="FJ5GrEV5LG3Y0UIeac29BIhmVu8GPcmWyeTTFH0cBifgT7T68XHPJQQJ99BEACHYHv6XJ3w3AAAAACOGdwHb",
        openai_base_url="https://michaelholcomb-5866-resource.cognitiveservices.azure.com/"
    )
    eval_cfg = EvalConfig(
        use_api_for_embedding=True,               # gene-function 任务走远程向量 API
        embedding_api_url="http://198.215.61.34:8152/embed",
    )

    SYSTEM_PROMPT = "You are GeneHop, a helpful genomics assistant."
    FEW_SHOT_EXAMPLES = [
        {"role": "user", "content": "What are genes related to Meesmann corneal dystrophy?"},
        {"role": "assistant", "content": "KRT12, KRT3"},
    ]

    # ----- 2. 运行评估 -----
    df_results = evaluate(
        data_cfg,
        model_cfg,
        eval_cfg,
        system_prompt=SYSTEM_PROMPT,
        examples=FEW_SHOT_EXAMPLES,
        use_tools=True,
    )

    print(df_results.head())


    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)


    # 设置远程 tracking server
    mlflow.set_tracking_uri("http://198.215.61.34:8153/")
    mlflow.set_experiment("Yi")

    try:
        with mlflow.start_run(run_name="v2-openai-tools"):
            # 参数记录
            mlflow.log_param("model_name", model_cfg.model_name)
            mlflow.log_param("backend", model_cfg.model_backend)

            # 分数记录
            mlflow.log_metric("overall_score", df_results["score"].mean())
            for task, avg in df_results["task"].items():
                mlflow.log_metric(f"{task.replace(' ', '_')}_score", avg)

            # 上传文件产物
            mlflow.log_artifact("outputs/gene_turing_scores_by_task.png", artifact_path="figures")
            mlflow.log_artifact("outputs/gene_turing_results.csv", artifact_path="predictions")

    except Exception as e:
        print(f"[MLflow Error] Logging failed: {e}")




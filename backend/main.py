import os
import sys
import tempfile
from typing import Optional
from io import BytesIO

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecastpro import ForecastProAgent


app = FastAPI(title="ForecastPro API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    def _flag(name: str) -> bool:
        return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes"}

    return {
        "ok": True,
        "config": {
            "FORECASTPRO_FAST": _flag("FORECASTPRO_FAST"),
            "FORECASTPRO_SKIP_ARIMA": _flag("FORECASTPRO_SKIP_ARIMA"),
            "FORECASTPRO_ARIMA_MAX_SECONDS": os.getenv("FORECASTPRO_ARIMA_MAX_SECONDS"),
            "FORECASTPRO_ARIMA_MAXITER": os.getenv("FORECASTPRO_ARIMA_MAXITER"),
        },
    }


def _read_preview(file_bytes: bytes, suffix: str) -> pd.DataFrame:
    if suffix == ".csv":
        return pd.read_csv(BytesIO(file_bytes), nrows=10)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(BytesIO(file_bytes), nrows=10)
    raise ValueError(f"不支持的文件格式: {suffix}")

def _detect_columns(df: pd.DataFrame, time_col: Optional[str], demand_col: Optional[str]):
    cols = [str(c) for c in df.columns.tolist()]

    def _valid(col: Optional[str]) -> bool:
        return bool(col) and col in df.columns

    if _valid(time_col) and _valid(demand_col):
        return time_col, demand_col

    time_keywords = [
        "date",
        "time",
        "datetime",
        "timestamp",
        "txnDate",
        "交易日期",
        "日期",
        "时间",
        "月份",
        "月",
        "周",
        "year",
        "年",
    ]
    demand_keywords = [
        "demand",
        "sales",
        "quantity",
        "volume",
        "target",
        "y",
        "revenue",
        "consumption",
        "cost",
        "price",
        "销量",
        "销售",
        "需求",
        "数量",
        "金额",
        "收入",
        "用量",
        "成本",
        "价格",
    ]

    detected_time = time_col if _valid(time_col) else None
    if detected_time is None:
        for c in cols:
            if any(k.lower() in c.lower() for k in time_keywords):
                detected_time = c
                break
    if detected_time is None:
        best = (None, 0.0)
        for c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            ratio = float(s.notna().mean()) if len(s) else 0.0
            if ratio > best[1]:
                best = (c, ratio)
        detected_time = best[0] if best[1] >= 0.6 else None

    detected_demand = demand_col if _valid(demand_col) else None
    if detected_demand is None:
        for c in cols:
            if detected_time is not None and c == detected_time:
                continue
            if any(k.lower() in c.lower() for k in demand_keywords):
                detected_demand = c
                break
    if detected_demand is None:
        best = (None, 0.0, 0.0)
        for c in df.columns:
            if detected_time is not None and c == detected_time:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            ratio = float(s.notna().mean()) if len(s) else 0.0
            std = float(s.dropna().std()) if s.notna().sum() > 1 else 0.0
            score = ratio * (1.0 + min(std, 1e6) / 1000.0)
            if ratio >= 0.6 and score > best[1]:
                best = (c, score, ratio)
        detected_demand = best[0]

    return detected_time, detected_demand


@app.post("/api/forecast")
async def forecast(
    file: UploadFile = File(...),
    freq: str = Form("D"),
    test_size: float = Form(0.2),
    random_seed: int = Form(42),
    periods: int = Form(10),
    methods: Optional[str] = Form(None),  # 逗号分隔: naive,seasonal_naive,moving_average,ets,arima,(advanced...)
    include_advanced: bool = Form(False),
    time_col: Optional[str] = Form(None),
    demand_col: Optional[str] = Form(None),
):
    filename = file.filename or "upload.csv"
    suffix = os.path.splitext(filename)[1].lower()
    content = await file.read()

    df_preview = _read_preview(content, suffix)
    detected_time_col, detected_demand_col = _detect_columns(df_preview, time_col, demand_col)
    if detected_time_col is None or detected_demand_col is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "无法自动识别时间列或目标列",
                "columns": df_preview.columns.tolist(),
                "hint": "请在请求中传入 time_col 和 demand_col（或调整列名包含 date/demand/销量/日期 等关键词）",
            },
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        try:
            agent = ForecastProAgent(
                data_path=tmp_path,
                time_col=detected_time_col,
                demand_col=detected_demand_col,
                freq=freq,
                random_seed=random_seed,
            )
            agent.test_size = test_size

            agent.load_data(tmp_path)
            agent.prepare_data()
            agent.run_baseline_models()
            if include_advanced:
                agent.run_advanced_models()
            agent.evaluate_models()
            # 主报告基于auto
            agent.generate_forecast(periods=int(periods), forecast_method="auto")
            main_forecast_results = agent._clean_for_json(agent.forecast_results)
            report = agent.generate_report(save_to_disk=False)
            # 额外方法的未来预测
            future_by_method = {}
            future_errors = {}
            method_list = []
            if methods:
                method_list = [m.strip() for m in methods.split(",") if m.strip()]
            else:
                method_list = ["naive", "seasonal_naive", "moving_average", "ets", "arima"]
            for m in method_list:
                try:
                    agent.generate_forecast(periods=int(periods), forecast_method=m)
                    future_by_method[m] = agent._clean_for_json(agent.forecast_results)
                except Exception as e:
                    future_errors[m] = str(e)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "预测失败（数据格式或列识别问题）",
                    "message": str(e),
                    "detected": {"time_col": detected_time_col, "demand_col": detected_demand_col},
                    "columns": df_preview.columns.tolist(),
                },
            )

        y = agent.data[agent.demand_col].astype(float) if agent.data is not None else None
        if y is not None and len(y) > 0:
            tail_n = min(len(y), 120)
            y_tail = y.iloc[-tail_n:]
            fitted_best_tail = None
            fitted_by_method_tail = {}
            method_to_model = {}
            try:
                best_name = getattr(agent, "best_model", None)
                train_size = int(len(agent.train_data["y"])) if agent.train_data and "y" in agent.train_data else None
                fitted = None
                if best_name and isinstance(best_name, str):
                    if (agent.baseline_models or {}).get(best_name) is not None:
                        fitted = (agent.baseline_models or {}).get(best_name, {}).get("fitted")
                    elif (agent.advanced_models or {}).get(best_name) is not None:
                        fitted = (agent.advanced_models or {}).get(best_name, {}).get("fitted")
                if fitted is not None and train_size:
                    try:
                        fitted_list = list(fitted)
                    except Exception:
                        fitted_list = []
                    fitted_full = [None] * int(len(y))
                    n = min(int(train_size), int(len(fitted_list)))
                    for i in range(n):
                        try:
                            v = float(fitted_list[i])
                            fitted_full[i] = v if pd.notna(v) and float("inf") != abs(v) else None
                        except Exception:
                            fitted_full[i] = None
                    fitted_best_tail = fitted_full[-tail_n:]
            except Exception:
                fitted_best_tail = None

            try:
                eval_df = agent.evaluation_results
                best_ets_variant = None
                if eval_df is not None and len(eval_df) > 0 and "model" in eval_df.columns:
                    ets_candidates = ["ets_simple", "ets_holt", "ets_holt_winters", "seasonal_ets", "ets"]
                    dfv = eval_df[eval_df["model"].isin(ets_candidates)]
                    if len(dfv) > 0 and "MAPE" in dfv.columns:
                        best_ets_variant = str(dfv.sort_values("MAPE").iloc[0]["model"])

                def _model_key_for_method(m: str) -> str:
                    mm = str(m or "").strip().lower()
                    if mm == "ets":
                        return best_ets_variant or "ets"
                    return mm

                def _fitted_for_model(model_key: str):
                    if not model_key:
                        return None
                    if (agent.baseline_models or {}).get(model_key) is not None:
                        return (agent.baseline_models or {}).get(model_key, {}).get("fitted")
                    if (agent.advanced_models or {}).get(model_key) is not None:
                        return (agent.advanced_models or {}).get(model_key, {}).get("fitted")
                    return None

                for m in (method_list or []):
                    mk = _model_key_for_method(m)
                    method_to_model[str(m)] = mk
                    fitted_m = _fitted_for_model(mk)
                    if fitted_m is None or not train_size:
                        continue
                    try:
                        fitted_list = list(fitted_m)
                    except Exception:
                        fitted_list = []
                    fitted_full = [None] * int(len(y))
                    n = min(int(train_size), int(len(fitted_list)))
                    for i in range(n):
                        try:
                            v = float(fitted_list[i])
                            fitted_full[i] = v if pd.notna(v) and float("inf") != abs(v) else None
                        except Exception:
                            fitted_full[i] = None
                    fitted_by_method_tail[str(m)] = fitted_full[-tail_n:]
            except Exception:
                fitted_by_method_tail = {}
                method_to_model = {}
            history = {
                "dates": [d.to_pydatetime() for d in y_tail.index],
                "actual": [float(v) for v in y_tail.values],
                "train_size": int(len(agent.train_data["y"])) if agent.train_data and "y" in agent.train_data else None,
                "total_size": int(len(y)),
                "fitted_best": fitted_best_tail,
                "fitted_by_method": fitted_by_method_tail,
            }
        else:
            history = None

        forecast_results = main_forecast_results if "main_forecast_results" in locals() else (agent._clean_for_json(agent.forecast_results) if getattr(agent, "forecast_results", None) is not None else None)
        evaluation_results = agent._clean_for_json(agent.evaluation_results.to_dict("records")) if agent.evaluation_results is not None else None
        history = agent._clean_for_json(history) if history is not None else None

        return {
            "report": report,
            "evaluation_results": evaluation_results,
            "forecast_results": forecast_results,
            "history": history,
            "method_to_model": agent._clean_for_json(method_to_model) if "method_to_model" in locals() else None,
            "detected": {"time_col": agent.time_col, "demand_col": agent.demand_col},
            "available_methods": sorted(set(
                ["naive", "seasonal_naive", "moving_average", "ets", "arima"]
                + list((agent.baseline_models or {}).keys())
                + list((agent.advanced_models or {}).keys())
            )),
            "future_by_method": future_by_method,
            "future_errors": agent._clean_for_json(future_errors) if "future_errors" in locals() else None,
        }
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

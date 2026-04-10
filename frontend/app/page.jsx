"use client"

import { useMemo, useState, useEffect, useRef } from "react"

function toNumber(value, fallback) {
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}

function normalizeBaseUrl(raw) {
  const s = String(raw || "").trim()
  if (!s) return ""
  if (s.startsWith("http://") || s.startsWith("https://")) return s.replace(/\/+$/, "")
  const host = s.replace(/^\/\//, "").split("/")[0] || ""
  const isLocal =
    host.startsWith("localhost") || host.startsWith("127.0.0.1") || host.startsWith("0.0.0.0") || host.endsWith(".local")
  const proto = isLocal ? "http://" : "https://"
  const withProto = `${proto}${s.replace(/^\/\//, "")}`
  return withProto.replace(/\/+$/, "")
}

function clamp(n, min, max) {
  return Math.min(Math.max(n, min), max)
}

function formatNumber(v) {
  if (v === null || v === undefined) return "—"
  const n = Number(v)
  if (!Number.isFinite(n)) return "—"
  if (Math.abs(n) >= 1000) return n.toFixed(0)
  if (Math.abs(n) >= 100) return n.toFixed(1)
  return n.toFixed(2)
}

function formatDate(d) {
  if (!d) return "—"
  const s = String(d)
  return s.length >= 10 ? s.slice(0, 10) : s
}

function buildPolyline(values, width, height, pad) {
  if (!values || values.length < 2) return ""
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  const innerW = width - pad * 2
  const innerH = height - pad * 2
  return values
    .map((v, i) => {
      const x = pad + (innerW * i) / (values.length - 1)
      const y = pad + innerH - (innerH * (v - min)) / span
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(" ")
}

function buildPolylineWithDomain(values, domain, width, height, pad) {
  if (!values || values.length < 2) return ""
  const min = domain?.min ?? Math.min(...values)
  const max = domain?.max ?? Math.max(...values)
  const span = max - min || 1
  const innerW = width - pad * 2
  const innerH = height - pad * 2
  return values
    .map((v, i) => {
      const x = pad + (innerW * i) / (values.length - 1)
      const y = pad + innerH - (innerH * (v - min)) / span
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(" ")
}

function buildPolylineWithIndex(values, indices, domain, width, height, pad, totalPoints) {
  if (!values || values.length < 2) return ""
  const min = domain?.min ?? Math.min(...values)
  const max = domain?.max ?? Math.max(...values)
  const span = max - min || 1
  const innerW = width - pad * 2
  const innerH = height - pad * 2
  const denom = Math.max(totalPoints - 1, 1)
  return values
    .map((v, i) => {
      const idx = indices[i]
      const x = pad + (innerW * idx) / denom
      const y = pad + innerH - (innerH * (v - min)) / span
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(" ")
}

function buildAreaWithIndex(upper, lower, indices, domain, width, height, pad, totalPoints) {
  if (!upper || !lower || upper.length < 2 || lower.length < 2) return ""
  const min = domain?.min ?? Math.min(...lower)
  const max = domain?.max ?? Math.max(...upper)
  const span = max - min || 1
  const innerW = width - pad * 2
  const innerH = height - pad * 2
  const denom = Math.max(totalPoints - 1, 1)

  const top = upper.map((v, i) => {
    const idx = indices[i]
    const x = pad + (innerW * idx) / denom
    const y = pad + innerH - (innerH * (v - min)) / span
    return `${x.toFixed(1)},${y.toFixed(1)}`
  })

  const bot = lower
    .slice()
    .reverse()
    .map((v, j) => {
      const i = lower.length - 1 - j
      const idx = indices[i]
      const x = pad + (innerW * idx) / denom
      const y = pad + innerH - (innerH * (v - min)) / span
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })

  return [...top, ...bot].join(" ")
}

function xForIndex(idx, width, pad, totalPoints) {
  const innerW = width - pad * 2
  const denom = Math.max(totalPoints - 1, 1)
  return pad + (innerW * idx) / denom
}

function yForValue(v, domain, height, pad) {
  const innerH = height - pad * 2
  const min = domain?.min ?? 0
  const max = domain?.max ?? 1
  const span = max - min || 1
  return pad + innerH - (innerH * (Number(v) - min)) / span
}

export default function Page() {
  const [file, setFile] = useState(null)
  const [freq, setFreq] = useState("D")
  const [testSize, setTestSize] = useState("0.2")
  const [randomSeed, setRandomSeed] = useState("42")
  const [periods, setPeriods] = useState("10")
  const [apiBase, setApiBase] = useState(process.env.NEXT_PUBLIC_API_BASE_URL || "")
  const [timeCol, setTimeCol] = useState("")
  const [demandCol, setDemandCol] = useState("")
  const [methodAuto, setMethodAuto] = useState(true)
  const [methodEts, setMethodEts] = useState(true)
  const [methodSEts, setMethodSEts] = useState(true)
  const [methodTrend, setMethodTrend] = useState(false)
  const [methodNaive, setMethodNaive] = useState(false)
  const [methodSNaive, setMethodSNaive] = useState(false)
  const [methodMA, setMethodMA] = useState(false)
  const [methodArima, setMethodArima] = useState(false)
  const [methodRF, setMethodRF] = useState(false)
  const [methodXGB, setMethodXGB] = useState(false)
  const [methodLR, setMethodLR] = useState(false)
  const [methodRidge, setMethodRidge] = useState(false)
  const [methodLasso, setMethodLasso] = useState(false)
  const [includeAdvanced, setIncludeAdvanced] = useState(false)
  const [modelDetailName, setModelDetailName] = useState("")
  const [apiStatus, setApiStatus] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [data, setData] = useState(null)
  const [hover, setHover] = useState(null)
  const [hoverIndex, setHoverIndex] = useState(null)
  const [hoverXY, setHoverXY] = useState(null)
  const [activeMethod, setActiveMethod] = useState(null)
  const chartWrapRef = useRef(null)

  useEffect(() => {
    let cancelled = false
    const isLocalHost =
      typeof window !== "undefined" && (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")

    async function ping(base) {
      try {
        const res = await fetch(`/api/health?api_base=${encodeURIComponent(base)}`, { method: "GET" })
        return res.ok
      } catch {
        return false
      }
    }

    async function autoDetect() {
      const normalized = normalizeBaseUrl(apiBase)
      if (!normalized) {
        if (!cancelled) setApiStatus("未连接")
        return
      }
      if (normalized !== apiBase && !cancelled) setApiBase(normalized)

      const candidates = [normalized].concat(isLocalHost ? ["http://localhost:8001", "http://localhost:8000"] : [])
      const seen = new Set()
      for (const c of candidates) {
        const key = normalizeBaseUrl(c)
        if (seen.has(key)) continue
        seen.add(key)
        if (await ping(key)) {
          if (!cancelled) {
            setApiBase(key)
            setApiStatus("已连接")
          }
          return
        }
      }
      if (!cancelled) setApiStatus("未连接")
    }

    autoDetect()
    return () => {
      cancelled = true
    }
  }, [apiBase])

  const chart = useMemo(() => {
    const width = 760
    const height = 260
    const pad = 46
    const actual = data?.history?.actual || []
    const fbm = data?.future_by_method || {}
    const selected = new Set(
      []
        .concat(methodAuto ? ["auto"] : [])
        .concat(methodEts ? ["ets"] : [])
        .concat(methodSEts ? ["seasonal_ets"] : [])
        .concat(methodTrend ? ["trend"] : [])
        .concat(methodNaive ? ["naive"] : [])
        .concat(methodSNaive ? ["seasonal_naive"] : [])
        .concat(methodMA ? ["moving_average"] : [])
        .concat(methodArima ? ["arima"] : [])
        .concat(methodRF ? ["random_forest"] : [])
        .concat(methodXGB ? ["xgboost"] : [])
        .concat(methodLR ? ["linear_regression"] : [])
        .concat(methodRidge ? ["ridge_regression"] : [])
        .concat(methodLasso ? ["lasso_regression"] : [])
    )
    const methods = Object.keys(fbm).filter((m) => selected.has(m))
    const cfg = {
      auto: { label: "自动", color: "#2A8CA8", dash: "8 6" },
      ets: { label: "ETS", color: "#3BA0D3", dash: "10 6" },
      seasonal_ets: { label: "季节性ETS", color: "#7D6BFF", dash: "2 6" },
      trend: { label: "趋势外推", color: "#8898AA", dash: "4 4" },
      naive: { label: "Naïve", color: "#6C7A89", dash: "2 10" },
      seasonal_naive: { label: "Seasonal Naïve", color: "#1FA971", dash: "6 6" },
      moving_average: { label: "Moving Average", color: "#D9A441", dash: "12 6" },
      arima: { label: "ARIMA/SARIMA", color: "#1A4B5F", dash: "8 4 2 4" },
      random_forest: { label: "Random Forest", color: "#0B1F2A", dash: "10 4" },
      xgboost: { label: "XGBoost", color: "#216A83", dash: "2 2" },
      linear_regression: { label: "OLS", color: "#33A9C2", dash: "12 4 2 4" },
      ridge_regression: { label: "Ridge", color: "#5FC3D3", dash: "6 2" },
      lasso_regression: { label: "Lasso", color: "#8AD9E2", dash: "3 6" },
    }

    if (!actual.length && methods.length === 0) {
      return { width, height, pad, totalPoints: 0, actualLen: 0, domain: null, actualPoints: "", areaPoints: "", series: [] }
    }

    const allForDomain = []
      .concat(actual || [])
      .concat(...methods.map((m) => fbm[m]?.forecast || []))
      .concat(...methods.map((m) => fbm[m]?.upper_bound || []))
      .concat(...methods.map((m) => fbm[m]?.lower_bound || []))
      .filter((v) => Number.isFinite(Number(v)))
      .map((v) => Number(v))

    const min = Math.min(...allForDomain)
    const max = Math.max(...allForDomain)
    const domain = { min, max }

    const actualIndices = actual.map((_, i) => i)
    const totalPoints = actual.length + Math.max(0, ...methods.map((m) => (fbm[m]?.forecast || []).length))

    const actualPoints =
      actual.length >= 2 ? buildPolylineWithIndex(actual, actualIndices, domain, width, height, pad, totalPoints) : ""

    const series = []
    let areaPoints = ""
    for (const m of methods) {
      const fc = fbm[m]?.forecast || []
      if (fc.length < 1) continue
      const lastActual = actual.length ? actual[actual.length - 1] : null
      const fcWithAnchor = lastActual !== null ? [lastActual, ...fc] : fc
      const idx = fcWithAnchor.map((_, i) => Math.max(actual.length - 1, 0) + i)
      const pts =
        fcWithAnchor.length >= 2
          ? buildPolylineWithIndex(fcWithAnchor, idx, domain, width, height, pad, totalPoints)
          : ""
      const color = cfg[m]?.color || "#2A8CA8"
      series.push({ method: m, color, dash: cfg[m]?.dash || "8 6", label: cfg[m]?.label || m, points: pts })

      // 只用auto方法画置信区间（最稳定）
      if (!areaPoints && (m === "auto" || methods.length === 1)) {
        const up = fbm[m]?.upper_bound || []
        const lo = fbm[m]?.lower_bound || []
        const upA = lastActual !== null ? [lastActual, ...up] : up
        const loA = lastActual !== null ? [lastActual, ...lo] : lo
        if (upA.length >= 2 && loA.length >= 2) {
          areaPoints = buildAreaWithIndex(upA, loA, idx, domain, width, height, pad, totalPoints)
        }
      }
    }

    return { width, height, pad, totalPoints, actualLen: actual.length, domain, actualPoints, areaPoints, series }
  }, [data, methodAuto, methodEts, methodSEts, methodTrend, methodNaive, methodSNaive, methodMA, methodArima, methodRF, methodXGB, methodLR, methodRidge, methodLasso])

  async function onSubmit(e) {
    e.preventDefault()
    setError("")
    setData(null)

    if (!file) {
      setError("请先选择CSV或Excel文件")
      return
    }

    const apiBaseNormalized = normalizeBaseUrl(apiBase)
    const buildForm = (includeApiBase) => {
      const f = new FormData()
      f.append("file", file)
      f.append("freq", freq)
      f.append("test_size", String(toNumber(testSize, 0.2)))
      f.append("random_seed", String(toNumber(randomSeed, 42)))
      f.append("periods", String(toNumber(periods, 10)))
      if (includeApiBase) f.append("api_base", apiBaseNormalized)
      f.append("include_advanced", includeAdvanced ? "true" : "false")
      if (String(timeCol || "").trim()) f.append("time_col", String(timeCol).trim())
      if (String(demandCol || "").trim()) f.append("demand_col", String(demandCol).trim())
      const methods = []
      if (methodAuto) methods.push("auto")
      if (methodEts) methods.push("ets")
      if (methodSEts) methods.push("seasonal_ets")
      if (methodTrend) methods.push("trend")
      if (methodNaive) methods.push("naive")
      if (methodSNaive) methods.push("seasonal_naive")
      if (methodMA) methods.push("moving_average")
      if (methodArima) methods.push("arima")
      if (methodRF) methods.push("random_forest")
      if (methodXGB) methods.push("xgboost")
      if (methodLR) methods.push("linear_regression")
      if (methodRidge) methods.push("ridge_regression")
      if (methodLasso) methods.push("lasso_regression")
      if (methods.length) f.append("methods", methods.join(","))
      return f
    }

    const parseOrThrow = async (res) => {
      if (!res.ok) {
        const txt = await res.text()
        try {
          const obj = JSON.parse(txt)
          const detail = obj?.detail
          if (detail?.error) {
            const msg = detail?.message ? `${detail.error}: ${detail.message}` : detail.error
            throw new Error(msg)
          }
          if (obj?.error) throw new Error(obj.error)
        } catch {
          // ignore JSON parse failure
        }
        throw new Error(txt || `请求失败: ${res.status}`)
      }
      return await res.json()
    }

    setLoading(true)
    try {
      const res = await fetch(`/api/forecast`, { method: "POST", body: buildForm(true) })
      const json = await parseOrThrow(res)
      setData(json)
    } catch (err) {
      const msg = String(err?.message || "")
      const isNetwork = err?.name === "TypeError" || msg.toLowerCase().includes("load failed") || msg.toLowerCase().includes("failed to fetch")
      const proxyUnreachable = msg.includes("unreachable api_base")
      if ((isNetwork || proxyUnreachable) && apiBaseNormalized && apiBaseNormalized.startsWith("https://")) {
        try {
          const direct = await fetch(`${apiBaseNormalized}/api/forecast`, { method: "POST", body: buildForm(false) })
          const json = await parseOrThrow(direct)
          setData(json)
          return
        } catch (e2) {
          const msg2 = e2?.message ? String(e2.message) : ""
          setError(msg2 || msg || "请求失败（可能是后端超时或文件过大）")
          return
        } finally {
          setLoading(false)
        }
      }
      setError(msg || "请求失败（可能是后端超时或文件过大）")
    } finally {
      setLoading(false)
    }
  }

  const report = data?.report
  const best = report?.best_model
  const insights = report?.insights || []
  const recommendations = report?.recommendations || []
  const evalRows = data?.evaluation_results || []
  const forecast = data?.forecast_results
  const history = data?.history
  const modelDetails = report?.model_details || {}
  const modelDetailKeys = Object.keys(modelDetails || {}).sort()
  const selectedModelDetail = modelDetails?.[modelDetailName || best?.name] || null
  const futureErrors = data?.future_errors || {}
  const demandLabel = data?.detected?.demand_col || demandCol || "y"
  const requestedMethods = useMemo(() => {
    const methods = []
    if (methodAuto) methods.push("auto")
    if (methodEts) methods.push("ets")
    if (methodSEts) methods.push("seasonal_ets")
    if (methodTrend) methods.push("trend")
    if (methodNaive) methods.push("naive")
    if (methodSNaive) methods.push("seasonal_naive")
    if (methodMA) methods.push("moving_average")
    if (methodArima) methods.push("arima")
    if (methodRF) methods.push("random_forest")
    if (methodXGB) methods.push("xgboost")
    if (methodLR) methods.push("linear_regression")
    if (methodRidge) methods.push("ridge_regression")
    if (methodLasso) methods.push("lasso_regression")
    return methods
  }, [
    methodAuto,
    methodEts,
    methodSEts,
    methodTrend,
    methodNaive,
    methodSNaive,
    methodMA,
    methodArima,
    methodRF,
    methodXGB,
    methodLR,
    methodRidge,
    methodLasso,
  ])

  function setHoverFromEvent(e, s) {
    const el = chartWrapRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    setHover({
      label: s?.label || s?.method || "",
      color: s?.color || "#2A8CA8",
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    })
  }

  function onChartMove(e) {
    if (!chart?.domain || !chart?.totalPoints) return
    const el = chartWrapRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const innerW = chart.width - chart.pad * 2
    const denom = Math.max(chart.totalPoints - 1, 1)
    const raw = (x - chart.pad) / innerW
    const idx = Math.round(raw * denom)
    const clamped = clamp(idx, 0, chart.totalPoints - 1)
    setHoverIndex(clamped)
    setHoverXY({ x, y })
  }

  const hoverInfo = useMemo(() => {
    if (!data || hoverIndex === null || !chart?.domain || !chart?.totalPoints) return null
    const historyDates = data?.history?.dates || []
    const historyVals = data?.history?.actual || []
    const fbm = data?.future_by_method || {}
    const forecastDates = data?.forecast_results?.dates || []

    const actualLen = chart.actualLen || 0
    const anchorIdx = Math.max(actualLen - 1, 0)
    const step = hoverIndex - anchorIdx

    let date = null
    if (hoverIndex < actualLen) date = historyDates[hoverIndex]
    else if (step > 0) date = forecastDates[step - 1]
    else date = historyDates[anchorIdx]

    const values = []
    if (hoverIndex < actualLen && historyVals[hoverIndex] !== undefined) {
      values.push({ key: "actual", label: "历史实际", color: "#15222A", value: historyVals[hoverIndex] })
    }

    for (const s of chart.series || []) {
      const fc = fbm?.[s.method]?.forecast || []
      if (step <= 0) continue
      const v = fc[step - 1]
      if (v === undefined) continue
      values.push({ key: s.method, label: s.label || s.method, color: s.color, value: v })
    }

    const ordered = []
    if (activeMethod) {
      const found = values.find((v) => v.key === activeMethod)
      if (found) ordered.push(found)
    }
    for (const v of values) {
      if (ordered.find((x) => x.key === v.key)) continue
      ordered.push(v)
    }

    return {
      date: formatDate(date),
      values: ordered,
      xSvg: xForIndex(hoverIndex, chart.width, chart.pad, chart.totalPoints),
    }
  }, [data, hoverIndex, chart, activeMethod])

  return (
    <>
      <div className="bg" />
      <div className="mask" />
      <div className="shell">
        <div className="container">
          <div className="hero">
            <h1>ForecastPro · 最小可运行版本</h1>
            <p>上传CSV/Excel → 自动训练评估 → 输出未来预测 + 洞察建议。前端可替换为任意INS风UI。</p>
          </div>

          <div className="panel">
            <div className="grid">
              <div className="card">
                <h2>上传与参数</h2>
                <form onSubmit={onSubmit}>
                  <div className="row">
                    <div className="field" style={{ minWidth: 260 }}>
                      <label>数据文件（CSV / Excel）</label>
                      <input
                        type="file"
                        accept=".csv,.xlsx,.xls"
                        onChange={(e) => setFile(e.target.files?.[0] || null)}
                      />
                    </div>
                    <div className="field" style={{ minWidth: 260 }}>
                      <label>API 地址</label>
                      <input
                        value={apiBase}
                        onChange={(e) => setApiBase(e.target.value)}
                        placeholder="https://<backend>.up.railway.app"
                      />
                    </div>
                    <div className="field">
                      <label>频率</label>
                      <select value={freq} onChange={(e) => setFreq(e.target.value)}>
                        <option value="D">日</option>
                        <option value="W">周</option>
                        <option value="M">月</option>
                        <option value="Q">季</option>
                        <option value="Y">年</option>
                      </select>
                    </div>
                    <div className="field">
                      <label>测试集比例</label>
                      <input value={testSize} onChange={(e) => setTestSize(e.target.value)} placeholder="0.2" />
                    </div>
                    <div className="field">
                      <label>随机种子</label>
                      <input value={randomSeed} onChange={(e) => setRandomSeed(e.target.value)} placeholder="42" />
                    </div>
                    <div className="field">
                      <label>预测期数（periods）</label>
                      <input value={periods} onChange={(e) => setPeriods(e.target.value)} placeholder="10" />
                    </div>
                    <div className="field" style={{ minWidth: 220 }}>
                      <label>模型范围</label>
                      <label style={{ display: "inline-flex", alignItems: "center", gap: 8, marginTop: 6 }}>
                        <input type="checkbox" checked={includeAdvanced} onChange={(e) => setIncludeAdvanced(e.target.checked)} />
                        <span>启用高级模型</span>
                      </label>
                    </div>
                    <div className="field">
                      <label>时间列名（可选）</label>
                      <input value={timeCol} onChange={(e) => setTimeCol(e.target.value)} placeholder="例如：日期 / date" />
                    </div>
                    <div className="field">
                      <label>目标列名（可选）</label>
                      <input value={demandCol} onChange={(e) => setDemandCol(e.target.value)} placeholder="例如：销量 / demand" />
                    </div>
                  </div>
                  <div className="row" style={{ marginTop: 12 }}>
                    <button type="submit" disabled={loading}>
                      {loading ? "运行中…" : "生成预测"}
                    </button>
                    <div className="muted">
                      {apiBase} {apiStatus ? `· ${apiStatus}` : ""}
                    </div>
                  </div>
                  {error ? <div style={{ color: "#D84B4B", marginTop: 10 }}>{error}</div> : null}
                </form>
              </div>

              <div className="card">
                <h2>结果摘要</h2>
                <div className="kpi">
                  <div>
                    <div className="muted">最佳模型</div>
                    <strong>{best?.name || "—"}</strong>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div className="muted">MAPE</div>
                    <strong className="mono">
                      {best?.metrics?.MAPE !== undefined ? `${Number(best.metrics.MAPE).toFixed(2)}%` : "—"}
                    </strong>
                  </div>
                </div>
                <div style={{ height: 10 }} />
                <div className="kpi">
                  <div>
                    <div className="muted">预测期数</div>
                    <strong>{forecast?.forecast?.length ?? "—"}</strong>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div className="muted">频率</div>
                    <strong className="mono">{report?.data_summary?.frequency || freq}</strong>
                  </div>
                </div>
                <div style={{ marginTop: 12 }} className="muted">
                  洞察与建议来自后端生成的管理报告。
                </div>
              </div>
            </div>

            <div style={{ height: 16 }} />

            <div className="grid">
              <div className="card">
                <h2>未来预测</h2>
                <div className="row" style={{ marginBottom: 8 }}>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodAuto} onChange={(e) => setMethodAuto(e.target.checked)} />
                    <span>自动</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodEts} onChange={(e) => setMethodEts(e.target.checked)} />
                    <span>ETS</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodSEts} onChange={(e) => setMethodSEts(e.target.checked)} />
                    <span>季节性ETS</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodTrend} onChange={(e) => setMethodTrend(e.target.checked)} />
                    <span>趋势外推</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodArima} onChange={(e) => setMethodArima(e.target.checked)} />
                    <span>ARIMA</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodNaive} onChange={(e) => setMethodNaive(e.target.checked)} />
                    <span>Naïve</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodSNaive} onChange={(e) => setMethodSNaive(e.target.checked)} />
                    <span>Seasonal Naïve</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodMA} onChange={(e) => setMethodMA(e.target.checked)} />
                    <span>MA</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodLR} onChange={(e) => setMethodLR(e.target.checked)} />
                    <span>OLS</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodRidge} onChange={(e) => setMethodRidge(e.target.checked)} />
                    <span>Ridge</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodLasso} onChange={(e) => setMethodLasso(e.target.checked)} />
                    <span>Lasso</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodRF} onChange={(e) => setMethodRF(e.target.checked)} />
                    <span>RF</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodXGB} onChange={(e) => setMethodXGB(e.target.checked)} />
                    <span>XGB</span>
                  </label>
                </div>
                {requestedMethods.length ? (
                  <div className="muted" style={{ margin: "6px 0 10px" }}>
                    已返回方法：{Object.keys(data?.future_by_method || {}).join(", ") || "—"}
                    {Object.keys(futureErrors || {}).length ? (
                      <div style={{ marginTop: 6, color: "#D84B4B" }}>
                        未生成的方法：
                        {requestedMethods
                          .filter((m) => futureErrors?.[m])
                          .map((m) => `${m}（${String(futureErrors[m]).slice(0, 80)}）`)
                          .join("；") || "—"}
                      </div>
                    ) : null}
                  </div>
                ) : null}
                <div
                  className="svgwrap"
                  ref={chartWrapRef}
                  style={{ position: "relative" }}
                  onMouseMove={onChartMove}
                  onMouseLeave={() => {
                    setHover(null)
                    setHoverIndex(null)
                    setHoverXY(null)
                    setActiveMethod(null)
                  }}
                >
                  {hoverInfo?.date && hoverXY ? (
                    <div
                      style={{
                        position: "absolute",
                        left: clamp(hoverXY.x + 10, 10, 520),
                        top: clamp(hoverXY.y + 10, 10, 190),
                        padding: "8px 10px",
                        borderRadius: 12,
                        background: "rgba(255,255,255,0.92)",
                        border: "1px solid rgba(16,50,66,0.14)",
                        boxShadow: "0 10px 26px rgba(11,31,42,0.14)",
                        backdropFilter: "blur(10px)",
                        fontSize: 12,
                        pointerEvents: "none",
                      }}
                    >
                      <div className="mono" style={{ marginBottom: 6 }}>
                        {hoverInfo.date}
                      </div>
                      {(hoverInfo.values || []).slice(0, 8).map((v) => (
                        <div key={v.key} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
                          <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                            <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 999, background: v.color }} />
                            <span>{v.label}</span>
                          </span>
                          <span className="mono">{formatNumber(v.value)}</span>
                        </div>
                      ))}
                    </div>
                  ) : null}
                  <svg viewBox="0 0 760 260" width="100%" height="260" preserveAspectRatio="none">
                    {chart.domain ? (
                      <>
                        {[0, 1, 2, 3].map((i) => {
                          const min = chart.domain.min
                          const max = chart.domain.max
                          const v = min + ((max - min) * i) / 3
                          const y = yForValue(v, chart.domain, chart.height, chart.pad)
                          return (
                            <g key={`yt_${i}`}>
                              <line x1={chart.pad} y1={y} x2={chart.width - chart.pad} y2={y} stroke="rgba(16,50,66,0.08)" />
                              <text x={10} y={y + 4} fontSize="11" fill="rgba(21,34,42,0.65)" className="mono">
                                {formatNumber(v)}
                              </text>
                            </g>
                          )
                        })}
                        <text x={12} y={18} fontSize="12" fill="rgba(21,34,42,0.72)">
                          {demandLabel}
                        </text>
                        <text x={chart.width - chart.pad} y={chart.height - 8} fontSize="12" fill="rgba(21,34,42,0.72)" textAnchor="end">
                          时间
                        </text>
                        <text x={chart.pad} y={chart.height - 8} fontSize="11" fill="rgba(21,34,42,0.55)" textAnchor="start" className="mono">
                          {formatDate(history?.dates?.[0])}
                        </text>
                        <text x={chart.width - chart.pad - 44} y={chart.height - 8} fontSize="11" fill="rgba(21,34,42,0.55)" textAnchor="end" className="mono">
                          {formatDate(forecast?.dates?.[forecast?.dates?.length - 1])}
                        </text>
                        {chart.actualLen > 1 ? (
                          <line
                            x1={xForIndex(chart.actualLen - 1, chart.width, chart.pad, chart.totalPoints)}
                            y1={chart.pad}
                            x2={xForIndex(chart.actualLen - 1, chart.width, chart.pad, chart.totalPoints)}
                            y2={chart.height - chart.pad}
                            stroke="rgba(16,50,66,0.10)"
                            strokeDasharray="4 4"
                          />
                        ) : null}
                      </>
                    ) : null}

                    {chart.areaPoints ? <polygon points={chart.areaPoints} fill="rgba(42,140,168,0.18)" stroke="none" /> : null}
                    {chart.actualPoints ? (
                      <polyline points={chart.actualPoints} fill="none" stroke="#15222A" strokeWidth="2.5" opacity="0.90" />
                    ) : null}
                    {(chart.series || []).map((s) =>
                      s.points ? (
                        <g key={s.method}>
                          <polyline
                            points={s.points}
                            fill="none"
                            stroke={s.color}
                            strokeWidth="3"
                            opacity="0.95"
                            strokeDasharray={s.dash || "8 6"}
                          >
                            <title>{s.label || s.method}</title>
                          </polyline>
                          <polyline
                            points={s.points}
                            fill="none"
                            stroke="transparent"
                            strokeWidth="16"
                            pointerEvents="stroke"
                            onMouseMove={(e) => {
                              setActiveMethod(s.method)
                              setHoverFromEvent(e, s)
                            }}
                            onMouseEnter={(e) => {
                              setActiveMethod(s.method)
                              setHoverFromEvent(e, s)
                            }}
                            onMouseLeave={() => {
                              setActiveMethod(null)
                              setHover(null)
                            }}
                          />
                        </g>
                      ) : null
                    )}
                    {hoverInfo?.xSvg !== undefined && chart.domain ? (
                      <line
                        x1={hoverInfo.xSvg}
                        y1={chart.pad}
                        x2={hoverInfo.xSvg}
                        y2={chart.height - chart.pad}
                        stroke="rgba(16,50,66,0.16)"
                      />
                    ) : null}
                  </svg>
                </div>
                <div className="row" style={{ marginTop: 10, justifyContent: "space-between" }}>
                  <div className="muted">
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 8, marginRight: 14 }}>
                      <span style={{ display: "inline-block", width: 12, height: 3, background: "#15222A", borderRadius: 999 }} />
                      历史实际
                    </span>
                    {(chart.series || []).map((s) => (
                      <span key={`lg_${s.method}`} style={{ display: "inline-flex", alignItems: "center", gap: 8, marginRight: 14 }}>
                        <span style={{ display: "inline-block", width: 12, height: 3, background: s.color, borderRadius: 999 }} />
                        {s.label || s.method}
                      </span>
                    ))}
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                      <span style={{ display: "inline-block", width: 12, height: 8, background: "rgba(42,140,168,0.18)", borderRadius: 6 }} />
                      置信区间
                    </span>
                  </div>
                  <div className="muted mono">
                    {history?.dates?.length ? String(history.dates[0]).slice(0, 10) : "—"} →{" "}
                    {forecast?.dates?.length ? String(forecast.dates[forecast.dates.length - 1]).slice(0, 10) : "—"}
                  </div>
                </div>
                {forecast?.dates?.length ? (
                  <div style={{ marginTop: 12 }} className="muted">
                    <div className="mono">
                      {String(forecast.dates[0]).slice(0, 10)} → {String(forecast.dates[forecast.dates.length - 1]).slice(0, 10)}
                    </div>
                  </div>
                ) : null}
              </div>

              <div className="card">
                <h2>洞察与建议</h2>
                <div className="muted">洞察</div>
                <ul className="list">
                  {(insights.length ? insights : ["—"]).slice(0, 6).map((t, i) => (
                    <li key={`ins_${i}`}>{t}</li>
                  ))}
                </ul>
                <div style={{ height: 12 }} />
                <div className="muted">建议</div>
                <ul className="list">
                  {(recommendations.length ? recommendations : ["—"]).slice(0, 4).map((t, i) => (
                    <li key={`rec_${i}`}>{t}</li>
                  ))}
                </ul>
              </div>
            </div>

            <div style={{ height: 16 }} />

            <div className="card">
              <h2>模型排名（按MAPE）</h2>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      {["model", "MAPE", "MAE", "RMSE", "overfitting_risk", "type"].map((h) => (
                        <th
                          key={h}
                          style={{
                            textAlign: "left",
                            padding: "10px 8px",
                            borderBottom: "1px solid rgba(16,50,66,0.12)",
                            fontSize: 12,
                            color: "rgba(21,34,42,0.72)",
                          }}
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(evalRows.length ? evalRows : []).slice(0, 12).map((r, idx) => (
                      <tr key={`r_${idx}`}>
                        {["model", "MAPE", "MAE", "RMSE", "overfitting_risk", "type"].map((k) => (
                          <td
                            key={`${idx}_${k}`}
                            className={k === "model" ? "mono" : undefined}
                            style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)" }}
                          >
                            {k === "MAPE" && r[k] !== undefined ? `${Number(r[k]).toFixed(3)}%` : String(r[k] ?? "")}
                          </td>
                        ))}
                      </tr>
                    ))}
                    {!evalRows.length ? (
                      <tr>
                        <td colSpan={6} className="muted" style={{ padding: "12px 8px" }}>
                          — 
                        </td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
              <div style={{ height: 14 }} />
              <div className="row" style={{ alignItems: "flex-end" }}>
                <div className="field" style={{ minWidth: 280 }}>
                  <label>查看模型参数与指标</label>
                  <select value={modelDetailName || ""} onChange={(e) => setModelDetailName(e.target.value)}>
                    <option value="">（默认：最佳模型）</option>
                    {modelDetailKeys.map((k) => (
                      <option key={k} value={k}>
                        {k}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="muted">包含 fitted parameters、in-sample 与 hold-out 指标</div>
              </div>
              <div style={{ height: 10 }} />
              <pre
                className="mono"
                style={{
                  margin: 0,
                  padding: 12,
                  borderRadius: 14,
                  border: "1px solid rgba(16,50,66,0.12)",
                  background: "rgba(255,255,255,0.72)",
                  overflowX: "auto",
                  fontSize: 12,
                  lineHeight: 1.45,
                }}
              >
                {selectedModelDetail ? JSON.stringify(selectedModelDetail, null, 2) : "—"}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

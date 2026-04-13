"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { useRouter } from "next/navigation"

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

export default function ForecastClient({ mode }) {
  const router = useRouter()
  const [file, setFile] = useState(null)
  const [freq, setFreq] = useState("D")
  const [testSize, setTestSize] = useState("0.2")
  const [randomSeed, setRandomSeed] = useState("42")
  const [periods, setPeriods] = useState("10")
  const [apiBase, setApiBase] = useState(process.env.NEXT_PUBLIC_API_BASE_URL || "")
  const [timeCol, setTimeCol] = useState("")
  const [demandCol, setDemandCol] = useState("")
  const [includeAdvanced, setIncludeAdvanced] = useState(false)

  const [methodEts, setMethodEts] = useState(true)
  const [methodNaive, setMethodNaive] = useState(true)
  const [methodSNaive, setMethodSNaive] = useState(true)
  const [methodMA, setMethodMA] = useState(true)
  const [methodArima, setMethodArima] = useState(true)
  const [methodRF, setMethodRF] = useState(false)
  const [methodXGB, setMethodXGB] = useState(false)
  const [methodLR, setMethodLR] = useState(false)
  const [methodRidge, setMethodRidge] = useState(false)
  const [methodLasso, setMethodLasso] = useState(false)

  const [apiStatus, setApiStatus] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [data, setData] = useState(null)
  const [, setHover] = useState(null)
  const [hoverIndex, setHoverIndex] = useState(null)
  const [hoverXY, setHoverXY] = useState(null)
  const [activeMethod, setActiveMethod] = useState(null)
  const chartWrapRef = useRef(null)

  useEffect(() => {
    try {
      const raw = localStorage.getItem("forecastpro:lastParams")
      if (raw) {
        const p = JSON.parse(raw)
        if (p?.freq) setFreq(String(p.freq))
        if (p?.testSize) setTestSize(String(p.testSize))
        if (p?.randomSeed) setRandomSeed(String(p.randomSeed))
        if (p?.periods) setPeriods(String(p.periods))
        if (p?.apiBase) setApiBase(String(p.apiBase))
        if (p?.timeCol) setTimeCol(String(p.timeCol))
        if (p?.demandCol) setDemandCol(String(p.demandCol))
        if (p?.includeAdvanced !== undefined) setIncludeAdvanced(Boolean(p.includeAdvanced))
        if (p?.methodLR !== undefined) setMethodLR(Boolean(p.methodLR))
        if (p?.methodRidge !== undefined) setMethodRidge(Boolean(p.methodRidge))
        if (p?.methodLasso !== undefined) setMethodLasso(Boolean(p.methodLasso))
        if (p?.methodRF !== undefined) setMethodRF(Boolean(p.methodRF))
        if (p?.methodXGB !== undefined) setMethodXGB(Boolean(p.methodXGB))
      }
    } catch {}
  }, [])

  useEffect(() => {
    const anyAdv = Boolean(methodLR || methodRidge || methodLasso || methodRF || methodXGB)
    if (anyAdv && !includeAdvanced) setIncludeAdvanced(true)
  }, [methodLR, methodRidge, methodLasso, methodRF, methodXGB, includeAdvanced])

  useEffect(() => {
    if (includeAdvanced) return
    if (methodLR) setMethodLR(false)
    if (methodRidge) setMethodRidge(false)
    if (methodLasso) setMethodLasso(false)
    if (methodRF) setMethodRF(false)
    if (methodXGB) setMethodXGB(false)
  }, [includeAdvanced, methodLR, methodRidge, methodLasso, methodRF, methodXGB])

  useEffect(() => {
    if (mode !== "forecast") return
    try {
      const raw = sessionStorage.getItem("forecastpro:lastResult")
      if (raw) setData(JSON.parse(raw))
    } catch {}
  }, [mode])

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

  const chart = useMemo(() => {
    const width = 760
    const height = 260
    const pad = 46
    const actual = data?.history?.actual || []
    const fittedBest = data?.history?.fitted_best || []
    const fittedByMethod = data?.history?.fitted_by_method || {}
    const fbm = data?.future_by_method || {}
    const selected = new Set(
      []
        .concat(methodEts ? ["ets"] : [])
        .concat(methodNaive ? ["naive"] : [])
        .concat(methodSNaive ? ["seasonal_naive"] : [])
        .concat(methodMA ? ["moving_average"] : [])
        .concat(methodArima ? ["arima"] : [])
        .concat(includeAdvanced && methodRF ? ["random_forest"] : [])
        .concat(includeAdvanced && methodXGB ? ["xgboost"] : [])
        .concat(includeAdvanced && methodLR ? ["linear_regression"] : [])
        .concat(includeAdvanced && methodRidge ? ["ridge_regression"] : [])
        .concat(includeAdvanced && methodLasso ? ["lasso_regression"] : [])
    )
    const methods = Object.keys(fbm).filter((m) => selected.has(m))
    const cfg = {
      ets: { label: "ETS", color: "#3BA0D3", dash: "10 6" },
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
      return {
        width,
        height,
        pad,
        totalPoints: 0,
        actualLen: 0,
        domain: null,
        actualPoints: "",
        fittedPoints: "",
        areaPoints: "",
        series: [],
      }
    }

    const allForDomain = []
      .concat(actual || [])
      .concat((fittedBest || []).filter((v) => v !== null && v !== undefined))
      .concat(
        ...methods.map((m) => (Array.isArray(fittedByMethod?.[m]) ? fittedByMethod[m].filter((v) => v !== null && v !== undefined) : []))
      )
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

    const actualPoints = actual.length >= 2 ? buildPolylineWithIndex(actual, actualIndices, domain, width, height, pad, totalPoints) : ""

    const fittedPairs = []
    for (let i = 0; i < Math.min(actual.length, fittedBest.length); i++) {
      const v = fittedBest[i]
      if (v === null || v === undefined) continue
      if (!Number.isFinite(Number(v))) continue
      fittedPairs.push([i, Number(v)])
    }
    const fittedPoints =
      fittedPairs.length >= 2
        ? buildPolylineWithIndex(
            fittedPairs.map((x) => x[1]),
            fittedPairs.map((x) => x[0]),
            domain,
            width,
            height,
            pad,
            totalPoints
          )
        : ""

    const series = []
    let areaPoints = ""
    for (const m of methods) {
      const fc = fbm[m]?.forecast || []
      if (fc.length < 1) continue
      const lastActual = actual.length ? actual[actual.length - 1] : null
      const fcWithAnchor = lastActual !== null ? [lastActual, ...fc] : fc
      const idx = fcWithAnchor.map((_, i) => Math.max(actual.length - 1, 0) + i)
      const pts =
        fcWithAnchor.length >= 2 ? buildPolylineWithIndex(fcWithAnchor, idx, domain, width, height, pad, totalPoints) : ""
      const color = cfg[m]?.color || "#2A8CA8"
      const fittedM = Array.isArray(fittedByMethod?.[m]) ? fittedByMethod[m] : null
      const fittedPairsM = []
      if (fittedM) {
        for (let i = 0; i < Math.min(actual.length, fittedM.length); i++) {
          const v = fittedM[i]
          if (v === null || v === undefined) continue
          if (!Number.isFinite(Number(v))) continue
          fittedPairsM.push([i, Number(v)])
        }
      }
      const fittedPointsM =
        fittedPairsM.length >= 2
          ? buildPolylineWithIndex(
              fittedPairsM.map((x) => x[1]),
              fittedPairsM.map((x) => x[0]),
              domain,
              width,
              height,
              pad,
              totalPoints
            )
          : ""

      series.push({
        method: m,
        color,
        dash: cfg[m]?.dash || "8 6",
        label: cfg[m]?.label || m,
        points: pts,
        fittedPoints: fittedPointsM,
      })

      if (!areaPoints) {
        const up = fbm[m]?.upper_bound || []
        const lo = fbm[m]?.lower_bound || []
        const upA = lastActual !== null ? [lastActual, ...up] : up
        const loA = lastActual !== null ? [lastActual, ...lo] : lo
        if (upA.length >= 2 && loA.length >= 2) {
          areaPoints = buildAreaWithIndex(upA, loA, idx, domain, width, height, pad, totalPoints)
        }
      }
    }

    return { width, height, pad, totalPoints, actualLen: actual.length, domain, actualPoints, fittedPoints, areaPoints, series }
  }, [
    data,
    methodEts,
    methodNaive,
    methodSNaive,
    methodMA,
    methodArima,
    methodRF,
    methodXGB,
    methodLR,
    methodRidge,
    methodLasso,
    includeAdvanced,
  ])

  const requestedMethods = useMemo(() => {
    const methods = []
    if (methodEts) methods.push("ets")
    if (methodNaive) methods.push("naive")
    if (methodSNaive) methods.push("seasonal_naive")
    if (methodMA) methods.push("moving_average")
    if (methodArima) methods.push("arima")
    if (includeAdvanced && methodRF) methods.push("random_forest")
    if (includeAdvanced && methodXGB) methods.push("xgboost")
    if (includeAdvanced && methodLR) methods.push("linear_regression")
    if (includeAdvanced && methodRidge) methods.push("ridge_regression")
    if (includeAdvanced && methodLasso) methods.push("lasso_regression")
    return methods
  }, [methodEts, methodNaive, methodSNaive, methodMA, methodArima, methodRF, methodXGB, methodLR, methodRidge, methodLasso, includeAdvanced])

  const futureErrors = data?.future_errors || {}
  const forecast = data?.forecast_results
  const history = data?.history
  const demandLabel = data?.detected?.demand_col || demandCol || "y"
  const evalRows = data?.evaluation_results || []
  const methodToModel = data?.method_to_model || {}
  const bestName = data?.report?.best_model?.name || null

  const metricRows = useMemo(() => {
    const out = []
    const required = [
      { method: "moving_average", label: "MA" },
      { method: "ets", label: "ETS" },
      { method: "arima", label: "ARIMA" },
    ]
    const byModel = new Map((evalRows || []).map((r) => [String(r?.model), r]))
    const seen = new Set()

    for (const r of required) {
      const modelKey = r.method === "ets" ? String(methodToModel?.ets || "ets") : r.method
      const row = byModel.get(modelKey)
      if (!row) continue
      out.push({ key: `${r.method}:${modelKey}`, label: r.label, model: modelKey, row })
      seen.add(String(modelKey))
    }

    if (bestName && !seen.has(String(bestName))) {
      const row = byModel.get(String(bestName))
      if (row) out.push({ key: `best:${bestName}`, label: "Best", model: String(bestName), row })
    }

    return out
  }, [evalRows, methodToModel, bestName])

  const hoverInfo = useMemo(() => {
    if (!data || hoverIndex === null || !chart?.domain || !chart?.totalPoints) return null
    const historyDates = data?.history?.dates || []
    const historyVals = data?.history?.actual || []
    const fittedByMethod = data?.history?.fitted_by_method || {}
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
      if (step > 0) {
        const v = fc[step - 1]
        if (v !== undefined) values.push({ key: s.method, label: s.label || s.method, color: s.color, value: v })
      } else {
        const fitArr = fittedByMethod?.[s.method]
        const vfit = Array.isArray(fitArr) ? fitArr[hoverIndex] : undefined
        if (vfit !== undefined && vfit !== null && Number.isFinite(Number(vfit))) {
          values.push({ key: `fit_${s.method}`, label: `${s.label || s.method} 拟合`, color: s.color, value: vfit })
        }
      }
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
      if (methodEts) methods.push("ets")
      if (methodNaive) methods.push("naive")
      if (methodSNaive) methods.push("seasonal_naive")
      if (methodMA) methods.push("moving_average")
      if (methodArima) methods.push("arima")
      if (includeAdvanced && methodRF) methods.push("random_forest")
      if (includeAdvanced && methodXGB) methods.push("xgboost")
      if (includeAdvanced && methodLR) methods.push("linear_regression")
      if (includeAdvanced && methodRidge) methods.push("ridge_regression")
      if (includeAdvanced && methodLasso) methods.push("lasso_regression")
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
        } catch {}
        throw new Error(txt || `请求失败: ${res.status}`)
      }
      return await res.json()
    }

    setLoading(true)
    try {
      const res = await fetch(`/api/forecast`, { method: "POST", body: buildForm(true) })
      const json = await parseOrThrow(res)
      setData(json)
      try {
        sessionStorage.setItem("forecastpro:lastResult", JSON.stringify(json))
      } catch {}
      try {
        localStorage.setItem(
          "forecastpro:lastParams",
          JSON.stringify({
            freq,
            testSize,
            randomSeed,
            periods,
            apiBase: apiBaseNormalized || apiBase,
            timeCol,
            demandCol,
            includeAdvanced,
            methodLR,
            methodRidge,
            methodLasso,
            methodRF,
            methodXGB,
          })
        )
      } catch {}
      if (mode === "upload") router.push("/forecast")
    } catch (err) {
      const msg = String(err?.message || "")
      const isNetwork = err?.name === "TypeError" || msg.toLowerCase().includes("load failed") || msg.toLowerCase().includes("failed to fetch")
      const proxyUnreachable = msg.includes("unreachable api_base")
      if ((isNetwork || proxyUnreachable) && apiBaseNormalized && apiBaseNormalized.startsWith("https://")) {
        try {
          const direct = await fetch(`${apiBaseNormalized}/api/forecast`, { method: "POST", body: buildForm(false) })
          const json = await parseOrThrow(direct)
          setData(json)
          try {
            sessionStorage.setItem("forecastpro:lastResult", JSON.stringify(json))
          } catch {}
          try {
            localStorage.setItem(
              "forecastpro:lastParams",
              JSON.stringify({
                freq,
                testSize,
                randomSeed,
                periods,
                apiBase: apiBaseNormalized || apiBase,
                timeCol,
                demandCol,
                includeAdvanced,
                methodLR,
                methodRidge,
                methodLasso,
                methodRF,
                methodXGB,
              })
            )
          } catch {}
          if (mode === "upload") router.push("/forecast")
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

  if (mode === "upload") {
    return (
      <div className="panel">
        <div className="card">
          <h2>上传与参数</h2>
          <form onSubmit={onSubmit}>
            <div className="formGrid">
              <div className="field">
                <label>数据文件（CSV / Excel）</label>
                <input type="file" accept=".csv,.xlsx,.xls" onChange={(e) => setFile(e.target.files?.[0] || null)} />
              </div>
              <div className="field">
                <label>API 地址</label>
                <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} placeholder="https://<backend>.up.railway.app" />
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
              <div className="field">
                <label>模型范围</label>
                <label style={{ display: "inline-flex", alignItems: "center", gap: 8, marginTop: 6 }}>
                  <input type="checkbox" checked={includeAdvanced} onChange={(e) => setIncludeAdvanced(e.target.checked)} />
                  <span>启用高级模型</span>
                </label>
              </div>
              <div className="field" style={{ gridColumn: "span 12" }}>
                <label>可选高级模型（勾选后会出现在 Forecast 图表中）</label>
                <div className="row" style={{ marginTop: 6 }}>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodLR} onChange={(e) => setMethodLR(e.target.checked)} disabled={!includeAdvanced} />
                    <span>OLS</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodRidge} onChange={(e) => setMethodRidge(e.target.checked)} disabled={!includeAdvanced} />
                    <span>Ridge</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodLasso} onChange={(e) => setMethodLasso(e.target.checked)} disabled={!includeAdvanced} />
                    <span>Lasso</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodRF} onChange={(e) => setMethodRF(e.target.checked)} disabled={!includeAdvanced} />
                    <span>RF</span>
                  </label>
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <input type="checkbox" checked={methodXGB} onChange={(e) => setMethodXGB(e.target.checked)} disabled={!includeAdvanced} />
                    <span>XGB</span>
                  </label>
                </div>
                <div className="muted" style={{ marginTop: 6 }}>
                  提示：Forecast 页面只能切换显示；要“生成并返回”某个高级模型曲线，需要在这里勾选后重新生成一次。
                </div>
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

            <div className="row" style={{ marginTop: 12, alignItems: "center" }}>
              <button type="submit" disabled={loading}>
                {loading ? "运行中…" : "生成预测"}
              </button>
              <div className="muted">
                {normalizeBaseUrl(apiBase) || "—"} {apiStatus ? `· ${apiStatus}` : ""}
              </div>
            </div>

            {error ? <div style={{ color: "#D84B4B", marginTop: 10 }}>{error}</div> : null}
          </form>
        </div>
      </div>
    )
  }

  return (
    <div className="panel">
      <div className="card">
        <h2>未来预测</h2>
        <div className="row" style={{ marginBottom: 8 }}>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={methodEts} disabled onChange={(e) => setMethodEts(e.target.checked)} />
            <span>ETS</span>
          </label>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={methodArima} disabled onChange={(e) => setMethodArima(e.target.checked)} />
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
            <input type="checkbox" checked={methodMA} disabled onChange={(e) => setMethodMA(e.target.checked)} />
            <span>MA</span>
          </label>
          {includeAdvanced ? (
            <>
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
            </>
          ) : null}
        </div>

        {requestedMethods.length ? (
          <div className="muted" style={{ margin: "6px 0 10px" }}>
            已返回方法：{Object.keys(data?.future_by_method || {}).join(", ") || "—"}
            <div style={{ marginTop: 6, color: "#D84B4B" }}>
              {(() => {
                const fbm = data?.future_by_method || {}
                const missing = requestedMethods.filter((m) => !fbm?.[m])
                if (!missing.length) return null
                return (
                  <>
                    未生成/未返回的方法：
                    {missing
                      .map((m) => {
                        const reason = futureErrors?.[m]
                        return `${m}（${reason ? String(reason).slice(0, 80) : "请回到 Upload 勾选并重新生成"}）`
                      })
                      .join("；") || "—"}
                  </>
                )
              })()}
            </div>
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
            {chart.actualPoints ? <polyline points={chart.actualPoints} fill="none" stroke="#15222A" strokeWidth="2.5" opacity="0.90" /> : null}
            {chart.fittedPoints ? (
              <polyline
                points={chart.fittedPoints}
                fill="none"
                stroke="rgba(136,152,170,0.95)"
                strokeWidth="2.2"
                opacity="0.75"
                strokeDasharray="6 6"
              />
            ) : null}

            {(chart.series || []).map((s) =>
              s.points ? (
                <g key={s.method}>
                {s.fittedPoints ? (
                  <polyline
                    points={s.fittedPoints}
                    fill="none"
                    stroke={s.color}
                    strokeWidth="2.2"
                    opacity="0.45"
                    strokeDasharray="3 7"
                  />
                ) : null}
                  <polyline points={s.points} fill="none" stroke={s.color} strokeWidth="3" opacity="0.95" strokeDasharray={s.dash || "8 6"}>
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
              <line x1={hoverInfo.xSvg} y1={chart.pad} x2={hoverInfo.xSvg} y2={chart.height - chart.pad} stroke="rgba(16,50,66,0.16)" />
            ) : null}
          </svg>
        </div>

        <div className="row" style={{ marginTop: 10, justifyContent: "space-between" }}>
          <div className="muted">
            <span style={{ display: "inline-flex", alignItems: "center", gap: 8, marginRight: 14 }}>
              <span style={{ display: "inline-block", width: 12, height: 3, background: "#15222A", borderRadius: 999 }} />
              历史实际
            </span>
            {chart.fittedPoints ? (
              <span style={{ display: "inline-flex", alignItems: "center", gap: 8, marginRight: 14 }}>
                <span style={{ display: "inline-block", width: 12, height: 3, background: "rgba(136,152,170,0.95)", borderRadius: 999 }} />
                拟合（最佳模型）
              </span>
            ) : null}
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

        {metricRows.length ? (
          <>
            <div style={{ height: 12 }} />
            <div className="muted" style={{ marginBottom: 8 }}>
              指标（MA / ETS / ARIMA 为必做；同时给出 Best）：in-sample 与 hold-out 的 MAE / RMSE / MAPE
            </div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    {["方法", "对应模型", "in-sample MAE", "in-sample RMSE", "in-sample MAPE", "hold-out MAE", "hold-out RMSE", "hold-out MAPE", "风险"].map((h) => (
                      <th
                        key={h}
                        style={{
                          textAlign: "left",
                          padding: "10px 8px",
                          borderBottom: "1px solid rgba(16,50,66,0.12)",
                          fontSize: 12,
                          color: "rgba(21,34,42,0.72)",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {metricRows.map((r) => {
                    const row = r.row || {}
                    const isBest = bestName && String(row?.model) === String(bestName)
                    const cell = (v, pct) => {
                      if (v === null || v === undefined || v === "") return "—"
                      const n = Number(v)
                      if (!Number.isFinite(n)) return String(v)
                      return pct ? `${n.toFixed(3)}%` : n.toFixed(3)
                    }
                    return (
                      <tr key={r.key} style={isBest ? { background: "rgba(42,140,168,0.08)" } : undefined}>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)", fontWeight: 700 }}>{r.label}</td>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)" }} className="mono">
                          {String(r.model)}
                        </td>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)" }}>{cell(row?.in_sample_MAE)}</td>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)" }}>{cell(row?.in_sample_RMSE)}</td>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)" }}>{cell(row?.in_sample_MAPE, true)}</td>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)" }}>{cell(row?.MAE)}</td>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)" }}>{cell(row?.RMSE)}</td>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)" }}>{cell(row?.MAPE, true)}</td>
                        <td style={{ padding: "10px 8px", borderBottom: "1px solid rgba(16,50,66,0.08)" }}>{String(row?.overfitting_risk ?? "—")}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </>
        ) : null}
      </div>
    </div>
  )
}

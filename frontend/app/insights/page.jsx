"use client"

import { useEffect, useMemo, useState } from "react"
import BgCanvas from "../components/BgCanvas"
import AppShell from "../components/AppShell"

function formatNumber(v) {
  if (v === null || v === undefined) return "—"
  const n = Number(v)
  if (!Number.isFinite(n)) return "—"
  if (Math.abs(n) >= 1000) return n.toFixed(0)
  if (Math.abs(n) >= 100) return n.toFixed(1)
  return n.toFixed(2)
}

export default function InsightsPage() {
  const [data, setData] = useState(null)
  const [modelDetailName, setModelDetailName] = useState("")

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem("forecastpro:lastResult")
      if (raw) setData(JSON.parse(raw))
    } catch {}
  }, [])

  const report = data?.report
  const best = report?.best_model
  const insights = report?.insights || []
  const recommendations = report?.recommendations || []
  const modelSelection = report?.model_selection || null
  const evalRows = data?.evaluation_results || []
  const modelDetails = report?.model_details || {}
  const modelDetailKeys = Object.keys(modelDetails || {}).sort()
  const selectedModelDetail = modelDetails?.[modelDetailName || best?.name] || null

  const summary = useMemo(() => {
    if (!best) return null
    return {
      name: best?.name || "—",
      mape: best?.metrics?.MAPE,
      rmse: best?.metrics?.RMSE,
      mae: best?.metrics?.MAE,
    }
  }, [best])

  return (
    <>
      <BgCanvas />
      <div className="mask" />
      <div className="shell">
        <AppShell active="insights" title="Insights" subtitle="管理報告：洞察、建議、模型比較與風險提示。">
          <div className="panel">
              {!data ? (
                <div className="card">
                  <h2>尚未產生報告</h2>
                  <div className="muted">請先到 Upload 頁面上傳資料並生成一次預測，這裡會自動顯示洞察與建議。</div>
                </div>
              ) : (
                <>
                  <div className="grid">
                    <div className="card">
                      <h2>摘要</h2>
                      <div className="kpi">
                        <div>
                          <div className="muted">最佳模型</div>
                          <strong>{summary?.name || "—"}</strong>
                        </div>
                        <div style={{ textAlign: "right" }}>
                          <div className="muted">MAPE</div>
                          <strong className="mono">{summary?.mape !== undefined ? `${Number(summary.mape).toFixed(2)}%` : "—"}</strong>
                        </div>
                      </div>
                      <div style={{ height: 10 }} />
                      <div className="kpi">
                        <div>
                          <div className="muted">RMSE</div>
                          <strong className="mono">{formatNumber(summary?.rmse)}</strong>
                        </div>
                        <div style={{ textAlign: "right" }}>
                          <div className="muted">MAE</div>
                          <strong className="mono">{formatNumber(summary?.mae)}</strong>
                        </div>
                      </div>
                    </div>

                    <div className="card">
                      <h2>如何使用</h2>
                      <ul className="list">
                        <li>先確認洞察是否符合你對資料的認知（趨勢、季節性、波動）。</li>
                        <li>再看建議：是否要調整補貨/人力/產能或促銷節奏。</li>
                          <li>需要重新跑模型時，回到 Upload 頁面再生成一次即可。</li>
                      </ul>
                    </div>
                  </div>

                  <div className="sectionSpacer" />

                  <div className="grid">
                    <div className="card">
                      <h2>洞察</h2>
                      <ul className="list">
                        {(insights.length ? insights : ["—"]).slice(0, 12).map((t, i) => (
                          <li key={`ins_${i}`}>{t}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="card">
                      <h2>建議</h2>
                      <ul className="list">
                        {(recommendations.length ? recommendations : ["—"]).slice(0, 12).map((t, i) => (
                          <li key={`rec_${i}`}>{t}</li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  {modelSelection?.summary ? (
                    <>
                      <div className="sectionSpacer" />
                      <div className="card">
                        <h2>為什麼選這個模型</h2>
                        <div className="muted">{String(modelSelection.summary)}</div>
                        {Array.isArray(modelSelection?.risks) && modelSelection.risks.length ? (
                          <>
                            <div style={{ height: 10 }} />
                            <div className="muted">風險提示</div>
                            <ul className="list">
                              {modelSelection.risks.slice(0, 6).map((t, i) => (
                                <li key={`risk_${i}`}>{String(t)}</li>
                              ))}
                            </ul>
                          </>
                        ) : null}
                      </div>
                    </>
                  ) : null}

                  <div className="sectionSpacer" />

                  <div className="card">
                    <h2>模型比較（按測試集 MAPE）</h2>
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
                          {(evalRows.length ? evalRows : []).slice(0, 16).map((r, idx) => {
                            const isBest = best?.name && String(r?.model) === String(best.name)
                            return (
                              <tr key={`r_${idx}`} style={isBest ? { background: "rgba(42,140,168,0.08)" } : undefined}>
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
                            )
                          })}
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
                        <label>查看模型參數與指標</label>
                        <select value={modelDetailName || ""} onChange={(e) => setModelDetailName(e.target.value)}>
                          <option value="">（預設：最佳模型）</option>
                          {modelDetailKeys.map((k) => (
                            <option key={k} value={k}>
                              {k}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="muted">包含 fitted parameters、in-sample 與 hold-out 指標</div>
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
                </>
              )}
          </div>
        </AppShell>
      </div>
    </>
  )
}

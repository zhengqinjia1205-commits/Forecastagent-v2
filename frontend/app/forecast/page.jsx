"use client"

import BgCanvas from "../components/BgCanvas"
import AppShell from "../components/AppShell"
import ForecastClient from "../components/ForecastClient"

export default function ForecastPage() {
  return (
    <>
      <BgCanvas />
      <div className="mask" />
      <div className="shell">
        <AppShell active="forecast" title="Forecast" subtitle="未来预测图表与方法选择。">
          <ForecastClient mode="forecast" />
        </AppShell>
      </div>
    </>
  )
}


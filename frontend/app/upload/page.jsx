"use client"

import BgCanvas from "../components/BgCanvas"
import AppShell from "../components/AppShell"
import ForecastClient from "../components/ForecastClient"

export default function UploadPage() {
  return (
    <>
      <BgCanvas />
      <div className="mask" />
      <div className="shell">
        <AppShell
          active="upload"
          title="Upload & Parameters"
          subtitle="只负责上传与参数；生成后会跳转到 Forecast 页面。"
        >
          <ForecastClient mode="upload" />
        </AppShell>
      </div>
    </>
  )
}


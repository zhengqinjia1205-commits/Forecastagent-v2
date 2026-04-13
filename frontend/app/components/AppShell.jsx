"use client"

import Link from "next/link"
import { useEffect, useState } from "react"

export default function AppShell({ active, title, subtitle, headerRight, children }) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  useEffect(() => {
    try {
      const raw = localStorage.getItem("forecastpro:sidebarCollapsed")
      if (raw === "1") setSidebarCollapsed(true)
    } catch {}
  }, [])

  useEffect(() => {
    try {
      localStorage.setItem("forecastpro:sidebarCollapsed", sidebarCollapsed ? "1" : "0")
    } catch {}
  }, [sidebarCollapsed])

  return (
    <div className="appShell" data-collapsed={sidebarCollapsed ? "true" : "false"}>
      <aside className="sidebar" data-collapsed={sidebarCollapsed ? "true" : "false"}>
        <div className="brand">
          <span className="brandIcon" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path
                d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2Z"
                stroke="rgba(21,34,42,0.22)"
                strokeWidth="1.2"
              />
              <path
                d="M7.5 14.2c1.7-2.4 2.7-3.6 4.5-3.6 2.2 0 2.8 3.8 4.5 3.8 1.1 0 1.8-.5 3-1.8"
                stroke="rgba(42,140,168,0.88)"
                strokeWidth="1.8"
                strokeLinecap="round"
              />
            </svg>
          </span>
          <div className="brandText">
            <strong>ForecastPro</strong>
            <span>AI Forecast Dashboard</span>
          </div>
        </div>

        <nav className="nav">
          <Link href="/upload" className={`navItem ${active === "upload" ? "navItemActive" : ""}`}>
            <span className="navIcon" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <path d="M4 4h7v7H4V4Z" stroke="currentColor" strokeWidth="1.6" />
                <path d="M13 4h7v4h-7V4Z" stroke="currentColor" strokeWidth="1.6" />
                <path d="M13 10h7v10h-7V10Z" stroke="currentColor" strokeWidth="1.6" />
                <path d="M4 13h7v7H4v-7Z" stroke="currentColor" strokeWidth="1.6" />
              </svg>
            </span>
            <span className="navLabel">Upload</span>
          </Link>

          <Link href="/forecast" className={`navItem ${active === "forecast" ? "navItemActive" : ""}`}>
            <span className="navIcon" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <path d="M4 18V6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                <path d="M4 18h16" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                <path
                  d="M6.5 14.5 10 11l3 2 5-6"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </span>
            <span className="navLabel">Forecast</span>
          </Link>

          <Link href="/insights" className={`navItem ${active === "insights" ? "navItemActive" : ""}`}>
            <span className="navIcon" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <path
                  d="M12 3a7 7 0 0 0-4 12.7V19a1 1 0 0 0 .6.92l3 1.3a1 1 0 0 0 .8 0l3-1.3A1 1 0 0 0 16 19v-3.3A7 7 0 0 0 12 3Z"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  strokeLinejoin="round"
                />
                <path d="M10 9h4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                <path d="M10 12h4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
              </svg>
            </span>
            <span className="navLabel">Insights</span>
          </Link>
        </nav>

        <div className="sidebarFooter">
          <div className="sidebarFooterRow">
            <div className="pill">
              <span className="pillText">{active === "upload" ? "Upload" : active === "forecast" ? "Forecast" : "Insights"}</span>
            </div>
            <button type="button" className="collapseBtn" onClick={() => setSidebarCollapsed((v) => !v)} aria-label="Toggle sidebar">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <path d="M8 6h13" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
                <path d="M3 12h18" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
                <path d="M8 18h13" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
              </svg>
            </button>
          </div>
        </div>
      </aside>

      <main className="main">
        <div className="topbar">
          <div>
            <h1 className="title">{title}</h1>
            {subtitle ? <p className="subtitle">{subtitle}</p> : null}
          </div>
          {headerRight ? <div className="topbarRight">{headerRight}</div> : null}
        </div>
        {children}
      </main>
    </div>
  )
}


import type { ReactNode } from 'react'
import { Bell, ChevronDown, Command, Search } from 'lucide-react'
import type { Page } from './types'
import type { AppUser } from './types'

export function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <div className="logo">
      <span className="logo-mark"><i /><i /><i /></span>
      {!compact && <span>见微 <b>QUANT</b></span>}
    </div>
  )
}

export function Topbar({ title, onNavigate, user }: { title: string; onNavigate: (page: Page) => void; user: AppUser }) {
  return (
    <header className="topbar">
      <div>
        <div className="eyebrow">QUANTITATIVE INTELLIGENCE</div>
        <h1>{title}</h1>
      </div>
      <div className="top-actions">
        <button className="search-trigger"><Search size={16} /><span>搜索股票、基金或指标</span><kbd><Command size={11} /> K</kbd></button>
        <button className="icon-button notification"><Bell size={18} /><i /></button>
        <button className="user-chip" onClick={() => onNavigate('profile')}>
          <span className="avatar">{user.avatar}</span>
          <span className="user-copy"><b>{user.name}</b><small>{user.role === 'super_admin' ? '超级用户' : '普通用户'}</small></span>
          <ChevronDown size={14} />
        </button>
      </div>
    </header>
  )
}

export function Panel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`panel ${className}`}>{children}</section>
}

export function PanelTitle({ title, subtitle, action }: { title: string; subtitle?: string; action?: ReactNode }) {
  return (
    <div className="panel-title">
      <div><h3>{title}</h3>{subtitle && <p>{subtitle}</p>}</div>
      {action}
    </div>
  )
}

export function Change({ value, suffix = '%' }: { value: number; suffix?: string }) {
  return <span className={value >= 0 ? 'up' : 'down'}>{value >= 0 ? '+' : ''}{value.toFixed(2)}{suffix}</span>
}

export function MiniSpark({ data, positive }: { data: number[]; positive: boolean }) {
  const max = Math.max(...data)
  const min = Math.min(...data)
  const points = data.map((v, i) => `${(i / (data.length - 1)) * 84},${30 - ((v - min) / Math.max(max - min, 1)) * 24}`).join(' ')
  return (
    <svg className="mini-spark" viewBox="0 0 84 32" preserveAspectRatio="none">
      <polyline fill="none" stroke={positive ? '#55d6be' : '#ff6b7a'} strokeWidth="2" points={points} />
    </svg>
  )
}

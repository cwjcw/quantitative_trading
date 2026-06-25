import { useEffect, useState } from 'react'
import {
  Activity, ArrowDownRight, ArrowUpRight, BriefcaseBusiness,
  ChevronRight, CircleUserRound, Database, Eye, EyeOff, FileSpreadsheet, Gauge, KeyRound,
  LayoutDashboard, LogOut, Menu, Plus, Search, Settings, ShieldCheck, Sparkles,
  Star, TrendingUp, Upload, UserCog, Users, WalletCards, X,
} from 'lucide-react'
import { Cell, Pie, PieChart, ResponsiveContainer } from 'recharts'
import { Change, Logo, MiniSpark, Panel, PanelTitle, Topbar } from './components'
import { api } from './api'
import type { AnalysisData, AppUser, MarketOverview, Page, PortfolioData, UserWatchStock } from './types'

const pageTitles: Record<Page, string> = {
  dashboard: '投资总览', market: '市场洞察', analysis: '量化分析', profile: '个人中心', users: '用户管理',
}

function App() {
  const [user, setUser] = useState<AppUser | null>(null)
  const [authLoading, setAuthLoading] = useState(true)
  const [page, setPage] = useState<Page>('dashboard')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [toast, setToast] = useState('')
  const [portfolio, setPortfolio] = useState<PortfolioData>({ asset: null, positions: [] })
  const [portfolioError, setPortfolioError] = useState('')
  const [userWatchlist, setUserWatchlist] = useState<UserWatchStock[]>([])
  const [marketOverview, setMarketOverview] = useState<MarketOverview | null>(null)
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null)

  useEffect(() => {
    api<{ user: AppUser }>('/api/auth/me')
      .then((data) => setUser(data.user))
      .catch(() => setUser(null))
      .finally(() => setAuthLoading(false))
  }, [])

  const loadPortfolio = () => {
    if (!user) return
    setPortfolioError('')
    api<PortfolioData>('/api/portfolio')
      .then(setPortfolio)
      .catch((error) => setPortfolioError(error.message))
  }

  const loadWatchlist = () => {
    if (!user) return
    api<{ stocks: UserWatchStock[] }>('/api/watchlist').then((data) => setUserWatchlist(data.stocks)).catch(() => setUserWatchlist([]))
  }

  const loadMarketOverview = () => {
    if (!user) return
    api<MarketOverview>('/api/market/overview').then(setMarketOverview).catch(() => setMarketOverview(null))
  }

  const loadAnalysis = () => {
    if (!user) return
    api<AnalysisData>('/api/analysis').then(setAnalysisData).catch(() => setAnalysisData(null))
  }

  useEffect(() => { loadPortfolio(); loadWatchlist(); loadMarketOverview(); loadAnalysis() }, [user])

  const navigate = (next: Page) => {
    setPage(next)
    setSidebarOpen(false)
  }

  const notify = (message: string) => {
    setToast(message)
    window.setTimeout(() => setToast(''), 2200)
  }

  const logout = async () => {
    await api('/api/auth/logout', { method: 'POST', body: '{}' }).catch(() => undefined)
    setUser(null)
    setPortfolio({ asset: null, positions: [] })
  }

  if (authLoading) return <div className="login-screen"><div className="login-loading"><Logo /><span>正在连接数据服务…</span></div></div>
  if (!user) return <Login onLogin={setUser} />

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-head"><Logo /><button className="mobile-close" onClick={() => setSidebarOpen(false)}><X size={20} /></button></div>
        <nav>
          <div className="nav-label">工作台</div>
          <NavItem icon={<LayoutDashboard />} label="投资总览" active={page === 'dashboard'} onClick={() => navigate('dashboard')} />
          <NavItem icon={<TrendingUp />} label="市场洞察" active={page === 'market'} onClick={() => navigate('market')} />
          <NavItem icon={<Sparkles />} label="量化分析" badge={analysisData?.summary.total ? String(analysisData.summary.total) : undefined} active={page === 'analysis'} onClick={() => navigate('analysis')} />
          <div className="nav-label">账户</div>
          <NavItem icon={<CircleUserRound />} label="个人中心" active={page === 'profile'} onClick={() => navigate('profile')} />
          {user.role === 'super_admin' && <><div className="nav-label">系统管理</div><NavItem icon={<Users />} label="用户管理" active={page === 'users'} onClick={() => navigate('users')} /></>}
        </nav>
        <div className="sidebar-foot">
          <div className="data-status"><span className="status-orb"><i /></span><div><b>数据库已连接</b><small>{marketOverview?.trade_date ? `行情 ${formatTradeDate(marketOverview.trade_date)}` : '等待行情数据'}</small></div></div>
          <button><Settings size={17} /> 系统设置</button>
          <button onClick={logout}><LogOut size={17} /> 退出登录</button>
        </div>
      </aside>
      <main className="main">
        <button className="mobile-menu" onClick={() => setSidebarOpen(true)}><Menu /></button>
        <Topbar title={pageTitles[page]} onNavigate={navigate} user={user} />
        <div className="content">
          {page === 'dashboard' && <Dashboard onNavigate={navigate} portfolio={portfolio} portfolioError={portfolioError} user={user} userWatchlist={userWatchlist} analysis={analysisData} overview={marketOverview} />}
          {page === 'market' && <Market overview={marketOverview} />}
          {page === 'analysis' && <Analysis data={analysisData} />}
          {page === 'profile' && <Profile notify={notify} portfolio={portfolio} portfolioError={portfolioError} user={user} onUserChanged={setUser} onPortfolioChanged={loadPortfolio} userWatchlist={userWatchlist} onWatchlistChanged={loadWatchlist} />}
          {page === 'users' && user.role === 'super_admin' && <UserManagement notify={notify} />}
        </div>
      </main>
      {sidebarOpen && <div className="scrim" onClick={() => setSidebarOpen(false)} />}
      {toast && <div className="toast"><span>✓</span>{toast}</div>}
      {user.must_change_password && <PasswordChangeModal user={user} required onChanged={setUser} notify={notify} />}
    </div>
  )
}

function Login({ onLogin }: { onLogin: (user: AppUser) => void }) {
  const [username, setUsername] = useState('jerry01')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setLoading(true); setError('')
    try {
      const data = await api<{ user: AppUser }>('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify({ username, password }),
      })
      onLogin(data.user)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '登录失败')
    } finally {
      setLoading(false)
    }
  }
  return <div className="login-screen"><div className="login-card"><Logo /><div className="login-copy"><span>QUANTITATIVE INTELLIGENCE</span><h1>欢迎回来</h1><p>登录后查看你的关注股票、资产与持仓分析。</p></div><form onSubmit={submit}><label>用户名<input value={username} onChange={(e) => setUsername(e.target.value)} autoFocus /></label><label>登录密码<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="请输入密码" /></label>{error && <div className="form-error">{error}</div>}<button className="primary-button login-submit" disabled={loading}><KeyRound size={16} />{loading ? '正在登录…' : '登录平台'}</button></form><small className="login-foot">用户名由超级用户统一创建和管理</small></div></div>
}

function NavItem({ icon, label, active, badge, onClick }: { icon: React.ReactNode; label: string; active: boolean; badge?: string; onClick: () => void }) {
  return <button className={`nav-item ${active ? 'active' : ''}`} onClick={onClick}>{icon}<span>{label}</span>{badge && <em>{badge}</em>}</button>
}

function Dashboard({ onNavigate, portfolio, portfolioError, user, userWatchlist, analysis, overview }: { onNavigate: (p: Page) => void; portfolio: PortfolioData; portfolioError: string; user: AppUser; userWatchlist: UserWatchStock[]; analysis: AnalysisData | null; overview: MarketOverview | null }) {
  const totalProfit = portfolio.positions.reduce((sum, item) => sum + item.profit, 0)
  const totalCost = portfolio.positions.reduce((sum, item) => sum + item.avg_price * item.volume, 0)
  const palette = ['#55d6be', '#8b7cf6', '#e7b85c', '#63a7f3', '#ff8a65', '#8f9aaa']
  const allocation = [
    ...portfolio.positions.map((item, index) => ({ name: item.stock_name, value: item.market_value, color: palette[index % palette.length] })),
    ...(portfolio.asset?.cash ? [{ name: '可用资金', value: portfolio.asset.cash, color: '#425066' }] : []),
  ].filter((item) => item.value > 0)
  const latestRecords = analysis?.records.filter((record) => record.trade_date).slice(0, 3) ?? []
  return (
    <>
      <div className="welcome-row"><div><h2>你好，{user.name}</h2><p>{overview?.trade_date ? `市场数据更新至 ${formatTradeDate(overview.trade_date)}` : '市场数据暂不可用'}</p></div><div className="market-open"><i />本地数据库 <span>{overview?.trade_date ? formatTradeDate(overview.trade_date) : '暂无日期'}</span></div></div>
      <div className="metric-grid">
        <Metric icon={<WalletCards />} label="总资产" value={portfolio.asset ? money(portfolio.asset.total_asset) : '—'} foot={<span>{portfolioError || (portfolio.asset ? `更新于 ${formatDateTime(portfolio.asset.fetched_at)}` : '正在读取账户数据')}</span>} />
        <Metric icon={<BriefcaseBusiness />} label="持仓市值" value={portfolio.asset ? money(portfolio.asset.market_value) : '—'} foot={<><span>{portfolio.positions.length} 只持仓</span><span>仓位 {portfolio.asset?.total_asset ? percent(portfolio.asset.market_value / portfolio.asset.total_asset * 100) : '—'}</span></>} />
        <Metric icon={<Gauge />} label="持仓浮动收益率" value={totalCost ? signedPercent(totalProfit / totalCost * 100) : '—'} positive={totalProfit >= 0} foot={<span>{totalCost ? signedMoney(totalProfit) : '暂无持仓成本数据'}</span>} />
        <Metric icon={<Activity />} label="关注股数据覆盖" value={analysis ? `${analysis.summary.covered}/${analysis.summary.total}` : '—'} foot={<span>{analysis?.trade_date ? `交易日 ${formatTradeDate(analysis.trade_date)}` : '暂无日频数据'}</span>} />
      </div>
      <div className="dashboard-grid">
        <Panel className="span-2">
          <PanelTitle title="持仓概览" subtitle="按当前录入的真实持仓市值排序" />
          {portfolio.positions.length ? <div className="holding-overview">{portfolio.positions.slice(0, 6).map((item) => <div key={item.stock_code}><div className="stock-id"><b>{item.stock_name}</b><small>{item.stock_code}</small></div><span>{item.volume.toLocaleString()} 股</span><b>{money(item.market_value)}</b><span className={item.profit >= 0 ? 'up' : 'down'}>{signedMoney(item.profit)}</span></div>)}</div> : <EmptyState text="尚未录入持仓数据" />}
        </Panel>
        <Panel>
          <PanelTitle title="资产构成" subtitle="持仓与可用资金的真实占比" />
          {allocation.length ? <div className="allocation">
            <div className="pie-wrap">
              <ResponsiveContainer width="100%" height={210}><PieChart><Pie data={allocation} dataKey="value" innerRadius={66} outerRadius={86} paddingAngle={3} stroke="none">{allocation.map((x) => <Cell key={x.name} fill={x.color} />)}</Pie></PieChart></ResponsiveContainer>
              <div className="pie-center"><b>{portfolio.asset?.total_asset ? percent(portfolio.asset.market_value / portfolio.asset.total_asset * 100) : '—'}</b><span>持仓仓位</span></div>
            </div>
            <div className="allocation-list">{allocation.slice(0, 6).map((x) => <div key={x.name}><span><i style={{ background: x.color }} />{x.name}</span><b>{portfolio.asset?.total_asset ? percent(x.value / portfolio.asset.total_asset * 100) : '—'}</b></div>)}</div>
          </div> : <EmptyState text="暂无资产构成数据" />}
        </Panel>
        <Panel className="span-2">
          <PanelTitle title="关注股最新指标" subtitle={analysis?.trade_date ? `数据日期 ${formatTradeDate(analysis.trade_date)}` : '数据库暂无覆盖'} action={<button className="text-button" onClick={() => onNavigate('analysis')}>查看全部 <ChevronRight size={14} /></button>} />
          {latestRecords.length ? <div className="signal-list">{latestRecords.map((record, i) => <div className="signal-row" key={record.stock_code}><span className={`signal-icon s${i}`}><Database size={16} /></span><div className="stock-id"><b>{record.stock_name}</b><small>{record.stock_code}</small></div><span className={`strength ${(record.pct_chg ?? 0) >= 0 ? 'strong' : ''}`}>{record.pct_chg == null ? '无行情' : signedPercent(record.pct_chg)}</span><div className="signal-detail"><b>收盘 {record.close?.toFixed(2) ?? '—'}</b><small>主力净流入 {record.net_mf_amount_yuan == null ? '暂无' : formatSignedChineseMoney(record.net_mf_amount_yuan)}</small></div><time>{record.trade_date ? formatTradeDate(record.trade_date) : '—'}</time><ChevronRight size={16} /></div>)}</div> : <EmptyState text="关注股票暂无可用日频指标" />}
        </Panel>
        <Panel>
          <PanelTitle title="我的关注" subtitle={`${userWatchlist.length} 只股票`} action={<button className="text-button" onClick={() => onNavigate('profile')}>管理 <ChevronRight size={14} /></button>} />
          {userWatchlist.length ? <div className="compact-watch">{userWatchlist.slice(0, 4).map((s) => <div key={s.stock_code}><div><b>{s.stock_name}</b><small>{s.stock_code}</small></div>{s.spark.length > 1 ? <MiniSpark data={s.spark} positive={s.change >= 0} /> : <span className="muted">暂无历史</span>}<div className="price"><b>{s.price ? s.price.toFixed(2) : '—'}</b><Change value={s.change} /></div></div>)}</div> : <EmptyState text="尚未添加关注股票" />}
        </Panel>
      </div>
    </>
  )
}

function EmptyState({ text }: { text: string }) {
  return <div className="empty-state"><Database size={20} /><span>{text}</span></div>
}

function Metric({ icon, label, value, foot, positive }: { icon: React.ReactNode; label: string; value: string; foot: React.ReactNode; positive?: boolean }) {
  return <Panel className="metric-card"><div className="metric-head"><span>{icon}</span><label>{label}</label></div><strong className={positive ? 'up' : ''}>{value}</strong><div className="metric-foot">{foot}</div></Panel>
}

function Market({ overview }: { overview: MarketOverview | null }) {
  return (
    <>
      <div className="index-grid">
        <Panel className="index-card"><span>最新交易日</span><b>{overview?.trade_date ? formatTradeDate(overview.trade_date) : '—'}</b><small>日频数据库</small></Panel>
        <Panel className="index-card"><span>股票覆盖</span><b>{overview?.breadth?.stock_count?.toLocaleString() ?? '—'}</b><small>当日行情记录</small></Panel>
        <Panel className="index-card"><span>平盘家数</span><b>{overview?.breadth?.flat_count?.toLocaleString() ?? '—'}</b><small>涨跌幅等于 0</small></Panel>
        <Panel className="index-card"><span>板块更新时间</span><b>{overview?.sectors?.[0]?.captured_at ? formatDateTime(overview.sectors[0].captured_at) : '—'}</b><small>实时资金流快照</small></Panel>
      </div>
      <div className="market-layout">
        <Panel className="span-2">
          <PanelTitle title="指数行情" subtitle="本地数据库当前未采集指数价格与分时走势" />
          <EmptyState text="暂无可靠指数价格数据，已移除演示指数和模拟曲线" />
        </Panel>
        <Panel>
          <PanelTitle title="行业资金流" subtitle="最新行业快照，按主力净流入排序" />
          {overview?.sectors.length ? <div className="sector-list">{overview.sectors.map((sector, i) => <div key={sector.name}><span className="rank">{i + 1}</span><div><b>{sector.name}</b><small>净流入 {formatSignedChineseMoney(sector.main_net_amount_yuan)}</small></div><Change value={sector.pct_change} /><span className="heat-bar"><i style={{ width: `${Math.min(Math.abs(sector.pct_change) / 5 * 100, 100)}%` }} className={sector.pct_change < 0 ? 'negative' : ''} /></span></div>)}</div> : <EmptyState text="暂无行业资金流数据" />}
        </Panel>
        <Panel className="span-3">
          <PanelTitle title="全市场资金与情绪" subtitle={overview?.trade_date ? `数据日期 ${formatTradeDate(overview.trade_date)} · 来自本地数据库` : '正在读取真实市场数据'} />
          <div className="breadth-grid">
            <div><span>上涨家数</span><b className="up">{overview?.breadth?.up_count?.toLocaleString() ?? '—'}</b><small>{overview?.breadth ? `占比 ${percent(overview.breadth.up_count / overview.breadth.stock_count * 100)}` : '暂无数据'}</small></div>
            <div><span>下跌家数</span><b className="down">{overview?.breadth?.down_count?.toLocaleString() ?? '—'}</b><small>{overview?.breadth ? `占比 ${percent(overview.breadth.down_count / overview.breadth.stock_count * 100)}` : '暂无数据'}</small></div>
            <div><span>涨停 / 跌停</span><b>{overview?.limits ? `${overview.limits.limit_up_count} / ` : '— / '}<em className="down">{overview?.limits?.limit_down_count ?? '—'}</em></b><small>{overview?.limits ? `另有 ${overview.limits.open_board_count} 只炸板` : '暂无数据'}</small></div>
            <div><span>全市场成交额</span><b>{overview?.breadth ? formatChineseMoney(overview.breadth.turnover_yuan) : '—'}</b><small className={(overview?.breadth?.turnover_change_percent ?? 0) >= 0 ? 'up' : 'down'}>{overview?.breadth?.turnover_change_percent == null ? '暂无昨日对比' : `较前一交易日 ${signedPercent(overview.breadth.turnover_change_percent)}`}</small></div>
            <div><span>主力净流入</span><b className={(overview?.moneyflow?.main_net_amount_yuan ?? 0) >= 0 ? 'up' : 'down'}>{overview?.moneyflow ? formatSignedChineseMoney(overview.moneyflow.main_net_amount_yuan) : '—'}</b><small>{overview?.moneyflow ? `覆盖 ${overview.moneyflow.stock_count.toLocaleString()} 只股票` : '暂无数据'}</small></div>
          </div>
        </Panel>
      </div>
    </>
  )
}

function Analysis({ data }: { data: AnalysisData | null }) {
  return (
    <>
      <div className="analysis-hero">
        <div><span className="hero-icon"><Database /></span><div><h2>关注股票数据分析</h2><p>{data?.trade_date ? `以下指标来自本地数据库，数据日期 ${formatTradeDate(data.trade_date)}` : '当前关注股票暂无可用分析数据'}</p></div></div>
      </div>
      <div className="model-grid">
        <DataCard icon={<Star />} color="teal" title="关注股票" value={data?.summary.total ?? 0} desc="当前用户关注列表" />
        <DataCard icon={<Database />} color="purple" title="数据覆盖" value={data?.summary.covered ?? 0} desc="已有最新日频指标" />
        <DataCard icon={<TrendingUp />} color="gold" title="当日上涨" value={data?.summary.rising ?? 0} desc="最新交易日涨幅为正" />
        <DataCard icon={<Activity />} color="blue" title="主力净流入" value={data?.summary.positive_moneyflow ?? 0} desc="最新资金流为正" />
      </div>
      <Panel>
        <PanelTitle title="关注股指标" subtitle="价格、估值、换手、量比、资金流和筹码数据" />
        <div className="analysis-table table">
          <div className="tr th"><span>股票</span><span>收盘 / 涨跌</span><span>PE / PB</span><span>换手 / 量比</span><span>主力净流入</span><span>获利盘</span></div>
          {data?.records.map((record) => <div className="tr" key={record.stock_code}><span className="stock-id"><b>{record.stock_name}</b><small>{record.stock_code} · {record.trade_date ? formatTradeDate(record.trade_date) : '无日期'}</small></span><span><b>{record.close?.toFixed(2) ?? '—'}</b><small className={(record.pct_chg ?? 0) >= 0 ? 'up' : 'down'}>{record.pct_chg == null ? '—' : signedPercent(record.pct_chg)}</small></span><span>{record.pe_ttm?.toFixed(2) ?? '—'} / {record.pb?.toFixed(2) ?? '—'}</span><span>{record.turnover_rate?.toFixed(2) ?? '—'}% / {record.volume_ratio?.toFixed(2) ?? '—'}</span><span className={(record.net_mf_amount_yuan ?? 0) >= 0 ? 'up' : 'down'}>{record.net_mf_amount_yuan == null ? '—' : formatSignedChineseMoney(record.net_mf_amount_yuan)}</span><span>{record.winner_rate == null ? '—' : percent(record.winner_rate * (record.winner_rate <= 1 ? 100 : 1))}</span></div>)}
        </div>
        {!data?.records.length && <EmptyState text="请先在个人中心添加关注股票" />}
      </Panel>
    </>
  )
}

function DataCard({ icon, color, title, value, desc }: { icon: React.ReactNode; color: string; title: string; value: number; desc: string }) {
  return <Panel className="model-card"><span className={`model-icon ${color}`}>{icon}</span><div className="model-heading"><h3>{title}</h3></div><p>{desc}</p><div><b>{value}</b><span>只</span></div></Panel>
}

function Profile({ notify, portfolio, portfolioError, user, onUserChanged, onPortfolioChanged, userWatchlist, onWatchlistChanged }: { notify: (s: string) => void; portfolio: PortfolioData; portfolioError: string; user: AppUser; onUserChanged: (user: AppUser) => void; onPortfolioChanged: () => void; userWatchlist: UserWatchStock[]; onWatchlistChanged: () => void }) {
  const [tab, setTab] = useState<'watch' | 'positions' | 'security'>('watch')
  const [watchModal, setWatchModal] = useState(false)
  const [watchCode, setWatchCode] = useState('')
  const [watchName, setWatchName] = useState('')
  const addWatch = async () => {
    try {
      await api('/api/watchlist', { method: 'POST', body: JSON.stringify({ stock_code: watchCode, stock_name: watchName, tags: [] }) })
      setWatchModal(false); setWatchCode(''); setWatchName(''); onWatchlistChanged(); notify('关注股票已添加')
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '添加失败')
    }
  }
  const removeWatch = async (code: string) => {
    await api(`/api/watchlist/${code}`, { method: 'DELETE' })
    onWatchlistChanged(); notify('已取消关注')
  }
  return (
    <>
      <div className="profile-card">
        <div className="profile-avatar">{user.avatar}</div><div className="profile-main"><h2>{user.name}</h2><p>@{user.username} · {user.role === 'super_admin' ? '超级用户' : '普通用户'}</p><div><span><ShieldCheck size={14} /> 数据直接归属当前用户名</span><span>无需绑定证券账户</span></div></div>
        <button className="secondary-button" onClick={() => notify('个人资料已保存')}>编辑资料</button>
      </div>
      <div className="tabs"><button className={tab === 'watch' ? 'active' : ''} onClick={() => setTab('watch')}><Star size={16} />关注股票</button><button className={tab === 'positions' ? 'active' : ''} onClick={() => setTab('positions')}><BriefcaseBusiness size={16} />持仓信息</button><button className={tab === 'security' ? 'active' : ''} onClick={() => setTab('security')}><ShieldCheck size={16} />账户安全</button></div>
      {tab === 'watch' && <Panel>
        <PanelTitle title="我的关注" subtitle="关注的股票仅自己可见" action={<button className="primary-button small" onClick={() => setWatchModal(true)}><Plus size={15} />添加股票</button>} />
        <div className="watch-table table">
          <div className="tr th"><span>股票</span><span>最新价</span><span>涨跌幅</span><span>近7日走势</span><span>标签</span><span /></div>
          {userWatchlist.map((s) => <div className="tr" key={s.stock_code}><span className="stock-id"><b>{s.stock_name}</b><small>{s.stock_code}</small></span><span><b>{s.price ? s.price.toFixed(2) : '—'}</b></span><span><Change value={s.change} /></span><span>{s.spark.length > 1 ? <MiniSpark data={s.spark} positive={s.change >= 0} /> : <span className="muted">暂无历史</span>}</span><span className="tags">{s.tags.map((t) => <i key={t}>{t}</i>)}</span><button className="star-button active" aria-label="取消关注" onClick={() => removeWatch(s.stock_code)}><Star size={17} fill="currentColor" /></button></div>)}
        </div>
      </Panel>}
      {tab === 'positions' && <Positions portfolio={portfolio} error={portfolioError} notify={notify} onChanged={onPortfolioChanged} />}
      {tab === 'security' && <Security notify={notify} user={user} onUserChanged={onUserChanged} />}
      {watchModal && <div className="modal-layer"><div className="modal"><div className="modal-head"><div><h3>添加关注股票</h3><p>该关注列表只属于当前用户</p></div><button className="icon-button" onClick={() => setWatchModal(false)}><X /></button></div><label>股票代码<input value={watchCode} onChange={(e) => setWatchCode(e.target.value.toUpperCase())} placeholder="例如 600000.SH" autoFocus /></label><label>股票名称<input value={watchName} onChange={(e) => setWatchName(e.target.value)} placeholder="例如 浦发银行" /></label><div className="modal-actions"><button className="secondary-button" onClick={() => setWatchModal(false)}>取消</button><button className="primary-button" onClick={addWatch}>添加关注</button></div></div></div>}
    </>
  )
}

type EditablePosition = {
  stock_code: string
  stock_name: string
  volume: string
  can_use_volume: string
  avg_price: string
  current_price: string
}

const emptyPosition = (): EditablePosition => ({ stock_code: '', stock_name: '', volume: '', can_use_volume: '', avg_price: '', current_price: '' })

function Positions({ portfolio, error, notify, onChanged }: { portfolio: PortfolioData; error: string; notify: (s: string) => void; onChanged: () => void }) {
  const [editorOpen, setEditorOpen] = useState(false)
  const [cash, setCash] = useState('0')
  const [frozenCash, setFrozenCash] = useState('0')
  const [rows, setRows] = useState<EditablePosition[]>([emptyPosition()])
  const [saving, setSaving] = useState(false)
  const asset = portfolio.asset
  const totalProfit = portfolio.positions.reduce((sum, item) => sum + item.profit, 0)
  const totalCost = portfolio.positions.reduce((sum, item) => sum + item.avg_price * item.volume, 0)
  const save = async (source: 'manual' | 'import', nextRows = rows, nextCash = cash, nextFrozen = frozenCash) => {
    setSaving(true)
    try {
      await api('/api/portfolio/manual', {
        method: 'POST',
        body: JSON.stringify({
          source,
          cash: Number(nextCash || 0),
          frozen_cash: Number(nextFrozen || 0),
          positions: nextRows.map((row) => ({
            ...row,
            volume: Number(row.volume),
            can_use_volume: Number(row.can_use_volume || row.volume),
            avg_price: Number(row.avg_price),
            current_price: Number(row.current_price),
          })),
        }),
      })
      setEditorOpen(false)
      notify(source === 'import' ? '持仓文件导入成功' : '手工组合已保存')
      onChanged()
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '保存失败')
    } finally {
      setSaving(false)
    }
  }
  const openEditor = () => {
    setRows(portfolio.positions.length ? portfolio.positions.map((item) => ({
      stock_code: item.stock_code,
      stock_name: item.stock_name,
      volume: String(item.volume),
      can_use_volume: String(item.can_use_volume),
      avg_price: String(item.avg_price),
      current_price: String(item.current_price),
    })) : [emptyPosition()])
    setCash(String(portfolio.asset?.cash ?? 0))
    setFrozenCash(String(portfolio.asset?.frozen_cash ?? 0))
    setEditorOpen(true)
  }
  const importCsv = (file: File | undefined) => {
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      try {
        const lines = String(reader.result).replace(/^\uFEFF/, '').split(/\r?\n/).filter(Boolean)
        const delimiter = lines[0].includes('\t') ? '\t' : ','
        const headers = lines[0].split(delimiter).map((item) => item.trim().toLowerCase())
        const aliases: Record<string, string[]> = {
          stock_code: ['stock_code', '股票代码', '代码'],
          stock_name: ['stock_name', '股票名称', '名称'],
          volume: ['volume', '持仓数量', '数量'],
          can_use_volume: ['can_use_volume', '可用数量', '可卖数量'],
          avg_price: ['avg_price', '成本价', '持仓均价'],
          current_price: ['current_price', '现价', '最新价'],
        }
        const indexOf = (key: string) => headers.findIndex((item) => aliases[key].includes(item))
        const parsed = lines.slice(1).map((line) => {
          const cells = line.split(delimiter).map((item) => item.trim().replace(/^"|"$/g, ''))
          const value = (key: string) => cells[indexOf(key)] || ''
          return { stock_code: value('stock_code'), stock_name: value('stock_name'), volume: value('volume'), can_use_volume: value('can_use_volume'), avg_price: value('avg_price'), current_price: value('current_price') }
        }).filter((row) => row.stock_code)
        if (!parsed.length) throw new Error('文件中没有可识别的持仓')
        save('import', parsed, '0', '0')
      } catch (reason) {
        notify(reason instanceof Error ? reason.message : '文件解析失败')
      }
    }
    reader.readAsText(file)
  }
  return <><div className="position-summary"><Panel><span>总资产</span><b>{asset ? money(asset.total_asset) : '—'}</b><small>{error || (asset ? `更新于 ${formatDateTime(asset.fetched_at)}` : '尚未录入')}</small></Panel><Panel><span>持仓市值</span><b>{asset ? money(asset.market_value) : '—'}</b><small>仓位 {asset && asset.total_asset ? percent(asset.market_value / asset.total_asset * 100) : '—'}</small></Panel><Panel><span>持仓盈亏</span><b className={totalProfit >= 0 ? 'up' : 'down'}>{signedMoney(totalProfit)}</b><small className={totalProfit >= 0 ? 'up' : 'down'}>{totalCost ? `${totalProfit >= 0 ? '+' : ''}${percent(totalProfit / totalCost * 100)}` : '—'}</small></Panel><Panel><span>可用资金</span><b>{asset ? money(asset.cash) : '—'}</b><small>冻结 {asset ? money(asset.frozen_cash) : '—'}</small></Panel></div><Panel><PanelTitle title="当前持仓" subtitle="持仓数据直接归属当前用户名" action={<div className="portfolio-actions"><label className="secondary-button small file-button"><Upload size={14} />导入 CSV<input type="file" accept=".csv,.tsv,.txt" onChange={(e) => importCsv(e.target.files?.[0])} /></label><button className="primary-button small" onClick={openEditor}><Plus size={14} />手工录入</button></div>} /><div className="positions-table table"><div className="tr th"><span>股票</span><span>持仓 / 可用</span><span>成本价</span><span>现价</span><span>市值</span><span>持仓盈亏</span><span>收益率</span></div>{portfolio.positions.map((p) => { const rate = p.avg_price && p.volume ? p.profit / (p.avg_price * p.volume) * 100 : 0; return <div className="tr" key={p.stock_code}><span className="stock-id"><b>{p.stock_name}</b><small>{p.stock_code}</small></span><span>{p.volume.toLocaleString()} / {p.can_use_volume.toLocaleString()}</span><span>{p.avg_price.toFixed(3)}</span><span><b>{p.current_price.toFixed(3)}</b></span><span>{money(p.market_value)}</span><span className={p.profit >= 0 ? 'up' : 'down'}>{signedMoney(p.profit)}</span><span className={rate >= 0 ? 'up' : 'down'}>{rate >= 0 ? '+' : ''}{percent(rate)}</span></div>})}</div></Panel>{editorOpen && <div className="modal-layer"><div className="modal portfolio-modal"><div className="modal-head"><div><h3>编辑个人资产与持仓</h3><p>保存后直接更新当前用户名下的数据</p></div><button className="icon-button" onClick={() => setEditorOpen(false)}><X /></button></div><div className="cash-fields"><label>可用资金<input type="number" min="0" value={cash} onChange={(e) => setCash(e.target.value)} /></label><label>冻结资金<input type="number" min="0" value={frozenCash} onChange={(e) => setFrozenCash(e.target.value)} /></label></div><div className="manual-grid header"><span>股票代码</span><span>名称</span><span>数量</span><span>可用</span><span>成本价</span><span>现价</span><span /></div>{rows.map((row, index) => <div className="manual-grid" key={index}>{(['stock_code', 'stock_name', 'volume', 'can_use_volume', 'avg_price', 'current_price'] as const).map((key) => <input key={key} value={row[key]} type={['volume', 'can_use_volume', 'avg_price', 'current_price'].includes(key) ? 'number' : 'text'} placeholder={{ stock_code: '600000.SH', stock_name: '浦发银行', volume: '1000', can_use_volume: '1000', avg_price: '10.20', current_price: '10.50' }[key]} onChange={(e) => setRows(rows.map((item, rowIndex) => rowIndex === index ? { ...item, [key]: e.target.value } : item))} />)}<button className="icon-button remove-row" onClick={() => setRows(rows.filter((_, rowIndex) => rowIndex !== index))}><X size={14} /></button></div>)}<button className="add-row" onClick={() => setRows([...rows, emptyPosition()])}><Plus size={14} />增加一行</button><div className="csv-tip"><FileSpreadsheet size={16} /><span>CSV 表头支持：股票代码、股票名称、持仓数量、可用数量、成本价、现价</span></div><div className="modal-actions"><button className="secondary-button" onClick={() => setEditorOpen(false)}>取消</button><button className="primary-button" disabled={saving} onClick={() => save('manual')}>{saving ? '保存中…' : '保存数据'}</button></div></div></div>}</>
}

function Security({ notify, user, onUserChanged }: { notify: (s: string) => void; user: AppUser; onUserChanged: (user: AppUser) => void }) {
  const [open, setOpen] = useState(false)
  return <><Panel><PanelTitle title="账户安全" subtitle="管理登录密码与安全验证" /><div className="security-list"><div><span className="security-icon"><ShieldCheck /></span><div><b>登录密码</b><small>修改时需要验证当前密码，其他设备会自动退出</small></div><button className="secondary-button small" onClick={() => setOpen(true)}>修改密码</button></div><div><span className="security-icon"><CircleUserRound /></span><div><b>数据隔离</b><small>关注股票、资产和持仓均按当前用户名隔离</small></div><span className="enabled-label">已启用</span></div><div><span className="security-icon"><Activity /></span><div><b>登录会话</b><small>会话有效期 14 天，修改密码后其他会话失效</small></div><span className="enabled-label">安全</span></div></div></Panel>{open && <PasswordChangeModal user={user} onClose={() => setOpen(false)} onChanged={onUserChanged} notify={notify} />}</>
}

function PasswordChangeModal({ user, required = false, onClose, onChanged, notify }: { user: AppUser; required?: boolean; onClose?: () => void; onChanged: (user: AppUser) => void; notify: (message: string) => void }) {
  const [currentPassword, setCurrentPassword] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [visible, setVisible] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const submit = async () => {
    if (password.length < 10) return setError('新密码至少需要 10 位')
    if (password !== confirm) return setError('两次输入的新密码不一致')
    setSaving(true); setError('')
    try {
      await api('/api/auth/change-password', { method: 'POST', body: JSON.stringify({ current_password: currentPassword, password }) })
      onChanged({ ...user, must_change_password: false })
      onClose?.()
      notify('密码已修改，其他设备已退出登录')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '密码修改失败')
    } finally {
      setSaving(false)
    }
  }
  return <div className="modal-layer password-layer"><div className="modal password-modal"><div className="modal-head"><div><h3>{required ? '首次登录，请修改密码' : '修改登录密码'}</h3><p>{required ? '使用临时密码登录后，必须设置自己的新密码' : '验证当前密码后设置新密码'}</p></div>{!required && <button className="icon-button" onClick={onClose}><X /></button>}</div><label>当前密码<div className="password-input"><input type={visible ? 'text' : 'password'} value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} autoFocus /><button type="button" onClick={() => setVisible(!visible)}>{visible ? <EyeOff size={15} /> : <Eye size={15} />}</button></div></label><label>新密码<input type={visible ? 'text' : 'password'} value={password} onChange={(e) => setPassword(e.target.value)} placeholder="至少 10 位，不能与当前密码相同" /></label><label>确认新密码<input type={visible ? 'text' : 'password'} value={confirm} onChange={(e) => setConfirm(e.target.value)} placeholder="再次输入新密码" /></label><div className="password-rules"><span className={password.length >= 10 ? 'ok' : ''}>至少 10 位</span><span className={password && password !== currentPassword ? 'ok' : ''}>不同于当前密码</span><span className={confirm && confirm === password ? 'ok' : ''}>两次输入一致</span></div>{error && <div className="form-error">{error}</div>}<div className="modal-actions">{!required && <button className="secondary-button" onClick={onClose}>取消</button>}<button className="primary-button" disabled={saving} onClick={submit}>{saving ? '正在保存…' : '确认修改'}</button></div></div></div>
}

function UserManagement({ notify }: { notify: (s: string) => void }) {
  const [users, setUsers] = useState<AppUser[]>([])
  const [query, setQuery] = useState('')
  const [modal, setModal] = useState(false)
  const [name, setName] = useState('')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [resetUser, setResetUser] = useState<AppUser | null>(null)
  const [resetPassword, setResetPassword] = useState('')
  const loadUsers = () => api<{ users: AppUser[] }>('/api/users').then((data) => setUsers(data.users)).catch((error) => notify(error.message))
  useEffect(() => { void loadUsers() }, [])
  const filtered = users.filter((u) => `${u.name}${u.username}`.toLowerCase().includes(query.toLowerCase()))
  const addUser = async () => {
    try {
      await api('/api/users', { method: 'POST', body: JSON.stringify({ name, username, password }) })
      setModal(false); setName(''); setUsername(''); setPassword(''); notify('新用户已创建')
      loadUsers()
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '创建失败')
    }
  }
  const toggle = async (user: AppUser) => {
    try {
      await api(`/api/users/${user.id}/status`, { method: 'PATCH', body: JSON.stringify({ status: user.status === 'active' ? 'disabled' : 'active' }) })
      notify(user.status === 'active' ? '用户已停用' : '用户已启用')
      loadUsers()
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '操作失败')
    }
  }
  const resetUserPassword = async () => {
    if (!resetUser) return
    try {
      await api(`/api/users/${resetUser.id}/reset-password`, { method: 'POST', body: JSON.stringify({ password: resetPassword }) })
      setResetUser(null); setResetPassword('')
      notify('临时密码已重置，该用户下次登录必须修改密码')
    } catch (reason) {
      notify(reason instanceof Error ? reason.message : '重置失败')
    }
  }
  return (
    <>
      <div className="admin-banner"><span><UserCog /></span><div><h2>用户与权限</h2><p>创建平台用户、管理访问状态与角色权限。当前登录账号拥有超级用户权限。</p></div><span className="admin-badge"><ShieldCheck size={15} />SUPER ADMIN</span></div>
      <div className="user-stats"><Panel><span>全部用户</span><b>{users.length}</b><Users /></Panel><Panel><span>正常使用</span><b>{users.filter((u) => u.status === 'active').length}</b><Activity /></Panel><Panel><span>普通用户</span><b>{users.filter((u) => u.role === 'user').length}</b><ArrowUpRight /></Panel><Panel><span>已停用</span><b>{users.filter((u) => u.status === 'disabled').length}</b><ArrowDownRight /></Panel></div>
      <Panel>
        <div className="users-toolbar"><div><h3>用户列表</h3><p>管理所有平台成员及其账户状态</p></div><div className="toolbar-actions"><label className="search-box"><Search size={16} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="搜索用户" /></label><button className="primary-button" onClick={() => setModal(true)}><Plus size={17} />新建用户</button></div></div>
        <div className="users-table table"><div className="tr th"><span>用户</span><span>角色</span><span>状态</span><span>最近登录</span><span>操作</span></div>{filtered.map((u) => <div className="tr" key={u.id}><span className="user-cell"><i>{u.avatar}</i><span><b>{u.name}</b><small>@{u.username}</small></span></span><span><i className={`role-badge ${u.role}`}>{u.role === 'super_admin' ? '超级用户' : '普通用户'}</i></span><span><i className={`status-badge ${u.status}`}><i />{u.status === 'active' ? '正常' : '已停用'}</i></span><span className="muted">{u.last_login_at ? formatDateTime(u.last_login_at) : '尚未登录'}</span><span className="row-actions">{u.role !== 'super_admin' && <><button className="secondary-button small" onClick={() => setResetUser(u)}>重置密码</button><button className={u.status === 'active' ? 'ghost-danger' : 'secondary-button small'} onClick={() => toggle(u)}>{u.status === 'active' ? '停用' : '启用'}</button></>}</span></div>)}</div>
      </Panel>
      {modal && <div className="modal-layer" onMouseDown={() => setModal(false)}><div className="modal" onMouseDown={(e) => e.stopPropagation()}><div className="modal-head"><div><h3>新建普通用户</h3><p>资产、持仓和关注将直接归属该用户名</p></div><button className="icon-button" onClick={() => setModal(false)}><X /></button></div><label>显示名称<input value={name} onChange={(e) => setName(e.target.value)} placeholder="例如 张三" autoFocus /></label><label>用户名<input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="例如 zhangsan01" /><small className="field-hint">英文开头，仅英文和数字，至少 5 个字符</small></label><label>初始密码<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="至少 10 位" /></label><label>用户角色<select disabled><option>普通用户</option></select></label><div className="modal-actions"><button className="secondary-button" onClick={() => setModal(false)}>取消</button><button className="primary-button" onClick={addUser}>创建用户</button></div></div></div>}
      {resetUser && <div className="modal-layer"><div className="modal"><div className="modal-head"><div><h3>重置用户密码</h3><p>为 @{resetUser.username} 设置临时密码，并退出其所有设备</p></div><button className="icon-button" onClick={() => setResetUser(null)}><X /></button></div><label>临时密码<input type="password" value={resetPassword} onChange={(e) => setResetPassword(e.target.value)} placeholder="至少 10 位" autoFocus /></label><div className="form-notice">用户下次登录时必须修改此临时密码。</div><div className="modal-actions"><button className="secondary-button" onClick={() => setResetUser(null)}>取消</button><button className="primary-button" onClick={resetUserPassword}>确认重置</button></div></div></div>}
    </>
  )
}

const moneyFormatter = new Intl.NumberFormat('zh-CN', { style: 'currency', currency: 'CNY', minimumFractionDigits: 2 })
const money = (value: number) => moneyFormatter.format(value)
const signedMoney = (value: number) => `${value >= 0 ? '+' : '-'}${moneyFormatter.format(Math.abs(value))}`
const percent = (value: number) => `${value.toFixed(1)}%`
const signedPercent = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`
const formatChineseMoney = (value: number) => {
  if (Math.abs(value) >= 1e12) return `${(value / 1e12).toFixed(2)} 万亿`
  if (Math.abs(value) >= 1e8) return `${(value / 1e8).toFixed(1)} 亿`
  if (Math.abs(value) >= 1e4) return `${(value / 1e4).toFixed(1)} 万`
  return moneyFormatter.format(value)
}
const formatSignedChineseMoney = (value: number) => `${value >= 0 ? '+' : '-'}${formatChineseMoney(Math.abs(value))}`
const formatTradeDate = (value: string) => new Intl.DateTimeFormat('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date(`${value}T00:00:00+08:00`))
const formatDateTime = (value: string) => {
  const zoned = /(?:Z|[+-]\d{2}:\d{2})$/.test(value) ? value : `${value}+08:00`
  return new Intl.DateTimeFormat('zh-CN', { timeZone: 'Asia/Shanghai', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(zoned))
}

export default App

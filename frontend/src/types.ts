export type Page = 'dashboard' | 'market' | 'analysis' | 'profile' | 'users'

export type UserRole = 'super_admin' | 'user'

export interface AppUser {
  id: number
  name: string
  username: string
  role: UserRole
  status: 'active' | 'disabled'
  lastLogin: string
  avatar: string
  must_change_password?: boolean
  created_at?: string
  last_login_at?: string | null
}

export interface Stock {
  code: string
  name: string
  price: number
  change: number
  tags: string[]
  spark: number[]
}

export interface UserWatchStock {
  stock_code: string
  stock_name: string
  price: number
  change: number
  tags: string[]
  note: string
  spark: number[]
}

export interface PortfolioAsset {
  user_id: number
  cash: number
  frozen_cash: number
  market_value: number
  total_asset: number
  data_source: 'manual' | 'import' | 'migrated'
  fetched_at: string
}

export interface PortfolioPosition {
  stock_code: string
  stock_name: string
  volume: number
  can_use_volume: number
  avg_price: number
  market_value: number
  current_price: number
  profit: number
  data_source: 'manual' | 'import' | 'migrated'
  fetched_at: string
}

export interface PortfolioData {
  asset: PortfolioAsset | null
  positions: PortfolioPosition[]
}

export interface MarketOverview {
  trade_date: string | null
  breadth: {
    trade_date: string
    up_count: number
    down_count: number
    flat_count: number
    stock_count: number
    turnover_yuan: number
    turnover_change_percent: number | null
  } | null
  limits: {
    trade_date: string
    limit_up_count: number
    limit_down_count: number
    open_board_count: number
  } | null
  moneyflow: {
    trade_date: string
    stock_count: number
    main_net_amount_yuan: number
  } | null
  sectors: Array<{
    name: string
    pct_change: number
    main_net_amount_yuan: number
    captured_at: string
  }>
  sources: Record<string, string>
}

export interface AnalysisData {
  trade_date: string | null
  summary: {
    covered: number
    rising: number
    positive_moneyflow: number
    total: number
  }
  records: Array<{
    stock_code: string
    stock_name: string
    trade_date: string | null
    close: number | null
    pct_chg: number | null
    pe_ttm: number | null
    pb: number | null
    turnover_rate: number | null
    volume_ratio: number | null
    net_mf_amount_yuan: number | null
    winner_rate: number | null
  }>
}

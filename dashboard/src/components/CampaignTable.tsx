import { useMemo, useState } from 'react'
import { strategyLabel, timeAgo } from '../format'
import type { CampaignSummary } from '../types'

type SortKey = 'name' | 'strategy' | 'status' | 'runs' | 'rounds' | 'r2' | 'rmse' | 'updated'
type StatusFilter = 'all' | 'running' | 'complete'

const NUMERIC_KEYS = new Set<SortKey>(['runs', 'rounds', 'r2', 'rmse', 'updated'])

function metricText(v: number | null | undefined, digits = 3): string {
  return v === null || v === undefined ? '—' : v.toFixed(digits)
}

function sortValue(c: CampaignSummary, key: SortKey): string | number {
  switch (key) {
    case 'name':
      return c.name.toLowerCase()
    case 'strategy':
      return strategyLabel(c.strategy).toLowerCase()
    case 'status':
      return c.status
    case 'runs':
      return c.n_complete
    case 'rounds':
      return c.tracking.n_rounds ?? -1
    case 'r2':
      return c.tracking.r_squared ?? -Infinity
    case 'rmse':
      return c.tracking.rmse ?? Infinity
    case 'updated':
      return c.updated_at ? new Date(c.updated_at).getTime() : 0
  }
}

function SortHeader({
  label,
  sortKey,
  active,
  dir,
  onClick,
  className,
}: {
  label: string
  sortKey: SortKey
  active: boolean
  dir: 'asc' | 'desc'
  onClick: (key: SortKey) => void
  className?: string
}) {
  return (
    <button
      type="button"
      className={`sort-btn${active ? ' active' : ''}${className ? ` ${className}` : ''}`}
      onClick={() => onClick(sortKey)}
    >
      {label}
      {active && <span className="sort-arrow">{dir === 'asc' ? '▲' : '▼'}</span>}
    </button>
  )
}

export function CampaignTable({
  campaigns,
  expandedId,
  onToggle,
  renderDetail,
}: {
  campaigns: CampaignSummary[]
  expandedId: string | null
  onToggle: (id: string) => void
  renderDetail: (id: string) => React.ReactNode
}) {
  const [sortKey, setSortKey] = useState<SortKey | null>(null)
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [strategyFilter, setStrategyFilter] = useState<Set<string>>(new Set())

  const strategies = useMemo(
    () => Array.from(new Set(campaigns.map((c) => c.strategy))).sort(),
    [campaigns],
  )

  function toggleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir(NUMERIC_KEYS.has(key) ? 'desc' : 'asc')
    }
  }

  function toggleStrategy(s: string) {
    setStrategyFilter((prev) => {
      const next = new Set(prev)
      if (next.has(s)) next.delete(s)
      else next.add(s)
      return next
    })
  }

  const rows = useMemo(() => {
    let filtered = campaigns
    if (statusFilter !== 'all') filtered = filtered.filter((c) => c.status === statusFilter)
    if (strategyFilter.size > 0) filtered = filtered.filter((c) => strategyFilter.has(c.strategy))

    const sorted = filtered.slice()
    if (sortKey) {
      sorted.sort((a, b) => {
        const av = sortValue(a, sortKey)
        const bv = sortValue(b, sortKey)
        const cmp = av < bv ? -1 : av > bv ? 1 : 0
        return sortDir === 'asc' ? cmp : -cmp
      })
    } else {
      sorted.sort((a, b) => (a.status !== 'running' ? 1 : 0) - (b.status !== 'running' ? 1 : 0) ||
        a.name.localeCompare(b.name))
    }
    return sorted
  }, [campaigns, sortKey, sortDir, statusFilter, strategyFilter])

  return (
    <>
      <div className="campaign-toolbar">
        <div className="filter-group">
          {(['all', 'running', 'complete'] as StatusFilter[]).map((s) => (
            <button
              key={s}
              type="button"
              className={`filter-chip${statusFilter === s ? ' active' : ''}`}
              onClick={() => setStatusFilter(s)}
            >
              {s}
            </button>
          ))}
        </div>
        <div className="filter-group">
          {strategies.map((s) => (
            <button
              key={s}
              type="button"
              className={`filter-chip${strategyFilter.has(s) ? ' active' : ''}`}
              onClick={() => toggleStrategy(s)}
            >
              {strategyLabel(s)}
            </button>
          ))}
        </div>
      </div>

      <div className="campaign-table">
        <div className="campaign-row campaign-head">
          <span />
          <SortHeader label="campaign" sortKey="name" active={sortKey === 'name'} dir={sortDir} onClick={toggleSort} />
          <SortHeader label="strategy" sortKey="strategy" active={sortKey === 'strategy'} dir={sortDir} onClick={toggleSort} />
          <SortHeader label="status" sortKey="status" active={sortKey === 'status'} dir={sortDir} onClick={toggleSort} className="num" />
          <SortHeader label="runs" sortKey="runs" active={sortKey === 'runs'} dir={sortDir} onClick={toggleSort} className="num" />
          <SortHeader label="rounds" sortKey="rounds" active={sortKey === 'rounds'} dir={sortDir} onClick={toggleSort} className="num" />
          <SortHeader label="R²" sortKey="r2" active={sortKey === 'r2'} dir={sortDir} onClick={toggleSort} className="num" />
          <SortHeader label="RMSE" sortKey="rmse" active={sortKey === 'rmse'} dir={sortDir} onClick={toggleSort} className="num" />
          <SortHeader label="updated" sortKey="updated" active={sortKey === 'updated'} dir={sortDir} onClick={toggleSort} className="num" />
          <span />
        </div>

        {rows.length === 0 && <div className="empty">No campaigns match these filters.</div>}

        {rows.map((c) => {
          const expanded = c.id === expandedId
          return (
            <div key={c.id} className="campaign-row-group">
              <div
                className={`campaign-row${expanded ? ' expanded' : ''}`}
                onClick={() => onToggle(c.id)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') onToggle(c.id)
                }}
              >
                <span className={`status-dot ${c.status === 'running' ? 'pending' : 'complete'}`} />
                <span className="campaign-row-name">{c.name}</span>
                <span className="campaign-row-strategy">{strategyLabel(c.strategy)}</span>
                <span className={`pill num ${c.status}`}>{c.status}</span>
                <span className="num tabular">{c.n_complete}/{c.budget.n_total}</span>
                <span className="num tabular">{c.tracking.n_rounds ?? '—'}</span>
                <span className="num tabular metric-r2">{metricText(c.tracking.r_squared, 4)}</span>
                <span className="num tabular">{metricText(c.tracking.rmse, 3)}</span>
                <span className="num campaign-row-updated">{timeAgo(c.updated_at)}</span>
                <span className="chevron">▶</span>
              </div>
              {expanded && <div className="campaign-detail-slot">{renderDetail(c.id)}</div>}
            </div>
          )
        })}
      </div>
    </>
  )
}

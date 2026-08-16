import type { CampaignSummary } from '../types'

const STRATEGY_LABEL: Record<string, string> = {
  alc_imse: 'ALC / IMSE',
  epistemic: 'Epistemic-only',
  max_variance: 'Max-variance',
  latin_hypercube: 'Latin hypercube',
  cost_aware_alc: 'Cost-aware ALC',
  fixed_length_alc: 'Fixed-length ALC',
}

export function strategyLabel(strategy: string): string {
  return STRATEGY_LABEL[strategy] ?? strategy
}

export function timeAgo(iso?: string): string {
  if (!iso) return ''
  // updated_at is written by Python's timezone-aware isoformat(), so it already
  // carries an explicit +00:00 offset — never append 'Z' to it.
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000)
  if (s < 60) return `${Math.round(s)}s ago`
  if (s < 3600) return `${Math.round(s / 60)}m ago`
  return `${Math.round(s / 3600)}h ago`
}

export function CampaignRail({
  campaigns,
  selectedId,
  onSelect,
}: {
  campaigns: CampaignSummary[]
  selectedId: string | null
  onSelect: (id: string) => void
}) {
  return (
    <div className="campaign-rail">
      {campaigns.map((c) => (
        <button
          key={c.id}
          type="button"
          className={`campaign-card${c.id === selectedId ? ' selected' : ''}`}
          onClick={() => onSelect(c.id)}
        >
          <div className="campaign-card-top">
            <div>
              <div className="campaign-name">{c.name}</div>
              <div className="strategy-tag">{strategyLabel(c.strategy)}</div>
            </div>
            <span className={`pill ${c.status}`}>{c.status}</span>
          </div>
          <div className={`bar ${c.status}`}>
            <span style={{ width: `${Math.min(100, Math.round(c.progress * 100))}%` }} />
          </div>
          <div className="progress-line">
            <span>
              <b>{c.n_complete}</b> / {c.budget.n_total} runs
            </span>
            <span>
              {c.status === 'running' ? timeAgo(c.updated_at) : `${Math.round(c.progress * 100)}%`}
            </span>
          </div>
        </button>
      ))}
    </div>
  )
}

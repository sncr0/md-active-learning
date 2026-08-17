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

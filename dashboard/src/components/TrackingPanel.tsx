import { useMemo, useRef, useState } from 'react'
import type { CampaignTracking, TrackingRound } from '../types'

const VB_W = 640

function niceFormat(v: number): string {
  const a = Math.abs(v)
  if (a !== 0 && (a < 0.01 || a >= 1000)) return v.toExponential(2)
  return v.toFixed(a < 1 ? 3 : 2)
}

interface Point {
  round: number
  value: number
  n_points: number
}

function RoundLineChart({
  title,
  unit,
  points,
  height = 160,
  emphasis = false,
}: {
  title: string
  unit?: string
  points: Point[]
  height?: number
  emphasis?: boolean
}) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)

  const pad = { top: 14, right: 16, bottom: 24, left: 44 }
  const plotW = VB_W - pad.left - pad.right
  const plotH = height - pad.top - pad.bottom

  const rounds = points.map((p) => p.round)
  const values = points.map((p) => p.value)
  const minR = Math.min(...rounds)
  const maxR = Math.max(...rounds)
  const minV = Math.min(...values)
  const maxV = Math.max(...values)
  const vSpan = maxV - minV || Math.abs(maxV) || 1
  const rSpan = maxR - minR || 1

  const xOf = (r: number) => pad.left + ((r - minR) / rSpan) * plotW
  const yOf = (v: number) => pad.top + (1 - (v - minV) / vSpan) * plotH

  const coords = points.map((p) => ({ x: xOf(p.round), y: yOf(p.value), ...p }))
  const linePath = coords.map((c, i) => `${i === 0 ? 'M' : 'L'} ${c.x} ${c.y}`).join(' ')
  const areaPath = `${linePath} L ${coords[coords.length - 1].x} ${pad.top + plotH} ` +
    `L ${coords[0].x} ${pad.top + plotH} Z`

  const last = coords[coords.length - 1]
  const hovered = hoverIdx !== null ? coords[hoverIdx] : null

  function handleMove(e: React.PointerEvent<SVGSVGElement>) {
    const svg = svgRef.current
    if (!svg || coords.length === 0) return
    const rect = svg.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * VB_W
    let nearest = 0
    let best = Infinity
    coords.forEach((c, i) => {
      const d = Math.abs(c.x - px)
      if (d < best) {
        best = d
        nearest = i
      }
    })
    setHoverIdx(nearest)
  }

  const color = emphasis ? 'var(--accent)' : 'var(--muted)'
  const fmt = (v: number) => `${niceFormat(v)}${unit ? ` ${unit}` : ''}`

  return (
    <div className={`tracking-chart${emphasis ? ' emphasis' : ''}`}>
      <div className="tracking-chart-title">{title}</div>
      {coords.length < 2 ? (
        <div className="tracking-chart-thin">not enough rounds yet</div>
      ) : (
        <svg
          ref={svgRef}
          viewBox={`0 0 ${VB_W} ${height}`}
          width="100%"
          height={height}
          className="tracking-svg"
          onPointerMove={handleMove}
          onPointerLeave={() => setHoverIdx(null)}
        >
          {[minV, (minV + maxV) / 2, maxV].map((v, i) => (
            <g key={i}>
              <line
                x1={pad.left} x2={VB_W - pad.right} y1={yOf(v)} y2={yOf(v)}
                stroke="var(--hair)" strokeWidth={1}
              />
              <text x={pad.left - 8} y={yOf(v)} textAnchor="end" dominantBaseline="middle"
                className="tracking-axis-label">
                {niceFormat(v)}
              </text>
            </g>
          ))}

          <text x={xOf(minR)} y={height - 6} textAnchor="start" className="tracking-axis-label">
            round {minR}
          </text>
          <text x={xOf(maxR)} y={height - 6} textAnchor="end" className="tracking-axis-label">
            round {maxR}
          </text>

          <path d={areaPath} fill={color} opacity={0.1} stroke="none" />
          <path d={linePath} fill="none" stroke={color} strokeWidth={2}
            strokeLinecap="round" strokeLinejoin="round" />

          <circle cx={last.x} cy={last.y} r={4} fill={color} stroke="var(--bg)" strokeWidth={2} />
          <text x={last.x - 8} y={last.y - 10} textAnchor="end" className="tracking-end-label">
            {fmt(last.value)}
          </text>

          {hovered && (
            <g>
              <line x1={hovered.x} x2={hovered.x} y1={pad.top} y2={pad.top + plotH}
                stroke="var(--hair-strong)" strokeWidth={1} />
              <circle cx={hovered.x} cy={hovered.y} r={4} fill={color}
                stroke="var(--bg)" strokeWidth={2} />
            </g>
          )}
        </svg>
      )}
      {hovered && (
        <div className="tracking-tooltip">
          <span className="tracking-tooltip-value">{fmt(hovered.value)}</span>
          <span className="tracking-tooltip-round">round {hovered.round} · {hovered.n_points} pts</span>
        </div>
      )}
    </div>
  )
}

const METRIC_LABELS: Record<string, string> = {
  rmse_vs_reference: 'RMSE vs. reference EOS',
  log_marginal_likelihood: 'Log-marginal-likelihood',
}

function seriesFor(rounds: TrackingRound[], key: string): Point[] {
  return rounds
    .filter((r) => r.metrics[key] !== undefined)
    .map((r) => ({ round: r.round, value: r.metrics[key], n_points: r.n_points }))
}

export function TrackingPanel({ tracking }: { tracking: CampaignTracking | null }) {
  const rmse = useMemo(
    () => (tracking?.available ? seriesFor(tracking.rounds, 'rmse_vs_reference') : []),
    [tracking],
  )
  const lml = useMemo(
    () => (tracking?.available ? seriesFor(tracking.rounds, 'log_marginal_likelihood') : []),
    [tracking],
  )

  if (!tracking) return null

  if (!tracking.available) {
    return (
      <div className="tracking-panel tracking-panel-empty">
        No MLflow tracking data for this campaign yet — surrogate fits are logged as the
        campaign runs its active-learning rounds.{' '}
        <a href="https://github.com/sncr0/md-active-learning#experiment-tracking-mlflow" target="_blank" rel="noreferrer">
          set up MLflow ↗
        </a>
      </div>
    )
  }

  return (
    <div className="tracking-panel">
      <div className="tracking-panel-head">
        <h4>Surrogate learning</h4>
        <a className="tracking-link" href={tracking.mlflow_url} target="_blank" rel="noreferrer">
          open in MLflow ↗
        </a>
      </div>
      <div className="tracking-charts">
        <RoundLineChart title={METRIC_LABELS.rmse_vs_reference} points={rmse} height={168} emphasis />
        <RoundLineChart title={METRIC_LABELS.log_marginal_likelihood} points={lml} height={112} />
      </div>
    </div>
  )
}

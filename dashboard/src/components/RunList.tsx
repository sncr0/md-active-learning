import { useState } from 'react'
import type { CampaignDetail, ObservationStat, Run } from '../types'

function fmt(n: number | undefined, d = 3): string {
  return n === undefined ? '—' : n.toFixed(d)
}

function roundLabel(run: Run): string {
  return run.round === 0 ? 'initial' : `round ${run.round}`
}

const OBS_UNIT: Record<string, string> = {
  pressure: 'P*',
  energy: 'U*',
}

function ObsCard({ name, stat }: { name: string; stat: ObservationStat }) {
  return (
    <div className="obs-card">
      <div className="obs-name">{name}</div>
      <div className="obs-value">
        {OBS_UNIT[name] ?? name} {fmt(stat.value, 4)} <span style={{ color: 'var(--faint)' }}>± {fmt(stat.sigma, 4)}</span>
      </div>
      <div className="obs-neff">n_eff {fmt(stat.n_eff, 1)}</div>
    </div>
  )
}

function RunDetail({ run }: { run: Run }) {
  const c = run.config
  return (
    <div className="run-detail">
      <div>
        <h4>Run</h4>
        <dl>
          <dt>hash</dt>
          <dd className="hash">{run.run_hash}</dd>
          <dt>status</dt>
          <dd>{run.status}</dd>
          <dt>created</dt>
          <dd>{run.created_at ? new Date(run.created_at + 'Z').toLocaleString() : '—'}</dd>
          <dt>wall clock</dt>
          <dd>{fmt(run.wall_clock_s, 2)} s</dd>
          <dt>frames</dt>
          <dd>{run.n_frames} (equil cutoff {run.equil_cutoff})</dd>
        </dl>
      </div>
      <div>
        <h4>Config</h4>
        <dl>
          <dt>T*, ρ*</dt>
          <dd>{fmt(c?.temperature, 4)}, {fmt(c?.density, 4)}</dd>
          <dt>n_steps</dt>
          <dd>{c?.n_steps} (equil {c?.equil_steps})</dd>
          <dt>particles</dt>
          <dd>{c?.n_particles}</dd>
          <dt>cutoff</dt>
          <dd>{c?.cutoff}σ {c?.tail_correction ? '+ tail' : ''}</dd>
          <dt>thermostat</dt>
          <dd>{c?.thermostat} (γ={c?.friction})</dd>
          <dt>seed</dt>
          <dd>{c?.seed}</dd>
        </dl>
      </div>
      <div>
        <h4>Observations</h4>
        {Object.entries(run.observations).map(([name, stat]) => (
          <ObsCard key={name} name={name} stat={stat} />
        ))}
      </div>
    </div>
  )
}

function RunRow({
  run,
  expanded,
  onToggle,
}: {
  run: Run
  expanded: boolean
  onToggle: () => void
}) {
  const pressure = run.observations.pressure
  return (
    <>
      <div
        className={`run-row${expanded ? ' expanded' : ''}`}
        onClick={onToggle}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') onToggle()
        }}
      >
        <span className="status-dot complete" title="complete" />
        <span className="round-tag">{roundLabel(run)}</span>
        <span className="statepoint">T {fmt(run.temperature, 2)}</span>
        <span className="statepoint">ρ {fmt(run.density, 2)}</span>
        <span className="observable">
          {pressure ? (
            <>
              P* {fmt(pressure.value, 3)} <span className="sigma">± {fmt(pressure.sigma, 3)}</span>
            </>
          ) : (
            '—'
          )}
        </span>
        <span className="frames">{run.n_frames} fr</span>
        <span className="wallclock">{fmt(run.wall_clock_s, 1)}s</span>
        <span className="chevron">▶</span>
      </div>
      {expanded && <RunDetail run={run} />}
    </>
  )
}

export function RunList({ campaign }: { campaign: CampaignDetail }) {
  const [expandedHash, setExpandedHash] = useState<string | null>(null)
  const pending = campaign.budget.n_total - campaign.n_complete

  return (
    <>
      <div className="legend">
        <span className="legend-item">
          <span className="status-dot complete" /> complete
        </span>
        <span className="legend-item">
          <span className="status-dot pending" /> pending / in progress
        </span>
      </div>
      <div className="run-list">
        {campaign.runs.length === 0 && <div className="empty">No runs recorded yet.</div>}
        {campaign.runs
          .slice()
          .reverse()
          .map((run) => (
            <RunRow
              key={run.run_hash}
              run={run}
              expanded={expandedHash === run.run_hash}
              onToggle={() => setExpandedHash(expandedHash === run.run_hash ? null : run.run_hash)}
            />
          ))}
        {pending > 0 && (
          <div className="pending-row">
            <span className="status-dot pending" />
            {pending} run{pending === 1 ? '' : 's'} remaining — chosen adaptively as the campaign proceeds
          </div>
        )}
      </div>
    </>
  )
}

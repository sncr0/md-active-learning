import { useEffect, useState } from 'react'
import { getCampaign, getCampaignTracking, listCampaigns } from './api'
import { CampaignRail, strategyLabel } from './components/CampaignRail'
import { RunList } from './components/RunList'
import { TrackingPanel } from './components/TrackingPanel'
import type { CampaignDetail, CampaignSummary, CampaignTracking } from './types'

const LIST_POLL_MS = 4000
const DETAIL_POLL_MS = 3000

export default function App() {
  const [campaigns, setCampaigns] = useState<CampaignSummary[] | null>(null)
  const [listError, setListError] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [detail, setDetail] = useState<CampaignDetail | null>(null)
  const [detailError, setDetailError] = useState<string | null>(null)
  const [tracking, setTracking] = useState<CampaignTracking | null>(null)
  const [online, setOnline] = useState(true)

  useEffect(() => {
    let cancelled = false
    async function tick() {
      try {
        const data = await listCampaigns()
        if (cancelled) return
        setCampaigns(data)
        setListError(null)
        setOnline(true)
        setSelectedId((prev) => {
          if (prev && data.some((c) => c.id === prev)) return prev
          const running = data.find((c) => c.status === 'running')
          return (running ?? data[0])?.id ?? null
        })
      } catch {
        if (cancelled) return
        setOnline(false)
        setListError('Cannot reach the dashboard API — is scripts/run_api.py running?')
      }
    }
    tick()
    const id = setInterval(tick, LIST_POLL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  const selectedSummary = campaigns?.find((c) => c.id === selectedId) ?? null

  useEffect(() => {
    if (!selectedId) {
      setDetail(null)
      setTracking(null)
      return
    }
    let cancelled = false
    async function fetchDetail() {
      try {
        const d = await getCampaign(selectedId as string)
        if (!cancelled) {
          setDetail(d)
          setDetailError(null)
        }
      } catch {
        if (!cancelled) setDetailError('Failed to load campaign detail.')
      }
      // MLflow tracking is optional and fails soft server-side — a failed
      // fetch here just means "nothing to show", not an error banner.
      try {
        const t = await getCampaignTracking(selectedId as string)
        if (!cancelled) setTracking(t)
      } catch {
        if (!cancelled) setTracking(null)
      }
    }
    fetchDetail()
    let intervalId: number | undefined
    if (selectedSummary?.status === 'running') {
      intervalId = window.setInterval(fetchDetail, DETAIL_POLL_MS)
    }
    return () => {
      cancelled = true
      if (intervalId) clearInterval(intervalId)
    }
  }, [selectedId, selectedSummary?.status])

  return (
    <>
      <header className="top">
        <p className="kicker">Molecular Dynamics × Active Learning</p>
        <h1>Campaign dashboard</h1>
        <p className="sub">
          Design of experiments for the LJ equation-of-state campaigns — which simulations have
          run, which are still in flight, and what each one measured.
        </p>
        <div className="api-state">
          <span className={`dot-live${online ? '' : ' offline'}`} />
          {online ? 'connected to dashboard API' : 'API offline'}
        </div>
      </header>

      <div className="page">
        <section className="block">
          <h2>Campaigns</h2>
          {listError && <div className="error">{listError}</div>}
          {!listError && !campaigns && <div className="loading">Loading campaigns…</div>}
          {!listError && campaigns && campaigns.length === 0 && (
            <div className="empty">
              No campaign snapshots found. Run <code>scripts/snapshot_campaign.py</code> or start a
              live campaign with <code>scripts/run_campaign.py</code>.
            </div>
          )}
          {campaigns && campaigns.length > 0 && (
            <CampaignRail campaigns={campaigns} selectedId={selectedId} onSelect={setSelectedId} />
          )}
        </section>

        {selectedSummary && (
          <section className="block">
            <h2>Simulations</h2>
            <div className="campaign-header">
              <h3>{selectedSummary.name}</h3>
              <div className="campaign-meta">
                <span>
                  strategy <b>{strategyLabel(selectedSummary.strategy)}</b>
                </span>
                <span>
                  observable <b>{selectedSummary.observable}</b>
                </span>
                {detail && (
                  <span>
                    domain{' '}
                    <b>
                      T* [{detail.domain.temperature[0]}, {detail.domain.temperature[1]}]
                    </b>{' '}
                    ×{' '}
                    <b>
                      ρ* [{detail.domain.density[0]}, {detail.domain.density[1]}]
                    </b>
                  </span>
                )}
                {detail && (
                  <span>
                    seed <b>{detail.seed}</b>
                  </span>
                )}
              </div>
            </div>
            {detailError && <div className="error">{detailError}</div>}
            {!detailError && !detail && <div className="loading">Loading runs…</div>}
            {detail && <TrackingPanel tracking={tracking} />}
            {detail && <RunList campaign={detail} />}
          </section>
        )}
      </div>

      <div className="page">
        <footer className="foot">
          md-active-learning · dashboard API queries Postgres live (src/mdal/api)
        </footer>
      </div>
    </>
  )
}

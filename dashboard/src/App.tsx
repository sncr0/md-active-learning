import { useEffect, useState } from 'react'
import { getCampaign, getCampaignTracking, listCampaigns } from './api'
import { CampaignDetailPanel } from './components/CampaignDetailPanel'
import { CampaignTable } from './components/CampaignTable'
import type { CampaignDetail, CampaignSummary, CampaignTracking } from './types'

const LIST_POLL_MS = 4000
const DETAIL_POLL_MS = 3000

export default function App() {
  const [campaigns, setCampaigns] = useState<CampaignSummary[] | null>(null)
  const [listError, setListError] = useState<string | null>(null)
  const [expandedId, setExpandedId] = useState<string | null>(null)
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

  const expandedSummary = campaigns?.find((c) => c.id === expandedId) ?? null

  function toggleExpanded(id: string) {
    setExpandedId((prev) => (prev === id ? null : id))
  }

  useEffect(() => {
    if (!expandedId) {
      setDetail(null)
      setTracking(null)
      return
    }
    let cancelled = false
    async function fetchDetail() {
      try {
        const d = await getCampaign(expandedId as string)
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
        const t = await getCampaignTracking(expandedId as string)
        if (!cancelled) setTracking(t)
      } catch {
        if (!cancelled) setTracking(null)
      }
    }
    fetchDetail()
    let intervalId: number | undefined
    if (expandedSummary?.status === 'running') {
      intervalId = window.setInterval(fetchDetail, DETAIL_POLL_MS)
    }
    return () => {
      cancelled = true
      if (intervalId) clearInterval(intervalId)
    }
  }, [expandedId, expandedSummary?.status])

  return (
    <>
      <header className="top">
        <p className="kicker">Molecular Dynamics × Active Learning</p>
        <h1>Campaign dashboard</h1>
        <p className="sub">
          Every LJ equation-of-state campaign, ranked by how well its surrogate predicts the
          reference EOS — sort and filter to compare strategies, open one for the learning curve
          and the individual runs behind it.
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
            <CampaignTable
              campaigns={campaigns}
              expandedId={expandedId}
              onToggle={toggleExpanded}
              renderDetail={() => (
                <CampaignDetailPanel detail={detail} tracking={tracking} detailError={detailError} />
              )}
            />
          )}
        </section>
      </div>

      <div className="page">
        <footer className="foot">
          md-active-learning · dashboard API queries Postgres live (src/mdal/api)
        </footer>
      </div>
    </>
  )
}

import { RunList } from './RunList'
import { TrackingPanel } from './TrackingPanel'
import type { CampaignDetail, CampaignTracking } from '../types'

export function CampaignDetailPanel({
  detail,
  tracking,
  detailError,
}: {
  detail: CampaignDetail | null
  tracking: CampaignTracking | null
  detailError: string | null
}) {
  if (detailError) return <div className="error">{detailError}</div>
  if (!detail) return <div className="loading">Loading campaign…</div>

  return (
    <div className="campaign-detail-panel">
      <div className="campaign-meta">
        <span>
          observable <b>{detail.observable}</b>
        </span>
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
        <span>
          seed <b>{detail.seed}</b>
        </span>
      </div>
      <TrackingPanel tracking={tracking} />
      <RunList campaign={detail} />
    </div>
  )
}

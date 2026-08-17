import type { CampaignDetail, CampaignSummary, CampaignTracking } from './types'

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url} -> ${res.status}`)
  return (await res.json()) as T
}

export function listCampaigns(): Promise<CampaignSummary[]> {
  return getJSON('/api/campaigns')
}

export function getCampaign(id: string): Promise<CampaignDetail> {
  return getJSON(`/api/campaigns/${encodeURIComponent(id)}`)
}

export function getCampaignTracking(id: string): Promise<CampaignTracking> {
  return getJSON(`/api/campaigns/${encodeURIComponent(id)}/tracking`)
}

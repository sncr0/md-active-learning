export type CampaignStatus = 'running' | 'complete'
export type RunStatus = 'complete' | 'failed' | 'phase_separated'

export interface Budget {
  n_initial: number
  n_total: number
  batch: number
}

export interface ObservationStat {
  value: number
  sigma: number
  n_eff: number
}

export interface RunConfigDetail {
  temperature: number
  density: number
  n_particles: number
  cutoff: number
  tail_correction: boolean
  n_steps: number
  equil_steps: number
  sample_interval: number
  timestep: number
  thermostat: string
  friction: number
  seed: number
}

export interface Run {
  run_hash: string
  index: number
  round: number
  temperature: number
  density: number
  n_steps: number
  equil_cutoff: number
  n_frames: number
  wall_clock_s: number
  status: RunStatus
  created_at: string | null
  config: RunConfigDetail
  observations: Record<string, ObservationStat>
}

export interface CampaignSummary {
  id: string
  name: string
  strategy: string
  observable: string
  status: CampaignStatus
  n_complete: number
  budget: Budget
  progress: number
  updated_at?: string
}

export interface CampaignDetail extends CampaignSummary {
  seed: number
  domain: Record<string, [number, number]>
  runs: Run[]
}

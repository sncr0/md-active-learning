"""Campaign orchestration: ties oracle + estimator + surrogate + decision + store."""

from mdal.campaign.spec import CampaignSpec
from mdal.campaign.loop import run_campaign
from mdal.campaign.cost_aware_loop import run_cost_aware_campaign

__all__ = ["CampaignSpec", "run_campaign", "run_cost_aware_campaign"]

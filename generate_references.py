"""
Generate reference data for parametrized evals.

Run with: uv run python generate_references.py

This extracts ground truth computation from response-schema evals and saves
to data/eval_references.json for use with parametrized testing.
"""

import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

from models import (
    ListEngagementInput,
    ListPurchasesInput,
    ListSubscriptionsInput,
    ListTeamsInput,
    ListUsersInput,
)
from world_runtime import World


def generate_all_references():
    """Generate reference data for all response-schema evals."""
    world = World()
    references = []

    # --- EASY EVALS ---

    # earliest_na_three_team_user
    na_users = world.list_users(ListUsersInput(region="NA")).users
    teams = world.list_teams(ListTeamsInput()).teams
    team_counts = Counter(team.user_id for team in teams)
    candidates = [user for user in na_users if team_counts[user.id] == 3]
    earliest = min(candidates, key=lambda user: user.signup_date)
    references.append({
        "id": "earliest_na_three_team_user",
        "prompt": "Among NA users who have exactly three teams, who signed up first? Provide the full name.",
        "schema_name": "Name",
        "dataset": "easy",
        "labels": ["testing"],
        "reference": {"full_name": earliest.name},
    })

    # earliest_latam_active_plan
    latam_users = world.list_users(ListUsersInput(region="LATAM")).users
    active_subs = world.list_subscriptions(ListSubscriptionsInput(status="active")).subscriptions
    active_latam_ids = {sub.user_id for sub in active_subs}
    candidates = [user for user in latam_users if user.id in active_latam_ids]
    earliest_user = min(candidates, key=lambda user: user.signup_date)
    user_active_subs = world.list_subscriptions(
        ListSubscriptionsInput(user_ids=[earliest_user.id], status="active")
    ).subscriptions
    latest_sub = max(user_active_subs, key=lambda sub: sub.started_at)
    references.append({
        "id": "earliest_latam_active_plan",
        "prompt": "For LATAM users with an active subscription, what plan belongs to the earliest signup?",
        "schema_name": "SubscriptionPlan",
        "dataset": "easy",
        "labels": ["testing"],
        "reference": {"plan": latest_sub.plan},
    })

    # eu_subscribers_with_recent_team
    cutoff_date = date(2025, 7, 1)
    eu_subscribers = world.list_users(ListUsersInput(region="EU", segment="subscriber")).users
    teams_after_cutoff = world.list_teams(ListTeamsInput(created_after=cutoff_date.isoformat())).teams
    subscriber_ids = {user.id for user in eu_subscribers}
    ids_with_recent_team = {team.user_id for team in teams_after_cutoff if team.user_id in subscriber_ids}
    references.append({
        "id": "eu_subscribers_with_recent_team",
        "prompt": "How many EU subscribers have created at least one team after 2025-07-01?",
        "schema_name": "UserCount",
        "dataset": "easy",
        "labels": ["testing"],
        "reference": {"count": len(ids_with_recent_team)},
    })

    # na_free_without_teams
    na_free = world.list_users(ListUsersInput(region="NA", segment="free")).users
    teams = world.list_teams(ListTeamsInput()).teams
    team_counts = Counter(team.user_id for team in teams)
    count = sum(1 for user in na_free if team_counts[user.id] == 0)
    references.append({
        "id": "na_free_without_teams",
        "prompt": "How many NA free users currently have zero teams?",
        "schema_name": "UserCount",
        "dataset": "easy",
        "labels": ["testing"],
        "reference": {"count": count},
    })

    # apac_whales_with_active_subs
    apac_whales = world.list_users(ListUsersInput(region="APAC", segment="whale")).users
    active_subs = world.list_subscriptions(ListSubscriptionsInput(status="active")).subscriptions
    whale_ids = {user.id for user in apac_whales}
    active_ids = {sub.user_id for sub in active_subs if sub.user_id in whale_ids}
    references.append({
        "id": "apac_whales_with_active_subs",
        "prompt": "How many APAC whales have an active subscription right now?",
        "schema_name": "UserCount",
        "dataset": "easy",
        "labels": ["testing"],
        "reference": {"count": len(active_ids)},
    })

    # latam_subscribers_with_post_sept_team
    latam_subscribers = world.list_users(ListUsersInput(region="LATAM", segment="subscriber")).users
    teams = world.list_teams(ListTeamsInput(created_after="2025-09-01")).teams
    latam_ids = {user.id for user in latam_subscribers}
    team_user_ids = {team.user_id for team in teams if team.user_id in latam_ids}
    references.append({
        "id": "latam_subscribers_with_post_sept_team",
        "prompt": "How many LATAM subscribers have created at least one team after 2025-09-01?",
        "schema_name": "UserCount",
        "dataset": "easy",
        "labels": ["testing"],
        "reference": {"count": len(team_user_ids)},
    })

    # eu_users_with_three_teams
    eu_users = world.list_users(ListUsersInput(region="EU")).users
    team_counts = Counter(team.user_id for team in world.list_teams(ListTeamsInput()).teams)
    count = sum(1 for user in eu_users if team_counts[user.id] == 3)
    references.append({
        "id": "eu_users_with_three_teams",
        "prompt": "How many EU users have exactly three teams?",
        "schema_name": "UserCount",
        "dataset": "easy",
        "labels": ["testing"],
        "reference": {"count": count},
    })

    # apac_subscribers_zero_ranked_mid_nov
    apac_subscribers = world.list_users(ListUsersInput(region="APAC", segment="subscriber")).users
    subscriber_ids = [user.id for user in apac_subscribers]
    engagement_rows = world.list_engagement(
        ListEngagementInput(user_ids=subscriber_ids, date_from="2025-11-13", date_to="2025-11-19")
    ).engagement
    ranked_totals = defaultdict(int)
    for row in engagement_rows:
        ranked_totals[row.user_id] += row.ranked_matches
    eligible_ids = [user_id for user_id in subscriber_ids if ranked_totals.get(user_id, 0) == 0]
    references.append({
        "id": "apac_subscribers_zero_ranked_mid_nov",
        "prompt": "How many APAC subscribers recorded zero ranked matches between 2025-11-13 and 2025-11-19?",
        "schema_name": "UserCount",
        "dataset": "easy",
        "labels": ["testing"],
        "reference": {"count": len(eligible_ids)},
    })

    # eu_whales_coins_pack_buyers
    eu_whales = world.list_users(ListUsersInput(region="EU", segment="whale")).users
    whale_ids = {user.id for user in eu_whales}
    purchases = world.list_purchases(ListPurchasesInput()).purchases
    buyer_ids = {p.user_id for p in purchases if p.sku == "coins_pack" and p.user_id in whale_ids}
    references.append({
        "id": "eu_whales_coins_pack_buyers",
        "prompt": "How many EU whales have purchased a coins_pack at least once?",
        "schema_name": "UserCount",
        "dataset": "easy",
        "labels": ["testing"],
        "reference": {"count": len(buyer_ids)},
    })

    # na_premium_active_subscribers_count
    na_subscribers = world.list_users(ListUsersInput(region="NA", segment="subscriber")).users
    subscriber_ids = {user.id for user in na_subscribers}
    premium_active = world.list_subscriptions(ListSubscriptionsInput(plan="premium", status="active")).subscriptions
    eligible_ids = {sub.user_id for sub in premium_active if sub.user_id in subscriber_ids}
    references.append({
        "id": "na_premium_active_subscribers_count",
        "prompt": "How many NA subscribers have an active premium subscription?",
        "schema_name": "UserCount",
        "dataset": "easy",
        "labels": ["testing"],
        "reference": {"count": len(eligible_ids)},
    })

    # latam_free_ranked_match_first_week_nov
    latam_free = world.list_users(ListUsersInput(region="LATAM", segment="free")).users
    free_ids = [user.id for user in latam_free]
    engagement_rows = world.list_engagement(
        ListEngagementInput(user_ids=free_ids, date_from="2025-11-01", date_to="2025-11-07")
    ).engagement
    ranked_totals = defaultdict(int)
    for row in engagement_rows:
        ranked_totals[row.user_id] += row.ranked_matches
    eligible_ids = [user_id for user_id in free_ids if ranked_totals.get(user_id, 0) > 0]
    references.append({
        "id": "latam_free_ranked_match_first_week_nov",
        "prompt": "During 2025-11-01 through 2025-11-07, how many LATAM free users played at least one ranked match?",
        "schema_name": "UserCount",
        "dataset": "easy",
        "labels": ["testing"],
        "reference": {"count": len(eligible_ids)},
    })

    # apac_single_subscription_users_count
    apac_users = world.list_users(ListUsersInput(region="APAC")).users
    apac_ids = [user.id for user in apac_users]
    subs = world.list_subscriptions(ListSubscriptionsInput(user_ids=apac_ids)).subscriptions
    sub_counts = Counter(sub.user_id for sub in subs)
    count = sum(1 for user_id in apac_ids if sub_counts[user_id] == 1)
    references.append({
        "id": "apac_single_subscription_users_count",
        "prompt": "How many APAC users have exactly one subscription record on file?",
        "schema_name": "UserCount",
        "dataset": "easy",
        "labels": ["testing"],
        "reference": {"count": count},
    })

    # eu_subscribers_with_canceled_subs
    eu_subscribers = world.list_users(ListUsersInput(region="EU", segment="subscriber")).users
    subscriber_ids = {user.id for user in eu_subscribers}
    canceled_subs = world.list_subscriptions(ListSubscriptionsInput(status="canceled")).subscriptions
    canceled_ids = {sub.user_id for sub in canceled_subs if sub.user_id in subscriber_ids}
    references.append({
        "id": "eu_subscribers_with_canceled_subs",
        "prompt": "How many EU subscribers have a canceled subscription on record?",
        "schema_name": "UserCount",
        "dataset": "easy",
        "labels": ["testing"],
        "reference": {"count": len(canceled_ids)},
    })

    # user_00020_team_total
    teams = world.list_teams(ListTeamsInput(user_ids=["user_00020"])).teams
    references.append({
        "id": "user_00020_team_total",
        "prompt": "How many teams does user_00020 have?",
        "schema_name": "TeamTotal",
        "dataset": "easy",
        "labels": ["testing"],
        "reference": {"team_count": len(teams)},
    })

    # user_00005_purchase_total
    purchases = world.list_purchases(ListPurchasesInput(user_ids=["user_00005"])).purchases
    references.append({
        "id": "user_00005_purchase_total",
        "prompt": "How many purchases has user_00005 made in total?",
        "schema_name": "PurchaseCount",
        "dataset": "easy",
        "labels": ["testing"],
        "reference": {"purchase_count": len(purchases)},
    })

    # user_00005_sessions_on_2025_11_19
    rows = world.list_engagement(
        ListEngagementInput(user_ids=["user_00005"], date_from="2025-11-19", date_to="2025-11-19")
    ).engagement
    total_sessions = sum(row.sessions for row in rows)
    references.append({
        "id": "user_00005_sessions_on_2025_11_19",
        "prompt": "How many sessions did user_00005 log on 2025-11-19?",
        "schema_name": "SessionsTotal",
        "dataset": "easy",
        "labels": ["testing"],
        "reference": {"sessions": total_sessions},
    })

    # --- MEDIUM EVALS ---

    # count_ultra_subs_by_region
    latam_users = world.list_users(ListUsersInput(region="LATAM")).users
    latam_user_ids = {user.id for user in latam_users}
    ultra_subs = world.list_subscriptions(ListSubscriptionsInput(plan="ultra", status="active")).subscriptions
    ultra_user_ids = {sub.user_id for sub in ultra_subs}
    teams = world.list_teams(ListTeamsInput()).teams
    team_counts = Counter(team.user_id for team in teams)
    eligible_ids = {user_id for user_id in latam_user_ids & ultra_user_ids if team_counts[user_id] >= 2}
    references.append({
        "id": "count_ultra_subs_by_region",
        "prompt": "How many LATAM users have an active ultra subscription and at least two teams?",
        "schema_name": "UserCount",
        "dataset": "medium",
        "labels": ["testing"],
        "reference": {"count": len(eligible_ids)},
    })

    # na_whale_lowest_ranked_matches
    whales = world.list_users(ListUsersInput(region="NA", segment="whale")).users
    window_start = date(2025, 11, 10)
    window_end = date(2025, 11, 19)
    whale_ids = [user.id for user in whales]
    engagement_rows = world.list_engagement(
        ListEngagementInput(user_ids=whale_ids, date_from=window_start.isoformat(), date_to=window_end.isoformat())
    ).engagement
    ranked_totals = defaultdict(int)
    for row in engagement_rows:
        ranked_totals[row.user_id] += row.ranked_matches
    all_ranked = {user.id: ranked_totals.get(user.id, 0) for user in whales}
    min_ranked = min(all_ranked.values())
    lowest_users = [user for user in whales if all_ranked[user.id] == min_ranked]
    earliest_user = min(lowest_users, key=lambda user: user.signup_date)
    references.append({
        "id": "na_whale_lowest_ranked_matches",
        "prompt": "Between 2025-11-10 and 2025-11-19 inclusive, which NA whale recorded the fewest ranked matches? If multiple users tie, return the one who signed up first. Provide the user id and full name.",
        "schema_name": "UserIdentity",
        "dataset": "medium",
        "labels": ["testing"],
        "reference": {"user_id": earliest_user.id, "full_name": earliest_user.name},
    })

    # apac_top_spender_after_sept
    apac_users = world.list_users(ListUsersInput(region="APAC")).users
    apac_lookup = {user.id: user for user in apac_users}
    purchases = world.list_purchases(ListPurchasesInput(purchased_after="2025-09-01")).purchases
    spend_totals = defaultdict(float)
    for purchase in purchases:
        if purchase.user_id in apac_lookup:
            spend_totals[purchase.user_id] += purchase.amount
    max_spend = max(spend_totals.values())
    leaders = [user_id for user_id, total in spend_totals.items() if total == max_spend]
    top_user = min((apac_lookup[user_id] for user_id in leaders), key=lambda user: user.signup_date)
    references.append({
        "id": "apac_top_spender_after_sept",
        "prompt": "What is the total amount spent after 2025-09-01 by the APAC user who spent the most in that period? Provide just the numeric total.",
        "schema_name": "SpendAmount",
        "dataset": "medium",
        "labels": ["testing"],
        "reference": {"total_amount": spend_totals[top_user.id]},
    })

    # na_free_top_sessions_window
    na_free = world.list_users(ListUsersInput(region="NA", segment="free")).users
    na_lookup = {user.id: user for user in na_free}
    engagement_rows = world.list_engagement(
        ListEngagementInput(user_ids=list(na_lookup.keys()), date_from="2025-11-10", date_to="2025-11-19")
    ).engagement
    session_totals = defaultdict(int)
    for row in engagement_rows:
        session_totals[row.user_id] += row.sessions
    max_sessions = max(session_totals.values())
    leaders = [na_lookup[user_id] for user_id, total in session_totals.items() if total == max_sessions]
    top_user = min(leaders, key=lambda user: user.signup_date)
    references.append({
        "id": "na_free_top_sessions_window",
        "prompt": "Between 2025-11-10 and 2025-11-19, which NA free user logged the most sessions? If multiple users tie, return the one who signed up first. Provide the user id and total sessions.",
        "schema_name": "UserSessions",
        "dataset": "medium",
        "labels": ["testing"],
        "reference": {"user_id": top_user.id, "total_sessions": session_totals[top_user.id]},
    })

    # latam_whale_top_minutes
    latam_whales = world.list_users(ListUsersInput(region="LATAM", segment="whale")).users
    whale_lookup = {user.id: user for user in latam_whales}
    engagement_rows = world.list_engagement(
        ListEngagementInput(user_ids=list(whale_lookup.keys()), date_from="2025-11-01", date_to="2025-11-19")
    ).engagement
    minute_totals = defaultdict(int)
    for row in engagement_rows:
        minute_totals[row.user_id] += row.minutes_played
    max_minutes = max(minute_totals.values())
    leaders = [whale_lookup[user_id] for user_id, total in minute_totals.items() if total == max_minutes]
    top_user = min(leaders, key=lambda user: user.signup_date)
    references.append({
        "id": "latam_whale_top_minutes",
        "prompt": "Between 2025-11-01 and 2025-11-19, which LATAM whale logged the most total minutes? If multiple users tie, return the one who signed up first. Provide the user id and total minutes.",
        "schema_name": "UserMinutes",
        "dataset": "medium",
        "labels": ["testing"],
        "reference": {"user_id": top_user.id, "total_minutes": minute_totals[top_user.id]},
    })

    # apac_active_premium_coins_after_sept15_count
    apac_users = world.list_users(ListUsersInput(region="APAC")).users
    apac_ids = {user.id for user in apac_users}
    active_premium = world.list_subscriptions(ListSubscriptionsInput(plan="premium", status="active")).subscriptions
    premium_ids = {sub.user_id for sub in active_premium if sub.user_id in apac_ids}
    purchases = world.list_purchases(ListPurchasesInput(purchased_after="2025-09-15")).purchases
    buyer_ids = {p.user_id for p in purchases if p.sku == "coins_pack" and p.user_id in premium_ids}
    references.append({
        "id": "apac_active_premium_coins_after_sept15_count",
        "prompt": "How many APAC users with an active premium subscription have bought a coins_pack after 2025-09-15?",
        "schema_name": "UserCount",
        "dataset": "medium",
        "labels": ["testing"],
        "reference": {"count": len(buyer_ids)},
    })

    # eu_active_premium_with_purchase_count
    eu_users = world.list_users(ListUsersInput(region="EU")).users
    eu_ids = {user.id for user in eu_users}
    premium_active = world.list_subscriptions(ListSubscriptionsInput(plan="premium", status="active")).subscriptions
    eligible_ids = {sub.user_id for sub in premium_active if sub.user_id in eu_ids}
    purchases = world.list_purchases(ListPurchasesInput()).purchases
    purchasers = {p.user_id for p in purchases if p.user_id in eligible_ids}
    references.append({
        "id": "eu_active_premium_with_purchase_count",
        "prompt": "How many EU users have an active premium subscription and at least one purchase?",
        "schema_name": "UserCount",
        "dataset": "medium",
        "labels": ["testing"],
        "reference": {"count": len(purchasers)},
    })

    # latam_subscriber_top_ranked_mid_nov
    latam_subscribers = world.list_users(ListUsersInput(region="LATAM", segment="subscriber")).users
    subscriber_lookup = {user.id: user for user in latam_subscribers}
    engagement_rows = world.list_engagement(
        ListEngagementInput(user_ids=list(subscriber_lookup.keys()), date_from="2025-11-13", date_to="2025-11-19")
    ).engagement
    ranked_totals = defaultdict(int)
    for row in engagement_rows:
        ranked_totals[row.user_id] += row.ranked_matches
    max_ranked = max(ranked_totals.values())
    leaders = [subscriber_lookup[user_id] for user_id, total in ranked_totals.items() if total == max_ranked]
    top_user = min(leaders, key=lambda user: user.signup_date)
    references.append({
        "id": "latam_subscriber_top_ranked_mid_nov",
        "prompt": "Between 2025-11-13 and 2025-11-19, which LATAM subscriber played the most ranked matches? Provide the user id and full name.",
        "schema_name": "UserIdentity",
        "dataset": "medium",
        "labels": ["testing"],
        "reference": {"user_id": top_user.id, "full_name": top_user.name},
    })

    # apac_ultra_recent_team_count
    apac_users = world.list_users(ListUsersInput(region="APAC")).users
    apac_ids = {user.id for user in apac_users}
    active_ultra = world.list_subscriptions(ListSubscriptionsInput(plan="ultra", status="active")).subscriptions
    ultra_ids = {sub.user_id for sub in active_ultra if sub.user_id in apac_ids}
    teams = world.list_teams(ListTeamsInput(user_ids=list(ultra_ids), created_after="2025-10-01")).teams
    user_ids_with_recent_team = {team.user_id for team in teams}
    references.append({
        "id": "apac_ultra_recent_team_count",
        "prompt": "How many APAC users with an active ultra subscription created at least one team after 2025-10-01?",
        "schema_name": "UserCount",
        "dataset": "medium",
        "labels": ["testing"],
        "reference": {"count": len(user_ids_with_recent_team)},
    })

    # most_recent_team_name_user_00024
    teams = world.list_teams(ListTeamsInput(user_ids=["user_00024"])).teams
    latest_team = max(teams, key=lambda team: team.created_at)
    references.append({
        "id": "most_recent_team_name_user_00024",
        "prompt": "What is the name of the most recently created team for user_00024?",
        "schema_name": "TeamName",
        "dataset": "medium",
        "labels": ["testing"],
        "reference": {"team_name": latest_team.name},
    })

    # earliest_latam_subscriber_purchase_count_after_june
    latam_subscribers = world.list_users(ListUsersInput(region="LATAM", segment="subscriber")).users
    earliest_user = min(latam_subscribers, key=lambda user: user.signup_date)
    purchases = world.list_purchases(
        ListPurchasesInput(user_ids=[earliest_user.id], purchased_after="2025-06-01")
    ).purchases
    references.append({
        "id": "earliest_latam_subscriber_purchase_count_after_june",
        "prompt": "For the earliest LATAM subscriber by signup date, how many purchases were made after 2025-06-01?",
        "schema_name": "PurchaseCount",
        "dataset": "medium",
        "labels": ["testing"],
        "reference": {"purchase_count": len(purchases)},
    })

    # top_sku_latam_subscribers
    latam_subscribers = world.list_users(ListUsersInput(region="LATAM", segment="subscriber")).users
    latam_ids = {user.id for user in latam_subscribers}
    purchases = world.list_purchases(ListPurchasesInput()).purchases
    sku_counts = Counter(p.sku for p in purchases if p.user_id in latam_ids)
    max_count = max(sku_counts.values())
    top_skus = sorted([sku for sku, count in sku_counts.items() if count == max_count])
    top_sku = top_skus[0]
    references.append({
        "id": "top_sku_latam_subscribers",
        "prompt": "Which SKU has the highest number of purchases among LATAM subscribers? Provide only the SKU.",
        "schema_name": "SkuName",
        "dataset": "medium",
        "labels": ["testing"],
        "reference": {"sku": top_sku},
    })

    # apac_whale_ranked_matches_for_top_sessions
    apac_whales = world.list_users(ListUsersInput(region="APAC", segment="whale")).users
    whale_lookup = {user.id: user for user in apac_whales}
    engagement_rows = world.list_engagement(
        ListEngagementInput(user_ids=list(whale_lookup.keys()), date_from="2025-11-13", date_to="2025-11-19")
    ).engagement
    session_totals = defaultdict(int)
    ranked_totals = defaultdict(int)
    for row in engagement_rows:
        session_totals[row.user_id] += row.sessions
        ranked_totals[row.user_id] += row.ranked_matches
    max_sessions = max(session_totals.values())
    leaders = [whale_lookup[user_id] for user_id, total in session_totals.items() if total == max_sessions]
    top_user = min(leaders, key=lambda user: user.signup_date)
    references.append({
        "id": "apac_whale_ranked_matches_for_top_sessions",
        "prompt": "Among APAC whales, take the user with the most sessions between 2025-11-13 and 2025-11-19. If multiple users tie, use the one who signed up first. What is their total ranked matches in that same window?",
        "schema_name": "RankedMatchesTotal",
        "dataset": "medium",
        "labels": ["testing"],
        "reference": {"ranked_matches": ranked_totals[top_user.id]},
    })

    # --- HARD EVALS ---

    # latam_ultra_zero_ranked
    latam_users = world.list_users(ListUsersInput(region="LATAM")).users
    ultra_active = world.list_subscriptions(ListSubscriptionsInput(plan="ultra", status="active")).subscriptions
    ultra_latam_ids = {sub.user_id for sub in ultra_active if sub.user_id in {user.id for user in latam_users}}
    window_start = date(2025, 11, 13)
    window_end = date(2025, 11, 19)
    engagement_rows = world.list_engagement(
        ListEngagementInput(user_ids=list(ultra_latam_ids), date_from=window_start.isoformat(), date_to=window_end.isoformat())
    ).engagement
    ranked_totals = defaultdict(int)
    for row in engagement_rows:
        ranked_totals[row.user_id] += row.ranked_matches
    eligible_ids = [user_id for user_id in ultra_latam_ids if ranked_totals.get(user_id, 0) == 0]
    references.append({
        "id": "latam_ultra_zero_ranked",
        "prompt": "How many LATAM users with an active ultra subscription recorded zero ranked matches between 2025-11-13 and 2025-11-19?",
        "schema_name": "UserCount",
        "dataset": "hard",
        "labels": ["testing"],
        "reference": {"count": len(eligible_ids)},
    })

    # eu_ultra_whale_earliest_post_oct_team
    eu_whales = world.list_users(ListUsersInput(region="EU", segment="whale")).users
    whale_lookup = {user.id: user for user in eu_whales}
    active_ultra = world.list_subscriptions(ListSubscriptionsInput(plan="ultra", status="active")).subscriptions
    eligible_ids = {sub.user_id for sub in active_ultra if sub.user_id in whale_lookup}
    teams = world.list_teams(ListTeamsInput(user_ids=list(eligible_ids), created_after="2025-10-01")).teams
    earliest_team = min(teams, key=lambda team: team.created_at)
    owner = whale_lookup[earliest_team.user_id]
    references.append({
        "id": "eu_ultra_whale_earliest_post_oct_team",
        "prompt": "Which EU whale with an active ultra subscription created the earliest team after 2025-10-01? Provide the user id and full name.",
        "schema_name": "UserIdentity",
        "dataset": "hard",
        "labels": ["testing"],
        "reference": {"user_id": owner.id, "full_name": owner.name},
    })

    # latam_subscriber_highest_spend_two_teams
    latam_subscribers = world.list_users(ListUsersInput(region="LATAM", segment="subscriber")).users
    subscriber_lookup = {user.id: user for user in latam_subscribers}
    teams = world.list_teams(ListTeamsInput(user_ids=list(subscriber_lookup.keys()))).teams
    team_counts = Counter(team.user_id for team in teams)
    eligible_ids = {user_id for user_id, count in team_counts.items() if count >= 2}
    purchases = world.list_purchases(ListPurchasesInput(user_ids=list(eligible_ids))).purchases
    spend_totals = defaultdict(float)
    for purchase in purchases:
        spend_totals[purchase.user_id] += purchase.amount
    max_spend = max(spend_totals.values())
    leaders = [subscriber_lookup[user_id] for user_id, total in spend_totals.items() if total == max_spend]
    top_user = min(leaders, key=lambda user: user.signup_date)
    references.append({
        "id": "latam_subscriber_highest_spend_two_teams",
        "prompt": "Among LATAM subscribers with at least two teams, who has the highest total purchase amount? If multiple users tie, return the one who signed up first. Provide the user id and total amount.",
        "schema_name": "UserAmount",
        "dataset": "hard",
        "labels": ["testing"],
        "reference": {"user_id": top_user.id, "total_amount": spend_totals[top_user.id]},
    })

    # apac_whale_zero_ranked_with_purchase
    apac_whales = world.list_users(ListUsersInput(region="APAC", segment="whale")).users
    whale_lookup = {user.id: user for user in apac_whales}
    purchases = world.list_purchases(ListPurchasesInput(purchased_after="2025-10-01")).purchases
    buyers = {p.user_id for p in purchases if p.user_id in whale_lookup}
    engagement_rows = world.list_engagement(
        ListEngagementInput(user_ids=list(whale_lookup.keys()), date_from="2025-11-15", date_to="2025-11-19")
    ).engagement
    ranked_totals = defaultdict(int)
    for row in engagement_rows:
        ranked_totals[row.user_id] += row.ranked_matches
    eligible_users = [user for user in apac_whales if user.id in buyers and ranked_totals.get(user.id, 0) == 0]
    earliest_user = min(eligible_users, key=lambda user: user.signup_date)
    references.append({
        "id": "apac_whale_zero_ranked_with_purchase",
        "prompt": "Among APAC whales with at least one purchase after 2025-10-01 and zero ranked matches between 2025-11-15 and 2025-11-19, who signed up first? Provide the user id and full name.",
        "schema_name": "UserIdentity",
        "dataset": "hard",
        "labels": ["testing"],
        "reference": {"user_id": earliest_user.id, "full_name": earliest_user.name},
    })

    # eu_premium_highest_average_minutes
    eu_users = world.list_users(ListUsersInput(region="EU")).users
    eu_lookup = {user.id: user for user in eu_users}
    premium_active = world.list_subscriptions(ListSubscriptionsInput(plan="premium", status="active")).subscriptions
    eligible_ids = {sub.user_id for sub in premium_active if sub.user_id in eu_lookup}
    engagement_rows = world.list_engagement(
        ListEngagementInput(user_ids=list(eligible_ids), date_from="2025-11-10", date_to="2025-11-19")
    ).engagement
    minutes_totals = defaultdict(int)
    session_totals = defaultdict(int)
    for row in engagement_rows:
        minutes_totals[row.user_id] += row.minutes_played
        session_totals[row.user_id] += row.sessions
    averages = {
        user_id: minutes_totals[user_id] / session_totals[user_id]
        for user_id in eligible_ids
        if session_totals[user_id] > 0
    }
    max_avg = max(averages.values())
    leaders = [eu_lookup[user_id] for user_id, avg in averages.items() if avg == max_avg]
    top_user = min(leaders, key=lambda user: user.signup_date)
    references.append({
        "id": "eu_premium_highest_average_minutes",
        "prompt": "Which EU user with an active premium subscription had the highest average minutes per session between 2025-11-10 and 2025-11-19? Provide the user id and full name.",
        "schema_name": "UserIdentity",
        "dataset": "hard",
        "labels": ["testing"],
        "reference": {"user_id": top_user.id, "full_name": top_user.name},
    })

    # latam_single_team_top_spender
    latam_subscribers = world.list_users(ListUsersInput(region="LATAM", segment="subscriber")).users
    subscriber_lookup = {user.id: user for user in latam_subscribers}
    teams = world.list_teams(ListTeamsInput(user_ids=list(subscriber_lookup.keys()))).teams
    team_counts = Counter(team.user_id for team in teams)
    eligible_ids = {user_id for user_id, count in team_counts.items() if count == 1}
    purchases = world.list_purchases(ListPurchasesInput(user_ids=list(eligible_ids))).purchases
    spend_totals = defaultdict(float)
    for purchase in purchases:
        spend_totals[purchase.user_id] += purchase.amount
    max_spend = max(spend_totals.values())
    leaders = [subscriber_lookup[user_id] for user_id, total in spend_totals.items() if total == max_spend]
    top_user = min(leaders, key=lambda user: user.signup_date)
    references.append({
        "id": "latam_single_team_top_spender",
        "prompt": "Among LATAM subscribers with exactly one team, who has the highest total purchase amount? If multiple users tie, return the one who signed up first. Provide the user id and total amount.",
        "schema_name": "UserAmount",
        "dataset": "hard",
        "labels": ["testing"],
        "reference": {"user_id": top_user.id, "total_amount": spend_totals[top_user.id]},
    })

    # purchase_count_top_buyer_after_oct
    purchases = world.list_purchases(ListPurchasesInput(purchased_after="2025-10-01")).purchases
    purchase_counts = Counter(p.user_id for p in purchases)
    max_count = max(purchase_counts.values())
    references.append({
        "id": "purchase_count_top_buyer_after_oct",
        "prompt": "How many purchases did the top buyer make after 2025-10-01?",
        "schema_name": "PurchaseCount",
        "dataset": "hard",
        "labels": ["testing"],
        "reference": {"purchase_count": max_count},
    })

    # na_whale_highest_total_minutes
    na_whales = world.list_users(ListUsersInput(region="NA", segment="whale")).users
    whale_lookup = {user.id: user for user in na_whales}
    engagement_rows = world.list_engagement(ListEngagementInput(user_ids=list(whale_lookup.keys()))).engagement
    minute_totals = defaultdict(int)
    for row in engagement_rows:
        minute_totals[row.user_id] += row.minutes_played
    max_minutes = max(minute_totals.values())
    leaders = [whale_lookup[user_id] for user_id, total in minute_totals.items() if total == max_minutes]
    top_user = min(leaders, key=lambda user: user.signup_date)
    references.append({
        "id": "na_whale_highest_total_minutes",
        "prompt": "Across all engagement data, which NA whale accumulated the most total minutes played? If multiple users tie, return the one who signed up first. Provide the user id and total minutes.",
        "schema_name": "UserMinutes",
        "dataset": "hard",
        "labels": ["testing"],
        "reference": {"user_id": top_user.id, "total_minutes": minute_totals[top_user.id]},
    })

    # apac_active_subscribers_three_pre_oct_teams
    apac_subscribers = world.list_users(ListUsersInput(region="APAC", segment="subscriber")).users
    subscriber_ids = {user.id for user in apac_subscribers}
    active_subs = world.list_subscriptions(ListSubscriptionsInput(status="active", user_ids=list(subscriber_ids))).subscriptions
    active_ids = {sub.user_id for sub in active_subs}
    teams = world.list_teams(ListTeamsInput(user_ids=list(active_ids))).teams
    pre_oct_counts = Counter(team.user_id for team in teams if team.created_at < date(2025, 10, 1))
    eligible_ids = [user_id for user_id in active_ids if pre_oct_counts.get(user_id, 0) == 3]
    references.append({
        "id": "apac_active_subscribers_three_pre_oct_teams",
        "prompt": "How many APAC subscribers have an active subscription and exactly three teams created before 2025-10-01?",
        "schema_name": "UserCount",
        "dataset": "hard",
        "labels": ["testing"],
        "reference": {"count": len(eligible_ids)},
    })

    return references


if __name__ == "__main__":
    references = generate_all_references()

    output_path = Path("data/eval_references.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(references, f, indent=2)

    print(f"Generated {len(references)} reference entries")
    print(f"Saved to {output_path}")

    # Summary by dataset
    by_dataset = {}
    for ref in references:
        ds = ref["dataset"]
        by_dataset[ds] = by_dataset.get(ds, 0) + 1

    for ds, count in sorted(by_dataset.items()):
        print(f"  {ds}: {count}")

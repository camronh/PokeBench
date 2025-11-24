from datetime import date as DateType, datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


# =========================
# Core data models (world)
# =========================

class User(BaseModel):
    id: str = Field(..., description="Unique identifier for the user")
    name: str = Field(..., description="Display name of the user")
    email: str = Field(..., description="Email address of the user")
    region: str = Field(..., description="Geographic region of the user, for example NA or EU")
    signup_date: DateType = Field(..., description="Date when the user signed up")
    segment: str = Field(..., description="Business segment for the user, for example free, subscriber, whale")
    admin_note: str = Field(
        default="",
        description="Free text note that admins or the agent can write about the user"
    )


class Subscription(BaseModel):
    id: str = Field(..., description="Unique identifier for the subscription record")
    user_id: str = Field(..., description="ID of the user that owns this subscription")
    plan: str = Field(..., description="Name of the plan, for example free, premium, ultra")
    status: str = Field(..., description="Current status of the subscription, for example active or canceled")
    started_at: DateType = Field(..., description="Date when the subscription started")
    ended_at: Optional[DateType] = Field(
        default=None,
        description="Date when the subscription ended, if it is no longer active"
    )


class UserFlag(BaseModel):
    id: str = Field(..., description="Unique identifier for the flag")
    user_id: str = Field(..., description="ID of the user that this flag applies to")
    flag_type: str = Field(..., description="Type of flag, for example churn_risk, whale, vip_support")
    reason: str = Field(..., description="Human readable reason explaining why this flag was created")
    created_at: datetime = Field(..., description="Timestamp when this flag was created")


class Pokemon(BaseModel):
    id: str = Field(..., description="Unique identifier for the Pokemon species, for example charizard")
    name: str = Field(..., description="Human readable Pokemon name, for example Charizard")
    primary_type: str = Field(..., description="Primary elemental type of the Pokemon, for example Fire")
    secondary_type: Optional[str] = Field(
        default=None,
        description="Optional secondary type of the Pokemon if it has one"
    )


class Team(BaseModel):
    id: str = Field(..., description="Unique identifier for the team")
    user_id: str = Field(..., description="ID of the user that owns this team")
    name: str = Field(..., description="Name of the team, for example Ranked OU team")
    created_at: DateType = Field(..., description="Date when the team was created")
    pokemon_ids: List[str] = Field(
        ...,
        description="List of Pokemon species IDs that are on this team"
    )


class Purchase(BaseModel):
    id: str = Field(..., description="Unique identifier for the purchase record")
    user_id: str = Field(..., description="ID of the user that made this purchase")
    pokemon_id: Optional[str] = Field(
        default=None,
        description="Optional Pokemon species ID that this purchase is associated with, if any"
    )
    sku: str = Field(..., description="Identifier of the product or bundle that was purchased")
    amount: float = Field(..., description="Monetary amount of the purchase in the chosen currency")
    purchased_at: datetime = Field(..., description="Timestamp when the purchase was made")


class EngagementRow(BaseModel):
    id: str = Field(..., description="Unique identifier for the engagement record")
    user_id: str = Field(..., description="ID of the user that this engagement row belongs to")
    date: DateType = Field(..., description="Calendar date for these engagement metrics")
    sessions: int = Field(..., description="Number of sessions the user had on this date")
    minutes_played: int = Field(..., description="Total minutes played by the user on this date")
    ranked_matches: int = Field(..., description="Number of ranked matches played on this date")


class Message(BaseModel):
    id: str = Field(..., description="Unique identifier for the message")
    channel: str = Field(..., description="Channel name where this message was posted, for example #ops-alerts")
    text: str = Field(..., description="Text content of the message")
    created_at: datetime = Field(..., description="Timestamp when this message was created")


# =========================
# Tool IO schemas
# =========================

# 1) list_users

class ListUsersInput(BaseModel):
    user_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific user IDs to include. If omitted, all users are considered."
    )
    segment: Optional[str] = Field(
        default=None,
        description="Optional user segment filter, for example free, subscriber or whale"
    )
    region: Optional[str] = Field(
        default=None,
        description="Optional region filter, for example NA or EU"
    )
    signed_up_after: Optional[str] = Field(
        default=None,
        description="If provided, only include users whose signup_date is after this date (ISO format, e.g. '2025-09-01')"
    )


class ListUsersOutput(BaseModel):
    users: List[User] = Field(
        ...,
        description="List of users matching the query filters"
    )


# 2) list_subscriptions

class ListSubscriptionsInput(BaseModel):
    user_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of user IDs to filter subscriptions by"
    )
    plan: Optional[str] = Field(
        default=None,
        description="Optional plan name filter, for example free, premium or ultra"
    )
    status: Optional[str] = Field(
        default=None,
        description="Optional status filter, for example active or canceled"
    )


class ListSubscriptionsOutput(BaseModel):
    subscriptions: List[Subscription] = Field(
        ...,
        description="List of subscriptions matching the query filters"
    )


# 3) bulk_update_user_notes

class UserNoteUpdate(BaseModel):
    user_id: str = Field(..., description="ID of the user whose admin note should be updated")
    note: str = Field(..., description="New admin note text to set for this user")


class BulkUpdateUserNotesInput(BaseModel):
    updates: List[UserNoteUpdate] = Field(
        ...,
        description="List of note updates to apply to multiple users"
    )


class BulkUpdateUserNotesOutput(BaseModel):
    users: List[User] = Field(
        ...,
        description="List of users after their admin notes have been updated"
    )


# 4) create_user_flags

class UserFlagCreate(BaseModel):
    user_id: str = Field(..., description="ID of the user to create a flag for")
    flag_type: str = Field(..., description="Type of flag to create, for example churn_risk or vip_support")
    reason: str = Field(..., description="Human readable reason explaining why this flag is being created")


class CreateUserFlagsInput(BaseModel):
    flags: List[UserFlagCreate] = Field(
        ...,
        description="List of flag specifications to create for users"
    )


class CreateUserFlagsOutput(BaseModel):
    flags: List[UserFlag] = Field(
        ...,
        description="List of UserFlag records that were created"
    )


# 5) list_teams

class ListTeamsInput(BaseModel):
    user_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of user IDs to filter teams by owner"
    )
    created_after: Optional[str] = Field(
        default=None,
        description="If provided, only include teams created after this date (ISO format, e.g. '2025-07-01')"
    )


class ListTeamsOutput(BaseModel):
    teams: List[Team] = Field(
        ...,
        description="List of teams matching the query filters"
    )


# 6) list_pokemon

class ListPokemonInput(BaseModel):
    pokemon_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of Pokemon species IDs to fetch. If omitted, all Pokemon may be returned."
    )


class ListPokemonOutput(BaseModel):
    pokemon: List[Pokemon] = Field(
        ...,
        description="List of Pokemon species records matching the query filters"
    )


# 7) list_purchases

class ListPurchasesInput(BaseModel):
    user_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of user IDs to filter purchases by owner"
    )
    purchased_after: Optional[str] = Field(
        default=None,
        description="If provided, only include purchases made after this timestamp (ISO format, e.g. '2025-09-01' or '2025-09-01T00:00:00')"
    )


class ListPurchasesOutput(BaseModel):
    purchases: List[Purchase] = Field(
        ...,
        description="List of purchases matching the query filters"
    )


# 8) list_engagement

class ListEngagementInput(BaseModel):
    user_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of user IDs to filter engagement rows by"
    )
    date_from: Optional[str] = Field(
        default=None,
        description="If provided, only include engagement rows on or after this date (ISO format, e.g. '2025-11-06')"
    )
    date_to: Optional[str] = Field(
        default=None,
        description="If provided, only include engagement rows on or before this date (ISO format, e.g. '2025-11-19')"
    )


class ListEngagementOutput(BaseModel):
    engagement: List[EngagementRow] = Field(
        ...,
        description="List of engagement records matching the query filters"
    )


# 9) post_message

class PostMessageInput(BaseModel):
    channel: str = Field(..., description="Name of the channel to post the message in, for example #crm-campaigns")
    text: str = Field(..., description="Text content of the message to post")


class PostMessageOutput(BaseModel):
    message: Message = Field(
        ...,
        description="The message record that was created and posted"
    )


# 10) list_messages

class ListMessagesInput(BaseModel):
    channel: str = Field(..., description="Channel name to fetch messages from")
    limit: int = Field(
        default=50,
        description="Maximum number of recent messages to return from the channel"
    )


class ListMessagesOutput(BaseModel):
    messages: List[Message] = Field(
        ...,
        description="List of messages from the requested channel, ordered most recent first"
    )

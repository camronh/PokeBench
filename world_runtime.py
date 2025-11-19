from typing import Callable, Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from langchain_core.tools import StructuredTool

from models import (
    User,
    Subscription,
    UserFlag,
    Pokemon,
    Team,
    Purchase,
    EngagementRow,
    Message,
    ListUsersInput,
    ListUsersOutput,
    ListSubscriptionsInput,
    ListSubscriptionsOutput,
    BulkUpdateUserNotesInput,
    BulkUpdateUserNotesOutput,
    CreateUserFlagsInput,
    CreateUserFlagsOutput,
    ListTeamsInput,
    ListTeamsOutput,
    ListPokemonInput,
    ListPokemonOutput,
    ListPurchasesInput,
    ListPurchasesOutput,
    ListEngagementInput,
    ListEngagementOutput,
    PostMessageInput,
    PostMessageOutput,
    ListMessagesInput,
    ListMessagesOutput,
)


class World(BaseModel):
    """
    In-memory world state with tool methods.
    Auto-loads from data/world_seed.json on instantiation.
    """

    users: Dict[str, User] = Field(
        default_factory=dict,
        description="Mapping from user ID to User objects in the world"
    )
    subscriptions: Dict[str, Subscription] = Field(
        default_factory=dict,
        description="Mapping from subscription ID to Subscription objects in the world"
    )
    flags: Dict[str, UserFlag] = Field(
        default_factory=dict,
        description="Mapping from flag ID to UserFlag objects in the world"
    )
    pokemon: Dict[str, Pokemon] = Field(
        default_factory=dict,
        description="Mapping from Pokemon species ID to Pokemon objects in the world"
    )
    teams: Dict[str, Team] = Field(
        default_factory=dict,
        description="Mapping from team ID to Team objects in the world"
    )
    purchases: Dict[str, Purchase] = Field(
        default_factory=dict,
        description="Mapping from purchase ID to Purchase objects in the world"
    )
    engagement: Dict[str, EngagementRow] = Field(
        default_factory=dict,
        description="Mapping from engagement row ID to EngagementRow objects in the world"
    )
    messages: List[Message] = Field(
        default_factory=list,
        description="List of all messages posted in internal channels"
    )

    def __init__(self, **data):
        """
        Load world from data/world_seed.json and make a deep copy.
        """
        if not data:
            # Auto-load from seed file
            world_seed_path = Path(__file__).parent / "data" / "world_seed.json"
            seed_data = self.model_validate_json(world_seed_path.read_text())
            # Make a deep copy
            data = seed_data.model_dump()
        super().__init__(**data)

    # ---------- Tool implementations ----------

    def list_users(self, args: ListUsersInput) -> ListUsersOutput:
        users = list(self.users.values())

        if args.user_ids is not None:
            ids = set(args.user_ids)
            users = [u for u in users if u.id in ids]

        if args.segment is not None:
            users = [u for u in users if u.segment == args.segment]

        if args.region is not None:
            users = [u for u in users if u.region == args.region]

        if args.signed_up_after is not None:
            users = [u for u in users if u.signup_date > args.signed_up_after]

        return ListUsersOutput(users=users)

    def list_subscriptions(self, args: ListSubscriptionsInput) -> ListSubscriptionsOutput:
        subs = list(self.subscriptions.values())

        if args.user_ids is not None:
            ids = set(args.user_ids)
            subs = [s for s in subs if s.user_id in ids]

        if args.plan is not None:
            subs = [s for s in subs if s.plan == args.plan]

        if args.status is not None:
            subs = [s for s in subs if s.status == args.status]

        return ListSubscriptionsOutput(subscriptions=subs)

    def bulk_update_user_notes(self, args: BulkUpdateUserNotesInput) -> BulkUpdateUserNotesOutput:
        updated = []
        for upd in args.updates:
            user = self.users.get(upd.user_id)
            if user is None:
                continue
            user = user.model_copy(update={"admin_note": upd.note})
            self.users[user.id] = user
            updated.append(user)
        return BulkUpdateUserNotesOutput(users=updated)

    def create_user_flags(self, args: CreateUserFlagsInput) -> CreateUserFlagsOutput:
        created = []
        now = datetime.utcnow()
        for spec in args.flags:
            flag = UserFlag(
                id=str(uuid4()),
                user_id=spec.user_id,
                flag_type=spec.flag_type,
                reason=spec.reason,
                created_at=now,
            )
            self.flags[flag.id] = flag
            created.append(flag)
        return CreateUserFlagsOutput(flags=created)

    def list_teams(self, args: ListTeamsInput) -> ListTeamsOutput:
        teams = list(self.teams.values())
        if args.user_ids is not None:
            ids = set(args.user_ids)
            teams = [t for t in teams if t.user_id in ids]
        if args.created_after is not None:
            teams = [t for t in teams if t.created_at > args.created_after]
        return ListTeamsOutput(teams=teams)

    def list_pokemon(self, args: ListPokemonInput) -> ListPokemonOutput:
        if args.pokemon_ids is None:
            pokemon = list(self.pokemon.values())
        else:
            ids = set(args.pokemon_ids)
            pokemon = [p for p in self.pokemon.values() if p.id in ids]
        return ListPokemonOutput(pokemon=pokemon)

    def list_purchases(self, args: ListPurchasesInput) -> ListPurchasesOutput:
        purchases = list(self.purchases.values())
        if args.user_ids is not None:
            ids = set(args.user_ids)
            purchases = [p for p in purchases if p.user_id in ids]
        if args.purchased_after is not None:
            purchases = [p for p in purchases if p.purchased_at > args.purchased_after]
        return ListPurchasesOutput(purchases=purchases)

    def list_engagement(self, args: ListEngagementInput) -> ListEngagementOutput:
        rows = list(self.engagement.values())
        if args.user_ids is not None:
            ids = set(args.user_ids)
            rows = [r for r in rows if r.user_id in ids]
        if args.date_from is not None:
            rows = [r for r in rows if r.date >= args.date_from]
        if args.date_to is not None:
            rows = [r for r in rows if r.date <= args.date_to]
        return ListEngagementOutput(engagement=rows)

    def post_message(self, args: PostMessageInput) -> PostMessageOutput:
        msg = Message(
            id=str(uuid4()),
            channel=args.channel,
            text=args.text,
            created_at=datetime.utcnow(),
        )
        self.messages.append(msg)
        return PostMessageOutput(message=msg)

    def list_messages(self, args: ListMessagesInput) -> ListMessagesOutput:
        msgs = [m for m in self.messages if m.channel == args.channel]
        msgs = sorted(msgs, key=lambda m: m.created_at, reverse=True)
        return ListMessagesOutput(messages=msgs[: args.limit])

    # =================================================
    # Adapter: expose tools to LangChain or OpenAI API
    # =================================================

    def tool_map(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a mapping from tool name to metadata:
        {
            "list_users": {
                "input_model": ListUsersInput,
                "output_model": ListUsersOutput,
                "func": self.list_users,
                "description": "...",
            },
            ...
        }
        """
        return {
            "list_users": {
                "input_model": ListUsersInput,
                "output_model": ListUsersOutput,
                "func": self.list_users,
                "description": "List users by ids, segment, region, and signup date.",
            },
            "list_subscriptions": {
                "input_model": ListSubscriptionsInput,
                "output_model": ListSubscriptionsOutput,
                "func": self.list_subscriptions,
                "description": "List subscriptions filtered by user, plan, and status.",
            },
            "bulk_update_user_notes": {
                "input_model": BulkUpdateUserNotesInput,
                "output_model": BulkUpdateUserNotesOutput,
                "func": self.bulk_update_user_notes,
                "description": "Update admin notes for many users at once.",
            },
            "create_user_flags": {
                "input_model": CreateUserFlagsInput,
                "output_model": CreateUserFlagsOutput,
                "func": self.create_user_flags,
                "description": "Create flags for users with reasons.",
            },
            "list_teams": {
                "input_model": ListTeamsInput,
                "output_model": ListTeamsOutput,
                "func": self.list_teams,
                "description": "List teams by users and creation date.",
            },
            "list_pokemon": {
                "input_model": ListPokemonInput,
                "output_model": ListPokemonOutput,
                "func": self.list_pokemon,
                "description": "List Pokémon by ids.",
            },
            "list_purchases": {
                "input_model": ListPurchasesInput,
                "output_model": ListPurchasesOutput,
                "func": self.list_purchases,
                "description": "List purchases by users and date.",
            },
            "list_engagement": {
                "input_model": ListEngagementInput,
                "output_model": ListEngagementOutput,
                "func": self.list_engagement,
                "description": "List engagement rows by users and date range.",
            },
            "post_message": {
                "input_model": PostMessageInput,
                "output_model": PostMessageOutput,
                "func": self.post_message,
                "description": "Post a message to an internal channel.",
            },
            "list_messages": {
                "input_model": ListMessagesInput,
                "output_model": ListMessagesOutput,
                "func": self.list_messages,
                "description": "List recent messages in a channel.",
            },
        }

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Build OpenAI tools definition from the Pydantic input models.
        The caller still needs to route tool calls into the funcs.
        """
        tools = []
        for name, meta in self.tool_map().items():
            input_model = meta["input_model"]
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": meta["description"],
                        "parameters": input_model.model_json_schema(),
                    },
                }
            )
        return tools

    def to_langchain_tools(self) -> List[StructuredTool]:
        """
        Build LangChain StructuredTool instances from the tool map.
        Each tool wraps the corresponding method and uses Pydantic models for validation.
        """
        tools = []
        for name, meta in self.tool_map().items():
            input_model = meta["input_model"]
            func = meta["func"]
            description = meta["description"]

            # Create a wrapper that converts dict args to Pydantic input
            def make_tool_func(f: Callable, in_model):
                def tool_func(**kwargs) -> Any:
                    args = in_model(**kwargs)
                    result = f(args)
                    # Return the Pydantic model output as a dict
                    return result.model_dump()
                return tool_func

            tool = StructuredTool(
                name=name,
                description=description,
                func=make_tool_func(func, input_model),
                args_schema=input_model,
            )
            tools.append(tool)

        return tools

"""Eval tests for tool selection behavior.

Tests whether the agent selects the correct tool(s) from a set of available
tools given direct, indirect, and multi-step user requests. Ported from the
agent-builder-graphs tool-discovery eval suite and adapted for deepagents.

The agent is given a pool of mock tools and a user query. Assertions check that
the agent called the right tool(s) and avoided calling wrong ones.

The module carries two catalogs. The original eight-tool pool covers direct,
indirect, and multi-step requests where each tool is easy to tell apart. The
32-tool catalog at the bottom of the file is the high-cardinality baseline: four
tools per business domain, with descriptions that separate only by object or
verb, which is the shape applications get when they wire up many MCP servers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent
from langchain_core.tools import tool

if TYPE_CHECKING:
    from typing import Any

    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

from tests.evals.utils import (
    ToolNotCalled,
    TrajectoryScorer,
    final_text_contains,
    run_agent,
    tool_call,
    tool_called,
    tool_not_called,
)

pytestmark = [pytest.mark.eval_category("tool_use")]
"""Apply tool_use category to all tests in this module. Tier is set per-test."""

# ---------------------------------------------------------------------------
# Mock tools — lightweight stubs that return a fixed string
# ---------------------------------------------------------------------------


@tool
def slack_send_dm(user_id: str, message: str) -> str:
    """Send a direct message to a user on Slack."""
    return f"Sent DM to {user_id}: {message}"


@tool
def slack_post_channel(channel: str, message: str) -> str:
    """Post a message to a Slack channel."""
    return f"Posted to #{channel}: {message}"


@tool
def github_create_issue(repo: str, title: str, body: str) -> str:
    """Create a new GitHub issue."""
    return f"Created issue '{title}' in {repo} — {body}"


@tool
def github_create_pr(repo: str, title: str, head: str, base: str) -> str:
    """Create a pull request on GitHub."""
    return f"Created PR '{title}' in {repo} ({head} -> {base})"


@tool
def linear_create_issue(team: str, title: str, description: str) -> str:
    """Create a new issue in Linear."""
    return f"Created Linear issue '{title}' in {team} — {description}"


@tool
def gmail_send_email(to: str, subject: str, body: str) -> str:
    """Send an email via Gmail."""
    return f"Sent email to {to}: {subject} — {body}"


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        f"Top 3 results for {query!r}:\n"
        f"1. Official documentation page covering the topic.\n"
        f"2. Recent blog post discussing updates and features.\n"
        f"3. Community discussion thread with examples."
    )


@tool
def calendar_create_event(title: str, date: str, attendees: list[str]) -> str:
    """Create a calendar event."""
    return f"Created event '{title}' on {date} with {', '.join(attendees)}"


ALL_TOOLS = [
    slack_send_dm,
    slack_post_channel,
    github_create_issue,
    github_create_pr,
    linear_create_issue,
    gmail_send_email,
    web_search,
    calendar_create_event,
]


def _make_agent(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
    """Build a deep agent with the standard tool pool for these evals."""
    return create_deep_agent(model=model, tools=ALL_TOOLS)


# ---------------------------------------------------------------------------
# Test cases — direct requests (user names the tool explicitly)
# ---------------------------------------------------------------------------


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_direct_request_slack_dm(model: BaseChatModel) -> None:
    """Agent uses the Slack DM tool when explicitly asked."""
    agent = _make_agent(model)
    run_agent(
        agent,
        model=model,
        query="Send a Slack DM to user U12345 saying 'Hello from evals'",
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[tool_call(name="slack_send_dm", args_contains={"user_id": "U12345"})],
            )
            .success(
                final_text_contains("U12345", case_insensitive=True),
            )
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_direct_request_github_pr(model: BaseChatModel) -> None:
    """Agent uses the GitHub PR tool when explicitly asked."""
    agent = _make_agent(model)
    run_agent(
        agent,
        model=model,
        query="Create a pull request in repo langchain-ai/deepagents with title 'fix: typo' from branch fix-typo to main",
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[
                    tool_call(
                        name="github_create_pr",
                        args_contains={"repo": "langchain-ai/deepagents"},
                    )
                ],
            )
            .success(
                final_text_contains("fix-typo", case_insensitive=True),
            )
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_direct_request_multiple_tools(model: BaseChatModel) -> None:
    """Agent uses both Linear and GitHub issue tools when asked for both."""
    agent = _make_agent(model)
    run_agent(
        agent,
        model=model,
        query=(
            "Create an issue titled 'Bug: crash on login' in the Linear team 'engineering' "
            "and also create a GitHub issue in repo org/app with the same title and body 'Tracking in Linear'"
        ),
        scorer=(
            TrajectoryScorer()
            .expect(
                tool_calls=[
                    tool_call(name="linear_create_issue"),
                    tool_call(name="github_create_issue"),
                ],
            )
            .success(
                final_text_contains("crash on login", case_insensitive=True),
            )
        ),
    )


# ---------------------------------------------------------------------------
# Indirect requests (user describes intent, agent infers tool)
# ---------------------------------------------------------------------------


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_indirect_schedule_meeting(model: BaseChatModel) -> None:
    """Agent infers the calendar tool from a scheduling request."""
    agent = _make_agent(model)
    run_agent(
        agent,
        model=model,
        query="Schedule a team standup for tomorrow at 10am with alice@co.com and bob@co.com",
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[tool_call(name="calendar_create_event")],
            )
            .success(
                final_text_contains("standup", case_insensitive=True),
            )
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_indirect_notify_team(model: BaseChatModel) -> None:
    """Agent infers the Slack channel post tool from a 'notify the team' request."""
    agent = _make_agent(model)
    run_agent(
        agent,
        model=model,
        query="Notify the #deployments channel that v2.0 has been released",
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[
                    tool_call(
                        name="slack_post_channel",
                        args_contains={"channel": "deployments"},
                    )
                ],
            )
            .success(
                final_text_contains("v2.0", case_insensitive=True),
            )
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_indirect_email_report(model: BaseChatModel) -> None:
    """Agent infers the Gmail tool from 'email a report' request."""
    agent = _make_agent(model)
    run_agent(
        agent,
        model=model,
        query="Email the weekly status report to manager@company.com with subject 'Week 10 Status'",
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[
                    tool_call(
                        name="gmail_send_email",
                        args_contains={"to": "manager@company.com"},
                    )
                ],
            )
            .success(
                final_text_contains("Week 10", case_insensitive=True),
            )
        ),
    )


# ---------------------------------------------------------------------------
# Multi-step: tool chaining
# ---------------------------------------------------------------------------


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_chain_search_then_email(model: BaseChatModel) -> None:
    """Agent searches the web then emails results — two tools in sequence."""
    agent = _make_agent(model)
    run_agent(
        agent,
        model=model,
        query="Search for 'LangGraph 0.3 release notes' and email a summary to team@co.com with subject 'LangGraph Update'",
        scorer=(
            TrajectoryScorer()
            .expect(
                tool_calls=[
                    tool_call(name="web_search"),
                    tool_call(name="gmail_send_email", args_contains={"to": "team@co.com"}),
                ],
            )
            .success(
                final_text_contains("team@co.com", case_insensitive=True),
            )
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_chain_create_issue_then_notify(model: BaseChatModel) -> None:
    """Agent creates a GitHub issue then notifies Slack — two tools in sequence."""
    agent = _make_agent(model)
    run_agent(
        agent,
        model=model,
        query=(
            "Create a GitHub issue in org/backend titled 'Fix memory leak' with body 'OOM in prod', "
            "then post a message to #incidents saying the issue was created"
        ),
        scorer=(
            TrajectoryScorer()
            .expect(
                tool_calls=[
                    tool_call(name="github_create_issue"),
                    tool_call(
                        name="slack_post_channel",
                        args_contains={"channel": "incidents"},
                    ),
                ],
            )
            .success(
                final_text_contains("memory leak", case_insensitive=True),
            )
        ),
    )


# ---------------------------------------------------------------------------
# High-cardinality catalog — 32 local stubs across eight business domains
# ---------------------------------------------------------------------------
#
# The eight tools above are easy to tell apart, so a model can pick correctly
# without reading descriptions closely. Real applications that connect a dozen
# MCP servers or vendor APIs hand the model a much more crowded list where
# several tools per domain are plausible: same object with different verbs, or
# the same verb against a nearby object. This catalog reproduces that shape with
# four tools per domain.
#
# Every stub is a pure local function returning a fixed string, and every reply
# starts with the calling tool's name so a near-miss is readable in the logged
# trajectory.

# CRM


@tool
def crm_find_contact(email: str) -> str:
    """Look up the single CRM contact that owns an exact email address."""
    return f"crm_find_contact: matched one contact for {email}"


@tool
def crm_search_contacts(query: str) -> str:
    """Search CRM contacts by free-text name, company, or tag."""
    return f"crm_search_contacts: 4 contacts match {query!r}"


@tool
def crm_create_contact(name: str, email: str) -> str:
    """Create a new CRM contact record."""
    return f"crm_create_contact: created {name} <{email}>"


@tool
def crm_list_contacts_by_owner(owner: str) -> str:
    """List every CRM contact owned by a given account executive."""
    return f"crm_list_contacts_by_owner: 12 contacts owned by {owner}"


# Billing


@tool
def billing_search_invoices(query: str) -> str:
    """Search billing invoices by customer name, amount, or free-text note."""
    return f"billing_search_invoices: 3 invoices match {query!r}"


@tool
def billing_get_invoice(invoice_id: str) -> str:
    """Retrieve a single billing invoice by its invoice ID."""
    return f"billing_get_invoice: invoice {invoice_id}, status open"


@tool
def billing_get_refund_status(refund_id: str) -> str:
    """Check the settlement status of a refund that was already issued."""
    return f"billing_get_refund_status: refund {refund_id} is pending settlement"


@tool
def billing_issue_refund(invoice_id: str, amount: str) -> str:
    """Issue a new refund against a paid invoice."""
    return f"billing_issue_refund: refunded {amount} against invoice {invoice_id}"


# Support


@tool
def support_open_ticket(subject: str, requester: str) -> str:
    """Open a new customer support ticket."""
    return f"support_open_ticket: opened {subject!r} for {requester}"


@tool
def support_get_ticket_status(ticket_id: str) -> str:
    """Read the current status of a single support ticket."""
    return f"support_get_ticket_status: ticket {ticket_id} is awaiting customer reply"


@tool
def support_list_tickets_by_status(status: str) -> str:
    """List every support ticket that currently shares a status."""
    return f"support_list_tickets_by_status: 7 tickets with status {status}"


@tool
def support_escalate_ticket(ticket_id: str, team: str) -> str:
    """Escalate an existing support ticket to another team."""
    return f"support_escalate_ticket: ticket {ticket_id} escalated to {team}"


# Calendar


@tool
def calendar_book_event(title: str, date: str) -> str:
    """Book a new event on the shared team calendar."""
    return f"calendar_book_event: booked {title!r} on {date}"


@tool
def calendar_reschedule_event(event_id: str, new_date: str) -> str:
    """Move an existing calendar event to a new date."""
    return f"calendar_reschedule_event: event {event_id} moved to {new_date}"


@tool
def calendar_find_free_slot(date: str, attendees: list[str]) -> str:
    """Find a time slot when every attendee is free on a given date."""
    return f"calendar_find_free_slot: 14:30 works for {', '.join(attendees)} on {date}"


@tool
def calendar_list_events(date: str) -> str:
    """List the events already scheduled on a given date."""
    return f"calendar_list_events: 5 events on {date}"


# GitHub


@tool
def github_search_pull_requests(repo: str, query: str) -> str:
    """Search pull requests in a repository by free-text."""
    return f"github_search_pull_requests: 2 matches for {query!r} in {repo}"


@tool
def github_list_pull_requests(repo: str) -> str:
    """List the open pull requests in a repository."""
    return f"github_list_pull_requests: 9 open pull requests in {repo}"


@tool
def github_merge_pull_request(repo: str, number: int) -> str:
    """Merge one open pull request into its base branch."""
    return f"github_merge_pull_request: merged PR #{number} in {repo}"


@tool
def github_create_pull_request(repo: str, title: str, head: str, base: str) -> str:
    """Open a new pull request from a head branch into a base branch."""
    return f"github_create_pull_request: opened {title!r} in {repo} ({head} -> {base})"


# Slack


@tool
def slack_search_messages(query: str) -> str:
    """Search Slack message history by keyword."""
    return f"slack_search_messages: 6 messages mention {query!r}"


@tool
def slack_send_direct_message(user: str, text: str) -> str:
    """Send a direct message to a single Slack user."""
    return f"slack_send_direct_message: sent to {user}: {text}"


@tool
def slack_list_channels() -> str:
    """List the Slack channels the workspace bot can see."""
    return "slack_list_channels: #general, #deployments, #incidents, #support"


@tool
def slack_set_channel_topic(channel: str, topic: str) -> str:
    """Replace the topic line shown at the top of a Slack channel."""
    return f"slack_set_channel_topic: #{channel} topic set to {topic!r}"


# Documentation


@tool
def docs_search(query: str) -> str:
    """Search the internal documentation wiki by keyword."""
    return f"docs_search: 5 pages match {query!r}"


@tool
def docs_get_page(page_id: str) -> str:
    """Fetch the full body of one documentation page."""
    return f"docs_get_page: page {page_id}, 41 lines"


@tool
def docs_create_page(title: str, body: str) -> str:
    """Create a new documentation page."""
    return f"docs_create_page: created {title!r} ({len(body)} chars)"


@tool
def docs_list_recent_updates(limit: int) -> str:
    """List the documentation pages changed most recently."""
    return f"docs_list_recent_updates: {limit} pages updated this week"


# Deployment


@tool
def deploy_trigger_release(service: str, version: str) -> str:
    """Start deploying a version of a service."""
    return f"deploy_trigger_release: deploying {version} to {service}"


@tool
def deploy_rollback_release(service: str, version: str) -> str:
    """Roll a service back to a previously deployed version."""
    return f"deploy_rollback_release: rolling {service} back to {version}"


@tool
def deploy_get_release_status(service: str) -> str:
    """Check whether the latest release of a service finished rolling out."""
    return f"deploy_get_release_status: {service} at 2.14.0, rollout 60% (3 of 5 pods)"


@tool
def deploy_list_releases(service: str) -> str:
    """List recent releases for a service."""
    return f"deploy_list_releases: 8 releases for {service}, newest 2.14.0"


HIGH_CARDINALITY_TOOLS: list[BaseTool] = [
    # CRM
    crm_find_contact,
    crm_search_contacts,
    crm_create_contact,
    crm_list_contacts_by_owner,
    # Billing
    billing_search_invoices,
    billing_get_invoice,
    billing_get_refund_status,
    billing_issue_refund,
    # Support
    support_open_ticket,
    support_get_ticket_status,
    support_list_tickets_by_status,
    support_escalate_ticket,
    # Calendar
    calendar_book_event,
    calendar_reschedule_event,
    calendar_find_free_slot,
    calendar_list_events,
    # GitHub
    github_search_pull_requests,
    github_list_pull_requests,
    github_merge_pull_request,
    github_create_pull_request,
    # Slack
    slack_search_messages,
    slack_send_direct_message,
    slack_list_channels,
    slack_set_channel_topic,
    # Documentation
    docs_search,
    docs_get_page,
    docs_create_page,
    docs_list_recent_updates,
    # Deployment
    deploy_trigger_release,
    deploy_rollback_release,
    deploy_get_release_status,
    deploy_list_releases,
]
"""The crowded catalog. Catalog size is the variable under test, so it is pinned below."""

_EXPECTED_CATALOG_SIZE = 32
"""32 tools is the cardinality this baseline measures; changing it changes the result."""


def _check_catalog(tools: list[BaseTool]) -> None:
    """Fail at import time if a tool was dropped, renamed, or duplicated.

    Editing the catalog is the normal way these evals drift: a deleted tool
    quietly lowers the cardinality, and a duplicated name makes the decoy
    assertions below ambiguous rather than wrong.

    Args:
        tools: The catalog to validate.

    Raises:
        ValueError: If the catalog is the wrong size or repeats a tool name.
    """
    names = [catalog_tool.name for catalog_tool in tools]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if len(names) != _EXPECTED_CATALOG_SIZE or duplicates:
        msg = (
            f"High-cardinality catalog needs {_EXPECTED_CATALOG_SIZE} uniquely named tools, "
            f"found {len(names)} with duplicates {duplicates}"
        )
        raise ValueError(msg)


_check_catalog(HIGH_CARDINALITY_TOOLS)

_CATALOG_NAMES: frozenset[str] = frozenset(
    catalog_tool.name for catalog_tool in HIGH_CARDINALITY_TOOLS
)


def _make_high_cardinality_agent(model: BaseChatModel) -> CompiledStateGraph[Any, Any]:
    """Build a deep agent whose visible tool list is the 32-tool catalog."""
    return create_deep_agent(model=model, tools=HIGH_CARDINALITY_TOOLS)


def _forbidden_actions(*names: str) -> tuple[ToolNotCalled, ...]:
    """Build hard-fail assertions for the near-miss *actions* a scenario forbids.

    Only mutating neighbors belong here. Reading a sibling tool for context is
    not a selection error — a model that merges the right pull request and also
    lists the others still chose correctly — so extra read calls are left to the
    efficiency expectations, where they surface as a `tool_call_ratio` above 1.
    Mutating the wrong object is the failure this baseline hard-fails on.

    Args:
        *names: Catalog tool names whose invocation would be the wrong action.

    Returns:
        One `tool_not_called` assertion per name.

    Raises:
        ValueError: If a name is not in the catalog. A `tool_not_called` check
            against a renamed or mistyped tool passes without testing anything.
    """
    unknown = sorted(set(names) - _CATALOG_NAMES)
    if unknown:
        msg = f"Decoy tools missing from the high-cardinality catalog: {unknown}"
        raise ValueError(msg)
    return tuple(tool_not_called(name) for name in names)


# ---------------------------------------------------------------------------
# High-cardinality: pick the right tool out of 32 neighbors
# ---------------------------------------------------------------------------


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_high_cardinality_direct_merge_pull_request(model: BaseChatModel) -> None:
    """Agent merges the named pull request rather than opening a new one.

    All four GitHub tools take a repository and talk about pull requests, so the
    verb is the only thing separating merge from its neighbors. Listing or
    searching first is fair game — that is normal reconnaissance — but opening a
    pull request instead of merging one is the wrong action, and any extra reads
    show up in `tool_call_ratio` rather than failing the case.
    """
    agent = _make_high_cardinality_agent(model)
    run_agent(
        agent,
        model=model,
        query="Merge pull request #512 in the org/payments repository.",
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[tool_call(name="github_merge_pull_request")],
            )
            .success(
                tool_called("github_merge_pull_request", args_contains={"repo": "org/payments"}),
                *_forbidden_actions("github_create_pull_request"),
                final_text_contains("512", case_insensitive=True),
            )
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_high_cardinality_indirect_release_rollout_status(model: BaseChatModel) -> None:
    """Agent inspects the rollout status instead of shipping or rolling back.

    Neither "deploy" nor "status" appears in the request; the ask is whether a
    rollout finished, which this domain expresses as a release-status lookup.
    Re-triggering the release or rolling it back would look like fixing the
    complaint while answering nothing, so both are hard failures. Listing
    releases for context is not.
    """
    agent = _make_high_cardinality_agent(model)
    run_agent(
        agent,
        model=model,
        query=(
            "The checkout team tagged 2.14.0 this morning, but customers still get the "
            "old bundle. Did the rollout to checkout-api actually finish, or is it "
            "stuck partway?"
        ),
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=1,
                tool_calls=[tool_call(name="deploy_get_release_status")],
            )
            .success(
                tool_called("deploy_get_release_status", args_contains={"service": "checkout-api"}),
                *_forbidden_actions("deploy_trigger_release", "deploy_rollback_release"),
                final_text_contains("checkout-api", case_insensitive=True),
            )
        ),
    )

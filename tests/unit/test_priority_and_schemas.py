import pytest
from pydantic import ValidationError

from app.models.schemas import TicketRequest
from app.services.priority import predict_priority


def test_production_outage_for_everyone_is_p1():
    result = predict_priority("The production database is down for all users")
    assert result.level == "P1"
    assert any("outage" in line for line in result.reasoning)


def test_no_signals_is_p4_with_explanation():
    result = predict_priority("Please install a new wallpaper on my laptop")
    assert result.level == "P4"
    assert result.reasoning


def test_user_urgency_and_impact_contribute_and_are_listed():
    result = predict_priority("Printer is not printing", urgency="high", impact="team")
    assert result.score == 3
    assert result.level == "P3"
    assert any("urgency" in line for line in result.reasoning)
    assert any("impact" in line for line in result.reasoning)


def test_stated_urgency_alone_cannot_reach_p1():
    assert predict_priority("something is odd", urgency="critical", impact="organization").level == "P2"


def test_workaround_lowers_score():
    base = predict_priority("Application is slow for the whole team")
    lowered = predict_priority("Application is slow for the whole team, there is a workaround")
    assert lowered.score == base.score - 1


def test_schema_normalises_enums_and_blanks():
    ticket = TicketRequest(
        title="  VPN down  ", description="  the vpn does not connect at all  ",
        urgency="HIGH", impact="", affected_system="  ", ticket_id=" ABC-1 ",
    )
    assert ticket.title == "VPN down"
    assert ticket.urgency == "high"
    assert ticket.impact is None
    assert ticket.affected_system is None
    assert ticket.ticket_id == "ABC-1"


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "   ", "description": "a valid long description"},
        {"title": "ok title", "description": "short"},
        {"title": "ok title", "description": "x" * 5001},
        {"title": "t" * 201, "description": "a valid long description"},
        {"title": "ok title", "description": "a valid long description", "urgency": "urgent!!"},
        {"title": "ok title", "description": "a valid long description", "ticket_id": "bad id/../"},
        {"title": "ok title", "description": "a valid long description", "unexpected": 1},
    ],
)
def test_schema_rejects_invalid_input(payload):
    with pytest.raises(ValidationError):
        TicketRequest(**payload)

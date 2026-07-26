"""Characterisation tests for entity grounding on generic, entity-less topics.

Both tests below are xfail(strict=True): they assert the behaviour we want and
document the behaviour we currently get. When either is fixed the test XPASSes
and pytest fails loudly, which is the signal to remove the marker.

Observed 2026-07-26 on a real run of the topic
"AI code review bottleneck: can reviewers still judge AI-produced work".
Of the eight rendered clusters, three were pure off-topic noise (a promotional
post for an AI course, a Chinese web-drama upload, and a space-astronomy video)
and they scored 39-41 against on-topic clusters scoring 39-47, so the drama
upload outranked the most on-topic thread in the corpus. Neither downstream
filter is at fault: `render._clusters_clearing_relevance_floor` only drops a
cluster whose representatives are *all* explicitly entity-miss-demoted, and the
demotion never fired. The two defects below are why it never fired.
"""

import pytest
from lib import rerank

GENERIC_TOPIC = "AI code review bottleneck: can reviewers still judge AI-produced work"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "'review' is in planner._INTENT_MODIFIER_PATTERNS, so _primary_entity strips it "
        "even when review is the subject of the topic rather than the intent. The topic "
        "above yields 'AI code bottleneck: ...', silently deleting its most specific noun."
    ),
)
def test_primary_entity_keeps_review_when_review_is_the_subject():
    entity = rerank._primary_entity(GENERIC_TOPIC)
    # Substring matching is not enough here: the unrelated token "reviewers"
    # survives stripping and would satisfy a bare `"review" in entity` check.
    # The phrase that actually disappears is "code review".
    assert "code review" in entity.lower(), (
        f"expected the subject phrase to survive intent-modifier stripping, got {entity!r}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "_entity_grounded keys on the head token of the primary entity. When that head "
        "token is a ubiquitous word like 'AI', every item in an AI-adjacent corpus counts "
        "as grounded, no candidate is marked entity-miss, and the entity-miss penalty plus "
        "every downstream entity-miss filter become inert on generic topics."
    ),
)
def test_ubiquitous_head_token_does_not_ground_off_topic_items():
    entity = rerank._primary_entity(GENERIC_TOPIC)
    off_topic = (
        "INSTEAD OF WATCHING NETFLIX TONIGHT Spend 1 hour with this Claude AI FULL "
        "COURSE that teaches you how to BUILD and AUTOMATE anything"
    )
    assert not rerank._entity_grounded(off_topic, entity), (
        "a course advertisement sharing only the token 'AI' should not count as grounded"
    )


def test_on_topic_item_is_grounded_control():
    """Control: the same machinery must still ground a genuinely on-topic item.

    Without this, a fix could satisfy the two xfails above by grounding nothing.
    """
    entity = rerank._primary_entity(GENERIC_TOPIC)
    on_topic = "How to keep QA from being a giant bottleneck with AI coding"
    assert rerank._entity_grounded(on_topic, entity)

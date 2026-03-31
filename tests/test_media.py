from __future__ import annotations

from src.media import pick_representative_image
from src.models import GeneratedArticle, IngestedArticle, Reliability


class TestRepresentativeImageSelection:
    def test_prefers_source_with_matching_story_keywords(self):
        related = IngestedArticle(
            source_name="NPR",
            source_slant=-0.35,
            source_reliability=Reliability.HIGH,
            title="Trump executive order on mail ballots triggers lawsuits",
            url="https://example.com/npr",
            text="Trump signed an executive order on voting and mail ballots. Legal experts expect lawsuits.",
            image_url="https://example.com/relevant.jpg",
        )
        unrelated = IngestedArticle(
            source_name="Fox News",
            source_slant=0.55,
            source_reliability=Reliability.HIGH,
            title="F-35 crashes near Las Vegas during training mission",
            url="https://example.com/fox",
            text="An F-35 fighter jet crashed near Las Vegas during a training mission.",
            image_url="https://example.com/unrelated.jpg",
        )

        article = GeneratedArticle(
            headline="Trump signs voting order amid legal challenges",
            body="President Donald Trump signed an executive order affecting mail ballots and voting procedures, drawing immediate legal challenges.",
            sources_used=[unrelated, related],
        )

        assert pick_representative_image(article) == "https://example.com/relevant.jpg"

    def test_returns_none_when_no_source_has_image(self):
        article = GeneratedArticle(
            headline="Test headline",
            body="Body text",
            sources_used=[
                IngestedArticle(
                    source_name="AP",
                    source_slant=0.0,
                    source_reliability=Reliability.VERY_HIGH,
                    title="Headline",
                    url="https://example.com/ap",
                    text="Body text",
                    image_url=None,
                )
            ],
        )

        assert pick_representative_image(article) is None

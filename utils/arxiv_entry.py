import json
from typing import Dict, List
from datetime import datetime


class ArxivEntry:
    def __init__(
            self,
            paper_id: str,
            title: str,
            authors: List[str],
            abstract: str,
            first_published: int,
            comment: str,
    ) -> None:
        self.paper_id = paper_id
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.first_published = first_published  # unix timestamp
        self.comment = comment

    def to_dict(self) -> Dict:
        return dict(
            paper_id=self.paper_id,
            title=self.title,
            authors=self.authors,
            abstract=self.abstract,
            first_published=self.first_published,
            comment=self.comment,
        )

    def pretty_print(self):
        print(json.dumps(self.to_dict(), indent=4))
        print(self.published_as_iso_str)

    @property
    def key(self) -> str:
        return self.paper_id

    @property
    def abstract_url(self) -> str:
        return f'https://arxiv.org/abs/{self.paper_id}'

    @property
    def pdf_url(self) -> str:
        return f'https://arxiv.org/pdf/{self.paper_id}'

    @property
    def published_as_iso_str(self) -> str:
        published_as_unix_ts = self.first_published
        return datetime.fromtimestamp(published_as_unix_ts).isoformat()

    @property
    def published_as_month_and_year(self) -> str:
        published_as_unix_ts = self.first_published
        return datetime.fromtimestamp(published_as_unix_ts).strftime('%B %Y')

    def __str__(self):
        return str(self.to_dict())

    @staticmethod
    def from_dict(data: Dict) -> 'ArxivEntry':
        entry = ArxivEntry(**data)
        return entry




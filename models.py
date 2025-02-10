from pydantic import BaseModel, HttpUrl, field_validator
from datetime import datetime

class ScrapedArticle(BaseModel):
    source: str
    link: HttpUrl
    title: str
    category: list[str]
    description: str
    content: str
    pub_date: datetime
    scraped_date: datetime

    @field_validator("category", mode="before")
    @classmethod
    def category_to_string(cls, value):
        """Convert list to comma-separated string before saving."""
        if isinstance(value, list):
            return ", ".join(value)
        return value  # If already a string, return as is

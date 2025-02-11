from pydantic import BaseModel, HttpUrl, field_validator
from datetime import datetime
from typing import List, Union

class ScrapedArticle(BaseModel):
    source: str
    link: HttpUrl
    title: str
    category: List[str]  # Keep as a list in Python
    description: str
    content: str
    pub_date: datetime
    scraped_date: datetime

    @field_validator("category", mode="before")
    @classmethod
    def category_to_string(cls, value: Union[str, List[str], None]):
        """Ensure category is a list of strings"""
        if isinstance(value, str):
            return value.split(", ")  # Convert stored string back into a list
        elif isinstance(value, list):
            return [str(v) for v in value]  # Ensure all elements are strings
        return []  # Default to an empty list if None

    def category_as_string(self) -> str:
        """Helper method to convert category list into a string for database storage."""
        return ", ".join(self.category)  # Convert list to comma-separated string

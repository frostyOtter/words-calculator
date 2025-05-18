from pydantic import BaseModel, Field
from typing import List


class SimilarWords(BaseModel):
    scratch_pad: str
    similar_words: List[str] = Field(
        ...,
        description="Generate at least 5 words that have high similarity to the input, maximum of 10 words.",
    )

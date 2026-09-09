from pydantic import BaseModel, HttpUrl, validator
from typing import List, Optional, Union
from datetime import date
from enum import Enum

class ReadingStatus(str, Enum):
    WANT_TO_READ = "Want to Read"
    READING = "Reading" 
    READ = "Read"

class PaperBase(BaseModel):
    title: str
    reading_status: ReadingStatus
    date_added: date
    link: Optional[str] = None
    topics: List[str] = []
    description: Optional[str] = None

class PaperCreate(PaperBase):
    pass

class PaperUpdate(BaseModel):
    title: Optional[str] = None
    reading_status: Optional[ReadingStatus] = None
    date_added: Optional[date] = None
    link: Optional[str] = None
    topics: Optional[List[str]] = None
    description: Optional[str] = None

class Paper(PaperBase):
    id: Optional[int] = None
    month: Optional[str] = None

    class Config:
        from_attributes = True

class ArxivFetchRequest(BaseModel):
    arxiv_url: str

class ArxivFetchResponse(BaseModel):
    title: str
    topics: List[str]
    description: str
    date: str
    link: str

class FilterParams(BaseModel):
    status_filter: Optional[str] = "All"
    topic_filter: Optional[List[str]] = None
    text_filter: Optional[str] = None
    sort_by: str = "Date Added"
    sort_direction: str = "Descending"

class AnalysisMetrics(BaseModel):
    total_papers: int
    read_papers: int
    reading_papers: int
    want_to_read_papers: int
    reading_velocity: float
    completion_rate: float
    topic_distribution: dict
    monthly_activity: dict

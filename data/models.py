
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any


class User(BaseModel):
    user_id: str = Field(alias='id')
    latitude: float
    longitude: float
    first_name: Optional[str] = None
    email: Optional[str] = None
    office_distance: float = 0.0

    @validator('latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v

    @validator('longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v

    class Config:
        allow_population_by_field_name = True


class Driver(BaseModel):
    driver_id: str = Field(alias='id')
    latitude: float
    longitude: float
    capacity: int
    vehicle_id: Optional[str] = None
    priority: int = 1

    @validator('capacity')
    def validate_capacity(cls, v):
        if v <= 0:
            raise ValueError('Capacity must be positive')
        return v

    @validator('latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v

    @validator('longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v

    class Config:
        allow_population_by_field_name = True


class AssignedUser(BaseModel):
    user_id: str
    lat: float
    lng: float
    office_distance: float = 0.0
    first_name: Optional[str] = None
    email: Optional[str] = None


class Route(BaseModel):
    driver_id: str
    vehicle_id: Optional[str] = None
    vehicle_type: int  # capacity
    latitude: float
    longitude: float
    assigned_users: List[AssignedUser] = []
    utilization: Optional[float] = None
    turning_score: Optional[float] = None
    tortuosity_ratio: Optional[float] = None
    direction_consistency: Optional[float] = None
    total_distance: Optional[float] = None


class AssignmentResult(BaseModel):
    status: str
    execution_time: float
    data: List[Route]
    unassignedUsers: List[Dict[str, Any]]
    unassignedDrivers: List[Dict[str, Any]]
    clustering_analysis: Dict[str, Any]
    optimization_mode: str
    parameter: int
    string_param: Optional[str] = None

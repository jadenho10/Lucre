from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional
from supabase import create_client, Client
import os
import bcrypt
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/profiles", tags=["profiles"])

supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_ANON_KEY"],
)


class ProfileCreate(BaseModel):
    name: str
    email: str
    password: str
    phone_number: Optional[str] = None


@router.post("/create_profile")
def create_profile(profile: ProfileCreate):
    try:
        password_hash = bcrypt.hashpw(
            profile.password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

        data = profile.model_dump(exclude_none=True, exclude={"password"})
        data["password_hash"] = password_hash

        result = (
            supabase.table("profiles")
            .insert(data)
            .execute()
        )
        return {"message": "Profile created successfully", "data": result.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/get_profile")
def get_profile(profile: ProfileCreate):
    try:
        result = (
            supabase.table("profiles")
            .select("*")
            .eq("email", profile.email)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Profile not found")
        return {"message": "Profile retrieved successfully", "data": result.data[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
"""
Auth API routes — register, login, refresh, logout, me.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from app.db.database import get_db
from app.db.models import User
from app.schemas.auth import UserCreate, UserResponse, Token
from app.core.security import (
    get_password_hash, verify_password, create_access_token,
    create_refresh_token, get_current_user,
)
from app.core.config import get_settings

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    # Check existing
    if db.query(User).filter(User.username == user_in.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    if db.query(User).filter(User.email == user_in.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        username=user_in.username,
        email=user_in.email,
        name=user_in.name,
        password_hash=get_password_hash(user_in.password),
        role="student",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(subject=user.username)
    return Token(access_token=access_token, token_type="bearer")


@router.post("/refresh", response_model=Token)
def refresh_token(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Issue a new access token for an authenticated user."""
    access_token = create_access_token(subject=current_user.username)
    return Token(access_token=access_token, token_type="bearer")


@router.post("/logout")
def logout(current_user: User = Depends(get_current_user)):
    """Logout — in production, revoke refresh token from Redis."""
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

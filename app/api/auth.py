"""
Auth API routes — register, login, logout, me.
"""
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import User
from app.schemas.auth import UserCreate, UserResponse, Token
from app.core.security import get_password_hash, verify_password, create_access_token, get_current_user
from app.core.config import get_settings
from app.core.rate_limit import limiter, client_ip

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, request: Request, db: Session = Depends(get_db)):
    limiter.check(f"register:{client_ip(request)}", get_settings().RATE_LIMIT_AUTH_PER_MIN)

    taken = db.query(User).filter(
        (func.lower(User.username) == user_in.username) | (func.lower(User.email) == user_in.email)
    ).first()
    if taken:
        raise HTTPException(status_code=400, detail="Username or email is already registered")

    user = User(
        username=user_in.username,
        email=user_in.email,
        name=user_in.name,
        password_hash=get_password_hash(user_in.password),
        role="student",
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Username or email is already registered")
    db.refresh(user)
    return user


@router.post("/login", response_model=Token)
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    settings = get_settings()
    identifier = form_data.username.strip().lower()[:254]
    limiter.check(f"login:{client_ip(request)}", settings.RATE_LIMIT_AUTH_PER_MIN)
    limiter.check(f"login-user:{identifier}", settings.RATE_LIMIT_AUTH_PER_MIN)

    user = db.query(User).filter(
        (func.lower(User.username) == identifier) | (func.lower(User.email) == identifier)
    ).first()
    # Always run bcrypt so response time doesn't reveal whether the user exists
    password_ok = verify_password(form_data.password[:72], user.password_hash if user else None)
    if not user or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return Token(
        access_token=create_access_token(user),
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/logout")
def logout(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Invalidate every token issued to this user."""
    current_user.token_version = (current_user.token_version or 0) + 1
    db.commit()
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

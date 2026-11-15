"""
Auth API routes — Google sign-in, logout, me.
"""
import re
import jwt
from jwt.exceptions import PyJWTError
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import User
from app.schemas.auth import GoogleLogin, UserResponse, Token
from app.core.security import create_access_token, get_current_user
from app.core.config import get_settings
from app.core.rate_limit import limiter, client_ip

router = APIRouter()

GOOGLE_ISSUERS = ["accounts.google.com", "https://accounts.google.com"]
# Fetches and caches Google's signing keys
_google_keys = jwt.PyJWKClient("https://www.googleapis.com/oauth2/v3/certs", cache_keys=True)


def verify_google_token(credential: str) -> dict:
    """Validate a Google ID token (signature, audience, issuer, expiry) and return its claims."""
    settings = get_settings()
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Google sign-in is not configured")
    try:
        key = _google_keys.get_signing_key_from_jwt(credential).key
        claims = jwt.decode(
            credential, key, algorithms=["RS256"], audience=settings.GOOGLE_CLIENT_ID,
            issuer=GOOGLE_ISSUERS, options={"require": ["exp", "iss", "aud", "sub", "email"]},
        )
    except PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid Google credential")
    if claims.get("email_verified") is not True:
        raise HTTPException(status_code=401, detail="Your Google email address is not verified")
    return claims


def _unique_username(db: Session, email: str) -> str:
    base = (re.sub(r"[^a-z0-9_.-]", "", email.split("@")[0].lower()) or "user")[:28].ljust(3, "0")
    name, n = base, 1
    while db.query(User.id).filter(func.lower(User.username) == name).first():
        n += 1
        name = f"{base}{n}"
    return name


@router.post("/google", response_model=Token)
def google_login(body: GoogleLogin, request: Request, db: Session = Depends(get_db)):
    settings = get_settings()
    limiter.check(f"login:{client_ip(request)}", settings.RATE_LIMIT_AUTH_PER_MIN)

    claims = verify_google_token(body.credential)
    email = claims["email"].lower()
    # shortcut: accounts are matched by verified email (no google_sub column); add one if users change Google emails
    user = db.query(User).filter(func.lower(User.email) == email).first()
    if user is None:
        limiter.check("register:global", settings.RATE_LIMIT_REGISTER_GLOBAL_PER_MIN)
        user = User(
            username=_unique_username(db, email),
            email=email,
            name=(claims.get("name") or email.split("@")[0])[:100],
            role="student",
        )
        db.add(user)
        try:
            db.commit()
        except IntegrityError:  # concurrent first sign-in with the same email
            db.rollback()
            user = db.query(User).filter(func.lower(User.email) == email).first()
            if user is None:
                raise HTTPException(status_code=409, detail="Please try signing in again")
        db.refresh(user)

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

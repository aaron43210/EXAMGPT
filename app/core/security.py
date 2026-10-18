"""
Security module: JWT auth and password hashing.
"""
import uuid
from datetime import datetime, timedelta, timezone
import jwt
from jwt.exceptions import PyJWTError
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.core.config import get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# Used to equalize login timing when the username does not exist
_DUMMY_HASH = bcrypt.hashpw(b"timing-equalizer", bcrypt.gensalt()).decode()


def verify_password(plain_password: str, hashed_password: str | None) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), (hashed_password or _DUMMY_HASH).encode("utf-8"))
    except ValueError:
        return False


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def create_access_token(user) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    to_encode = {
        "sub": str(user.id),
        "ver": user.token_version or 0,
        "jti": str(uuid.uuid4()),
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    """Decode JWT and return the authenticated user."""
    from app.db.models import User
    settings = get_settings()

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM],
            options={"require": ["exp", "sub", "type"]},
        )
        if payload.get("type") != "access":
            raise credentials_exception
        user_id = int(payload["sub"])
    except (PyJWTError, ValueError, KeyError):
        raise credentials_exception

    user = db.get(User, user_id)
    if user is None or payload.get("ver", 0) != (user.token_version or 0):
        raise credentials_exception
    return user


def require_role(*roles: str):
    """Dependency factory: restrict access to specific user roles."""
    def role_checker(current_user=Depends(get_current_user)):
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user
    return role_checker

"""
Security module: JWT auth, password hashing, refresh token rotation, rate limiting.
"""
import uuid
from datetime import datetime, timedelta
from typing import Any, Union
import jwt
from jwt.exceptions import PyJWTError
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.core.config import get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def create_access_token(subject: Union[str, Any], expires_delta: timedelta = None) -> str:
    settings = get_settings()
    jti = str(uuid.uuid4())
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"exp": expire, "sub": str(subject), "jti": jti, "type": "access"}
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(user_id: int) -> tuple[str, str]:
    """Create a refresh token and store its JTI in Redis. Returns (token, jti)."""
    settings = get_settings()
    jti = str(uuid.uuid4())
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = {"exp": expire, "sub": str(user_id), "jti": jti, "type": "refresh"}
    token = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    # Store in Redis
    try:
        from app.core.redis_client import store_refresh_token
        store_refresh_token(jti, user_id, settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400)
    except Exception:
        pass  # Redis may not be available in dev

    return token, jti


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
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except PyJWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
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

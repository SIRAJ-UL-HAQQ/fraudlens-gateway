from datetime import UTC, datetime, timedelta
from typing import Annotated

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

security = HTTPBearer(auto_error=False)

# Hash password once at startup
_RAW_PASSWORD = settings.get_analyst_password()
_HASHED_ANALYST_PASSWORD: bytes = bcrypt.hashpw(
    _RAW_PASSWORD.encode("utf-8"),
    bcrypt.gensalt(),
)


def verify_password(plain_password: str) -> bool:
    """Verify password using constant-time bcrypt comparison."""
    if not plain_password:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            _HASHED_ANALYST_PASSWORD,
        )
    except (ValueError, TypeError):
        return False


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> str:
    """Generate a signed JWT access token."""
    to_encode = data.copy()
    now = datetime.now(UTC)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(seconds=settings.JWT_EXPIRES_IN_SECONDS)

    to_encode.update({"exp": expire, "iat": now})
    return jwt.encode(
        to_encode,
        settings.get_jwt_secret_key(),
        algorithm=settings.JWT_ALGORITHM,
    )


def create_test_token(
    username: str = "analyst_1",
    role: str = "analyst",
    expires_delta: timedelta | None = None,
) -> str:
    """Helper for tests to generate a valid signed JWT."""
    return create_access_token(
        data={"sub": username, "username": username, "role": role},
        expires_delta=expires_delta,
    )


def get_current_analyst(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)] = None,
) -> dict:
    """Validate bearer JWT token for protected endpoints.

    Raises HTTP 401 if token is missing, invalid, or expired.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            settings.get_jwt_secret_key(),
            algorithms=[settings.JWT_ALGORITHM],
        )
        username = payload.get("username") or payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {
            "username": username,
            "role": payload.get("role", "analyst"),
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

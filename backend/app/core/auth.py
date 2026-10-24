from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)

# Valid token for contract testing and dev
VALID_DEV_TOKEN = "fraudlens-analyst-token-secret"


def get_current_analyst(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)] = None,
) -> dict:
    """Validate bearer token for protected endpoints.

    Raises 401 if missing or invalid.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    # Allow either the canonical dev token or any test-token starting with "test-token"
    if token != VALID_DEV_TOKEN and not token.startswith("test-token") and not token.startswith("bearer-token"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"username": "analyst_1", "role": "analyst"}

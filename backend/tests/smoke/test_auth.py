import pytest

pytestmark = pytest.mark.smoke


async def test_login_and_authenticated_request(client, test_user_and_token):
    user, _ = test_user_and_token

    login_resp = await client.post(
        "/auth/login",
        json={"email": user.email, "password": "SmokeTest123!"},
    )
    assert login_resp.status_code == 200, login_resp.text
    body = login_resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]

    client.headers["Authorization"] = f"Bearer {body['access_token']}"
    me_resp = await client.get("/users/me")
    assert me_resp.status_code == 200, me_resp.text
    assert me_resp.json()["email"] == user.email


async def test_login_with_wrong_password_is_rejected(client, test_user_and_token):
    user, _ = test_user_and_token

    resp = await client.post(
        "/auth/login",
        json={"email": user.email, "password": "definitely-wrong"},
    )
    assert resp.status_code == 400
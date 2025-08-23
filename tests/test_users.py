from fastapi.testclient import TestClient
from api.main import app
from api.database import get_db, Base, engine
from api.models import User
from sqlalchemy.orm import sessionmaker

# Use a separate test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

def test_create_user():
    response = client.post(
        "/users/",
        json={"username": "testuser", "email": "test@example.com", "password": "testpassword"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "id" in data
    user_id = data["id"]

    # Verify the user was actually created in the test database
    db = TestingSessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    assert user is not None
    assert user.username == "testuser"
    db.close()
from fastapi import FastAPI
from app.api.routes import router
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warmup: Load model on startup
    print("🔥 Warming up model...")
    try:
        from app.models.model_loader import predict_sms
        predict_sms("test")
        print("✅ Model ready!")
    except Exception as e:
        print(f"⚠️ Warmup failed: {e}")
    yield

app = FastAPI(
    title="AI Phishing SMS API",
    version="1.0",
    lifespan=lifespan
)

app.include_router(router)
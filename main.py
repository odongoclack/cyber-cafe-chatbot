import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Annotated
from dotenv import load_dotenv
from datetime import datetime
import uuid

# Langchain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser # THIS IMPORT IS CRITICAL

# SQLAlchemy imports for PostgreSQL
from sqlalchemy import create_engine, Column, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.dialects.postgresql import UUID

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL") # e.g., "postgresql://user:password@host:port/dbname"

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not found in environment variables. "
          "The AI model will not function. Please set it in a .env file or directly.")
if not DATABASE_URL:
    print("WARNING: DATABASE_URL not found in environment variables. "
          "Conversation history will NOT be saved to PostgreSQL.")

# Initialize FastAPI app
app = FastAPI(
    title="E-C Digital Hub AI Assistant Backend",
    description="Backend for the E-C Digital Hub chatbot, powered by Langchain, GPT-4o mini, and PostgreSQL."
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Update this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SQLAlchemy Database Setup ---
# Create a SQLAlchemy engine
engine = create_engine(DATABASE_URL) if DATABASE_URL else None

# Create a SessionLocal class for database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None

# Base class for SQLAlchemy models
Base = declarative_base()

# Define database models
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_name = Column(String, default="Guest")
    created_at = Column(DateTime, default=datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    sender = Column(String) # 'user' or 'bot'
    text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Function to create database tables
def create_db_tables():
    if engine:
        print("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("Database tables created (if they didn't exist).")
    else:
        print("Database URL not configured. Skipping table creation.")

# Dependency to get a database session
def get_db():
    db = SessionLocal() if SessionLocal else None
    try:
        yield db
    finally:
        if db:
            db.close()

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    create_db_tables()

# --- Pydantic Model for Request Body ---
class ChatRequest(BaseModel):
    message: str
    user_name: str = "Guest"
    conversation_id: str # This will be used to link messages to a conversation

# --- Langchain Setup ---
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are E-C Digital Hub's AI-powered virtual assistant in Nairobi, Kenya.
    Your goal is to provide helpful, accurate, and friendly information about the cyber cafe's services, rates, and facilities.
    You have advanced local knowledge and can answer questions about:
    - High-Speed Internet browsing (KSh 50/hour)
    - Premium Gaming Zone (KSh 80/hour for high-spec PCs, latest titles like FIFA 24, Call of Duty, Fortnite, Valorant)
    - Professional Printing (Color & B&W, KSh 30/hour for typing/printing)
    - Computer Training (KSh 100/hour, Beginner to Advanced)
    - Device Services (Repair & Troubleshooting)
    - Phone Charging (KSh 20/hour)
    - Laminating & Binding
    - Refreshment Zone (Complimentary for VIP)
    - 24/7 Tech Support
    - Event Hosting (Gaming tournaments)
    - Digital & Stationery Shop (USBs, cables, flash drives, earphones, basic office supplies)
    - Operating Hours: 24/7 (Morning Rush: 6AM-10AM, Day Sessions: 10AM-6PM, Prime Time: 6PM-12AM, Night Owls: 12AM-6AM)
    - Booking: Call +254-701-161779 or +254-112-670912, or use the app.
    - Address: Digital Plaza, Nairobi CBD.
    - Website: www.ecdigitalhub.co.ke
    - VIP Packages: Daily (KSh 400), Weekly (KSh 2,500), Monthly (KSh 8,000) - include complimentary refreshments and priority support.
    
    Always be polite, professional, and concise. Use Markdown for formatting where appropriate (e.g., bolding, lists).
    If a user asks about something not related to the cyber cafe, politely redirect them to services you offer.
    If you cannot answer a question based on the provided information, suggest they contact the cyber cafe directly.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser() # StrOutputParser is used here

# --- API Endpoint ---
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, db: Annotated[Session, Depends(get_db)]):
    """
    Handles chat messages from the frontend, processes them with Langchain/GPT-4o mini,
    and returns an AI-generated response. Conversation history is persisted to PostgreSQL.
    """
    user_message_text = request.message
    user_name = request.user_name
    conv_id_str = request.conversation_id

    if not db:
        raise HTTPException(status_code=500, detail="Database connection not available.")

    # Convert conversation_id string to UUID
    try:
        conv_uuid = uuid.UUID(conv_id_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID format.")

    # Check if conversation exists, create if not
    conversation = db.query(Conversation).filter(Conversation.id == conv_uuid).first()
    if not conversation:
        conversation = Conversation(id=conv_uuid, user_name=user_name)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        print(f"Created new conversation: {conversation.id}")

    # Load past messages from the database for this conversation
    db_messages = db.query(Message).filter(Message.conversation_id == conv_uuid).order_by(Message.timestamp).all()
    current_chat_history = []
    for msg in db_messages:
        if msg.sender == "user":
            current_chat_history.append(HumanMessage(content=msg.text))
        elif msg.sender == "bot":
            current_chat_history.append(AIMessage(content=msg.text))

    # Add the current user message to the database
    user_message_db = Message(conversation_id=conv_uuid, sender="user", text=user_message_text)
    db.add(user_message_db)
    db.commit()
    db.refresh(user_message_db)

    try:
        ai_response = await chain.ainvoke({
            "input": user_message_text,
            "user_name": user_name,
            "chat_history": current_chat_history
        })

        # Add the AI's response to the database
        bot_message_db = Message(conversation_id=conv_uuid, sender="bot", text=ai_response)
        db.add(bot_message_db)
        db.commit()
        db.refresh(bot_message_db)

        return {"response": ai_response}

    except Exception as e:
        print(f"Error processing chat message: {e}")
        db.rollback() # Rollback changes if AI call fails
        raise HTTPException(status_code=500, detail="Error generating AI response. Please try again later.")

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
      return {"status": "ok", "message": "E-C Digital Hub AI Backend is running!"}
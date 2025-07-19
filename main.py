"""
E-C Digital Hub AI Assistant Backend
Enhanced, well-organized FastAPI application with admin features
"""

import os
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Annotated, Optional

from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from sqlalchemy import create_engine, Column, String, DateTime, Text, ForeignKey, func
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship
from sqlalchemy.dialects.postgresql import UUID


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

ADMIN_USERS = {
    "edwin123": "admin_password_hash_edwin",
    "clacks123": "admin_password_hash_clacks"
}

if not OPENAI_API_KEY:
    print("⚠️  WARNING: OPENAI_API_KEY not found. AI features will not work.")
if not DATABASE_URL:
    print("⚠️  WARNING: DATABASE_URL not found. Database features disabled.")



def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_admin(username: str) -> bool:
    """Verify if user is admin"""
    return username.lower() in ADMIN_USERS

# Database setup
engine = create_engine(DATABASE_URL) if DATABASE_URL else None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None
Base = declarative_base()

class Conversation(Base):
    """Conversation model for storing chat sessions"""
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_name = Column(String, default="Guest")
    is_admin = Column(String, default="false")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    """Message model for storing individual chat messages"""
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    sender = Column(String)  # 'user' or 'bot'
    text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

class ServiceUpdate(Base):
    """Service update model for tracking price changes"""
    __tablename__ = "service_updates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_name = Column(String)
    old_price = Column(String)
    new_price = Column(String)
    updated_by = Column(String)
    updated_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)

# PYDANTIC MODELS (API SCHEMAS)


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=1000)
    user_name: str = Field(default="Guest", max_length=50)
    conversation_id: str = Field(..., description="UUID of the conversation")

class AdminLoginRequest(BaseModel):
    """Request model for admin login"""
    username: str = Field(..., min_length=3, max_length=20)

class ServiceUpdateRequest(BaseModel):
    """Request model for service updates"""
    service_name: str = Field(..., min_length=1, max_length=100)
    old_price: str = Field(..., max_length=50)
    new_price: str = Field(..., max_length=50)
    notes: str = Field(default="", max_length=500)

class ConversationResponse(BaseModel):
    """Response model for conversation data"""
    id: str
    user_name: str
    is_admin: str
    created_at: datetime
    last_activity: datetime
    message_count: int

class MessageResponse(BaseModel):
    """Response model for message data"""
    id: str
    sender: str
    text: str
    timestamp: datetime
    user_name: str

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    is_admin: bool
    username: str

class AdminLoginResponse(BaseModel):
    """Response model for admin login"""
    success: bool
    message: str
    is_admin: bool
    username: Optional[str] = None

def get_db():
    """Database dependency for FastAPI"""
    if not SessionLocal:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection not available"
        )
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_db_tables():
    """Create database tables"""
    if engine:
        print("📊 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
    else:
        print("⚠️  Database URL not configured. Skipping table creation.")

# AI/LLM CONFIGURATION
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    api_key=OPENAI_API_KEY, 
    temperature=0.7
) if OPENAI_API_KEY else None

# System prompts
CUSTOMER_SYSTEM_PROMPT = """You are E-C Digital Hub's advanced AI-powered virtual assistant in Bondo, Siaya County, Kenya.
You provide helpful, accurate, and friendly information about our premium cyber cafe services, rates, and facilities.

🏢 **BUSINESS INFORMATION**
- Name: E-C Digital Hub
- Location: Bondo, Siaya County, Kenya
- Address: Bondo Town, Siaya County, Kenya
- Owners: Edwin and Clacks Ager
- Contact: +254-701-161779 or +254-112-670912
- Website: www.ecdigitalhub.co.ke
- Operating Hours: 24/7 Service

💻 **CORE SERVICES & RATES**
- **High-Speed Internet**: KSh 50/hour (Fiber optic, 100+ Mbps)
- **Premium Gaming Zone**: KSh 80/hour (Latest titles: FIFA 24, Call of Duty, Fortnite, Valorant, Apex Legends)
- **Professional Printing & Typing**: KSh 30/hour (Color & B&W, All formats)
- **Computer Training**: KSh 100/hour (Beginner to Advanced - MS Office, Programming, Design)
- **Device Repair & Troubleshooting**: Starting KSh 500 (Phones, Laptops, PCs)
- **Phone Charging Station**: KSh 20/hour (Multiple ports, fast charging)
- **Laminating & Binding**: KSh 50-200 (Professional finish)
- **Photocopying & Scanning**: KSh 5/page (High quality)

🎮 **GAMING FEATURES**
- High-spec Gaming PCs (RTX Graphics, 16GB+ RAM)
- Mechanical keyboards & Gaming mice
- 27" Curved monitors with high refresh rates
- Gaming tournaments & events
- Discord & team communication support

💎 **VIP PACKAGES**
- **Daily VIP**: KSh 400 (Unlimited internet, complimentary refreshments, priority support)
- **Weekly VIP**: KSh 2,500 (All daily benefits + reserved seating)
- **Monthly VIP**: KSh 8,000 (Premium perks + exclusive access to new games first)

🛒 **DIGITAL & STATIONERY SHOP**
- USB Flash drives (8GB-128GB): KSh 500-2,500
- Phone accessories (Cables, earphones, cases): KSh 200-1,500
- Computer accessories (Mouse, keyboards): KSh 800-3,000
- Office supplies (Pens, papers, folders): KSh 50-500
- Memory cards & adapters: KSh 400-2,000

⏰ **TIME SLOTS & PEAK HOURS**
- Morning Rush (6AM-10AM): Business professionals, students
- Day Sessions (10AM-6PM): Regular computing, training
- Prime Time (6PM-12AM): Gaming peak hours, entertainment
- Night Owls (12AM-6AM): Gamers, late workers (20% night discount)

🍕 **REFRESHMENT ZONE**
- Complimentary for VIP members
- Snacks, beverages, light meals available
- Clean, comfortable seating area

📱 **BOOKING & SUPPORT**
- Call ahead for reservations: +254-701-161779 or +254-112-670912
- WhatsApp booking available
- Mobile app for easy booking (mention: "Download our app!")
- 24/7 technical support
- On-site technician always available

🎯 **SPECIAL FEATURES**
- Free WiFi in waiting area
- Air-conditioned environment
- CCTV security
- Backup power (UPS & Generator)
- Student discounts (10% with valid ID)
- Group bookings available
- Corporate training packages

**COMMUNICATION STYLE**
- Be warm, professional, and enthusiastic about our services
- Use emojis appropriately to make responses engaging
- Provide specific pricing and encourage bookings
- If asked about something not listed, suggest contacting us directly
- Always end with a helpful suggestion or call-to-action
- Use Kenyan context and local references when appropriate

For technical issues, pricing inquiries, or bookings, always encourage customers to call or visit us!"""

ADMIN_SYSTEM_PROMPT = """You are E-C Digital Hub's AI assistant in ADMIN MODE for business owners Edwin and Clacks.

As an admin user, you have access to:
- Business analytics and insights
- Service management recommendations  
- Customer behavior analysis
- Revenue optimization suggestions
- Operational improvements

You can provide:
- Detailed business metrics interpretation
- Suggestions for service pricing adjustments
- Customer satisfaction analysis
- Peak hours optimization
- Marketing recommendations
- Cost management advice

Always address admin users professionally and provide business-focused insights.
When providing regular customer service info, also include business context."""

# Create prompt templates
customer_prompt = ChatPromptTemplate.from_messages([
    ("system", CUSTOMER_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

admin_prompt = ChatPromptTemplate.from_messages([
    ("system", ADMIN_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Create chains
customer_chain = customer_prompt | llm | StrOutputParser() if llm else None
admin_chain = admin_prompt | llm | StrOutputParser() if llm else None


# FASTAPI APPLICATION SETUP

app = FastAPI(
    title="E-C Digital Hub AI Assistant Backend",
    description="Enhanced backend for the E-C Digital Hub chatbot with admin features, powered by Langchain, GPT-4o mini, and PostgreSQL.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://e-cdigitalhub.vercel.app/"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ================================
# EVENT HANDLERS
# ================================

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("🚀 Starting E-C Digital Hub AI Assistant Backend...")
    create_db_tables()
    print("✅ Application startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    print("🔄 Shutting down E-C Digital Hub AI Assistant Backend...")
    if engine:
        engine.dispose()
    print("✅ Shutdown complete!")

# ================================
# API ROUTES - MAIN ENDPOINTS
# ================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "Welcome to E-C Digital Hub AI Assistant Backend! 🚀",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "E-C Digital Hub AI Backend is running smoothly! 💪",
        "features": [
            "Enhanced AI Chat",
            "Admin Dashboard", 
            "Analytics & Insights",
            "Service Management"
        ],
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(
    request: ChatRequest, 
    db: Annotated[Session, Depends(get_db)]
):
    """Main chat endpoint with AI response generation"""
    
    # Validate AI availability
    if not customer_chain or not admin_chain:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is currently unavailable. Please check configuration."
        )
    
    # Parse and validate conversation ID
    try:
        conv_uuid = uuid.UUID(request.conversation_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid conversation ID format. Must be a valid UUID."
        )
    
    # Check admin status
    is_admin = verify_admin(request.user_name.lower())
    is_admin_str = "true" if is_admin else "false"
    
    # Find or create conversation
    conversation = db.query(Conversation).filter(Conversation.id == conv_uuid).first()
    
    if not conversation:
        # Create new conversation
        conversation = Conversation(
            id=conv_uuid,
            user_name=request.user_name,
            is_admin=is_admin_str,
            last_activity=datetime.utcnow()
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        print(f"📝 Created new conversation: {conversation.id} (Admin: {is_admin})")
    else:
        # Update existing conversation
        conversation.last_activity = datetime.utcnow()
        conversation.is_admin = is_admin_str
        db.commit()
    
    # Retrieve conversation history
    db_messages = (
        db.query(Message)
        .filter(Message.conversation_id == conv_uuid)
        .order_by(Message.timestamp)
        .all()
    )
    
    # Convert to LangChain message format
    chat_history = []
    for msg in db_messages:
        if msg.sender == "user":
            chat_history.append(HumanMessage(content=msg.text))
        elif msg.sender == "bot":
            chat_history.append(AIMessage(content=msg.text))
    
    # Save user message to database
    user_message = Message(
        conversation_id=conv_uuid,
        sender="user",
        text=request.message
    )
    db.add(user_message)
    db.commit()
    
    try:
        # Generate AI response using appropriate chain
        if is_admin:
            ai_response = await admin_chain.ainvoke({
                "input": request.message,
                "user_name": request.user_name,
                "chat_history": chat_history
            })
        else:
            ai_response = await customer_chain.ainvoke({
                "input": request.message,
                "user_name": request.user_name,
                "chat_history": chat_history
            })
        
        # Save bot response to database
        bot_message = Message(
            conversation_id=conv_uuid,
            sender="bot",
            text=ai_response
        )
        db.add(bot_message)
        db.commit()
        
        return ChatResponse(
            response=ai_response,
            is_admin=is_admin,
            username=request.user_name
        )
        
    except Exception as e:
        print(f"❌ Error generating AI response: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate AI response. Please try again."
        )

# ================================
# API ROUTES - ADMIN ENDPOINTS
# ================================

@app.post("/api/admin/login", response_model=AdminLoginResponse, tags=["Admin"])
async def admin_login(request: AdminLoginRequest):
    """Admin authentication endpoint"""
    username = request.username.lower()
    
    if verify_admin(username):
        return AdminLoginResponse(
            success=True,
            message=f"Welcome back, {username}! 🔑 Admin access granted.",
            is_admin=True,
            username=username
        )
    else:
        return AdminLoginResponse(
            success=False,
            message="❌ Access denied. Invalid admin credentials.",
            is_admin=False
        )

@app.get("/api/admin/dashboard", tags=["Admin"])
async def admin_dashboard(
    username: str = Query(..., description="Admin username"),
    db: Session = Depends(get_db)
):
    """Comprehensive admin dashboard with analytics"""
    
    # Verify admin access
    if not verify_admin(username.lower()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # Calculate date ranges
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)
    
    # Basic statistics
    total_conversations = db.query(Conversation).count()
    total_messages = db.query(Message).count()
    
    recent_conversations = (
        db.query(Conversation)
        .filter(Conversation.created_at >= week_ago)
        .count()
    )
    
    recent_messages = (
        db.query(Message)
        .filter(Message.timestamp >= week_ago)
        .count()
    )
    
    # Top active users (excluding admins)
    active_users = (
        db.query(
            Conversation.user_name,
            func.count(Message.id).label('message_count')
        )
        .join(Message)
        .filter(
            Conversation.is_admin == "false",
            Message.timestamp >= month_ago
        )
        .group_by(Conversation.user_name)
        .order_by(func.count(Message.id).desc())
        .limit(10)
        .all()
    )
    
    # Daily activity for the last 7 days
    daily_activity = []
    for i in range(7):
        date = (now - timedelta(days=i)).date()
        day_messages = (
            db.query(Message)
            .filter(func.date(Message.timestamp) == date)
            .count()
        )
        daily_activity.append({
            "date": date.strftime("%Y-%m-%d"),
            "day_name": date.strftime("%A"),
            "messages": day_messages
        })
    
    return {
        "overview": {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "recent_conversations": recent_conversations,
            "recent_messages": recent_messages,
            "avg_messages_per_conversation": round(total_messages / max(total_conversations, 1), 2)
        },
        "active_users": [
            {"name": name, "messages": count} 
            for name, count in active_users
        ],
        "daily_activity": daily_activity,
        "generated_at": now.isoformat(),
        "period": "Last 30 days"
    }

@app.get("/api/admin/conversations", response_model=List[ConversationResponse], tags=["Admin"])
async def get_conversations(
    username: str = Query(..., description="Admin username"),
    limit: int = Query(50, le=200, description="Maximum number of conversations to return"),
    db: Session = Depends(get_db)
):
    """Retrieve recent conversations for admin review"""
    
    # Verify admin access
    if not verify_admin(username.lower()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # Query conversations with message counts
    conversations = (
        db.query(
            Conversation,
            func.count(Message.id).label('message_count')
        )
        .outerjoin(Message)
        .group_by(Conversation.id)
        .order_by(Conversation.last_activity.desc())
        .limit(limit)
        .all()
    )
    
    return [
        ConversationResponse(
            id=str(conv.id),
            user_name=conv.user_name,
            is_admin=conv.is_admin,
            created_at=conv.created_at,
            last_activity=conv.last_activity,
            message_count=count or 0
        )
        for conv, count in conversations
    ]

@app.get("/api/admin/conversation/{conversation_id}/messages", tags=["Admin"])
async def get_conversation_messages(
    conversation_id: str,
    username: str = Query(..., description="Admin username"),
    db: Session = Depends(get_db)
):
    """Get detailed messages for a specific conversation"""
    
    # Verify admin access
    if not verify_admin(username.lower()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # Validate conversation ID
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid conversation ID format"
        )
    
    # Find conversation
    conversation = db.query(Conversation).filter(Conversation.id == conv_uuid).first()
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Get all messages for this conversation
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conv_uuid)
        .order_by(Message.timestamp)
        .all()
    )
    
    return {
        "conversation": {
            "id": str(conversation.id),
            "user_name": conversation.user_name,
            "is_admin": conversation.is_admin,
            "created_at": conversation.created_at,
            "last_activity": conversation.last_activity,
            "total_messages": len(messages)
        },
        "messages": [
            MessageResponse(
                id=str(msg.id),
                sender=msg.sender,
                text=msg.text,
                timestamp=msg.timestamp,
                user_name=conversation.user_name
            )
            for msg in messages
        ]
    }

@app.post("/api/admin/service-update", tags=["Admin"])
async def update_service(
    request: ServiceUpdateRequest,
    username: str = Query(..., description="Admin username"),
    db: Session = Depends(get_db)
):
    """Record service updates for tracking and future LLM training"""
    
    # Verify admin access
    if not verify_admin(username.lower()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # Create service update record
    service_update = ServiceUpdate(
        service_name=request.service_name,
        old_price=request.old_price,
        new_price=request.new_price,
        updated_by=username,
        notes=request.notes
    )
    
    db.add(service_update)
    db.commit()
    db.refresh(service_update)
    
    return {
        "success": True,
        "message": f"✅ Service '{request.service_name}' updated: {request.old_price} → {request.new_price}",
        "update_id": str(service_update.id),
        "updated_at": service_update.updated_at.isoformat()
    }

@app.get("/api/admin/recent-activity", tags=["Admin"])
async def get_recent_activity(
    username: str = Query(..., description="Admin username"),
    hours: int = Query(24, le=168, description="Hours to look back (max 168 = 1 week)"),
    db: Session = Depends(get_db)
):
    """Monitor recent system activity"""
    
    # Verify admin access
    if not verify_admin(username.lower()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # Calculate time range
    since = datetime.utcnow() - timedelta(hours=hours)
    
    # Get recent messages with user context
    recent_messages = (
        db.query(
            Message.text,
            Message.sender,
            Message.timestamp,
            Conversation.user_name,
            Conversation.is_admin
        )
        .join(Conversation)
        .filter(Message.timestamp >= since)
        .order_by(Message.timestamp.desc())
        .limit(100)
        .all()
    )
    
    # Format activity data
    activity = []
    for msg in recent_messages:
        text_preview = msg.text[:100] + "..." if len(msg.text) > 100 else msg.text
        activity.append({
            "text": text_preview,
            "sender": msg.sender,
            "timestamp": msg.timestamp,
            "user_name": msg.user_name,
            "is_admin_user": msg.is_admin == "true"
        })
    
    return {
        "activity": activity,
        "period_hours": hours,
        "total_messages": len(activity),
        "query_time": datetime.utcnow().isoformat()
    }

# ERROR HANDLERS

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return {
        "error": "Endpoint not found",
        "message": "The requested resource does not exist",
        "available_endpoints": [
            "/docs - API Documentation",
            "/health - Health Check",
            "/api/chat - Main Chat Endpoint"
        ]
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again later.",
        "support": "Contact support at +254-701-161779"
    }

# APPLICATION METADATA

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting E-C Digital Hub AI Assistant Backend...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
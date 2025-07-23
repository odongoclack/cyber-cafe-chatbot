import os
import uuid
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Annotated, Optional
from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import create_engine, Column, String, DateTime, Text, ForeignKey, func
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
import time


# ================================
# CONFIGURATION & SETUP
# ================================
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hashed admin passwords for local testing.
ADMIN_USERS = {
    "edwin123": hashlib.sha256("admin_password_for_edwin".encode()).hexdigest(),
    "clacks123": hashlib.sha256("admin_password_for_clacks".encode()).hexdigest()
}

if not ANTHROPIC_API_KEY:
    logging.warning("⚠️  WARNING: ANTHROPIC_API_KEY not found. AI features will not work.")
if not DATABASE_URL:
    logging.warning("⚠️  WARNING: DATABASE_URL not found. Database features disabled.")


def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_admin(username: str) -> bool:
    """Verify if user is an admin by checking against hardcoded list"""
    return username.lower() in ADMIN_USERS


# Database setup
if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
else:
    engine = None
    SessionLocal = None
    Base = None

if Base:
    class Conversation(Base):
        __tablename__ = "conversations"
        id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        user_name = Column(String, default="Guest")
        is_admin = Column(String, default="false")
        created_at = Column(DateTime, default=datetime.utcnow)
        last_activity = Column(DateTime, default=datetime.utcnow)
        messages = relationship("Message", back_populates="conversation")

    class Message(Base):
        __tablename__ = "messages"
        id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        conversation_id = Column(PG_UUID(as_uuid=True), ForeignKey("conversations.id"))
        sender = Column(String)
        text = Column(Text)
        timestamp = Column(DateTime, default=datetime.utcnow)
        conversation = relationship("Conversation", back_populates="messages")

    class ServiceUpdate(Base):
        __tablename__ = "service_updates"
        id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        service_name = Column(String)
        old_price = Column(String)
        new_price = Column(String)
        updated_by = Column(String)
        updated_at = Column(DateTime, default=datetime.utcnow)
        notes = Column(Text)
else:
    class Conversation: pass
    class Message: pass
    class ServiceUpdate: pass


# PYDANTIC MODELS (API SCHEMAS)
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    user_name: str = Field(default="Guest", max_length=50)
    conversation_id: str = Field(..., description="UUID of the conversation")

class AdminLoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=20)

class ServiceUpdateRequest(BaseModel):
    service_name: str = Field(..., min_length=1, max_length=100)
    old_price: str = Field(..., max_length=50)
    new_price: str = Field(..., max_length=50)
    notes: str = Field(default="", max_length=500)

class ConversationResponse(BaseModel):
    id: str
    user_name: str
    is_admin: str
    created_at: datetime
    last_activity: datetime
    message_count: int

class MessageResponse(BaseModel):
    id: str
    sender: str
    text: str
    timestamp: datetime
    user_name: str

class ChatResponse(BaseModel):
    response: str
    is_admin: bool
    username: str

class AdminLoginResponse(BaseModel):
    success: bool
    message: str
    is_admin: bool
    username: Optional[str] = None

# UTILITIES & KNOWLEDGE BASE
class CyberCafeKnowledge:
    """Knowledge base for cyber cafe services"""
    
    def __init__(self):
        self.services = {
            "printing": "We offer high-quality printing services: Black & white (KES 5/page), Color printing (KES 20/page), Lamination, Binding, Photocopying available.",
            "internet": "High-speed internet access available. Rates: KES 2/minute or KES 100/hour. Free WiFi for customers using other services.",
            "computer_services": "Computer repair, software installation, virus removal, data recovery, typing services, CV formatting available.",
            "scanning": "Document scanning services: KES 10/page for regular documents, KES 20/page for photos. Email delivery available.",
            "gaming": "Gaming section with latest games. Rates: KES 50/hour. Popular games: FIFA, GTA, Call of Duty, Fortnite.",
            "training": "Computer training available: Basic computer skills, Microsoft Office, Internet usage. Contact us for scheduling.",
            "mobile": "Mobile services: Airtime, mobile money transactions, phone charging (KES 20), phone accessories available.",
            "stationery": "Office supplies available: Pens, papers, folders, flash drives, CDs/DVDs, printer cartridges."
        }
    
    def get_info(self, query: str) -> str:
        """Searches the knowledge base for relevant info based on a query."""
        query_lower = query.lower()
        relevant_info = []
        
        for service, info in self.services.items():
            if any(word in query_lower for word in service.split('_')) or any(word in query_lower for word in ['print', 'internet', 'computer', 'scan', 'game', 'train', 'mobile', 'station']):
                if service in query_lower or any(key in query_lower for key in service.split('_')):
                    relevant_info.append(info)
        
        return "\n".join(relevant_info) if relevant_info else ""

# Instantiate the knowledge base globally
knowledge_base = CyberCafeKnowledge()


# AI/LLM CONFIGURATION
llm = ChatAnthropic(
    model="claude-3-5-haiku-latest", 
    anthropic_api_key=ANTHROPIC_API_KEY, 
    temperature=0.7
) if ANTHROPIC_API_KEY else None

CUSTOMER_SYSTEM_PROMPT = """You are an AI assistant for a cyber cafe that provides multiple digital services.
This cyber cafe, E-C Digital Hub, is owned and managed by Clackson Ager.

ABOUT THE OWNER:
- Name: Clackson Ager
- Education: JKUAT student, studying B.Sc. statistics and programming , Software Engineering and Cybersecurity student
- Profession: Fullstack Engineer, AI Engineer, IT Specialist, Graphic Designer
- Experience: 3 years of professional experience
- Skills: Proficient in Python, JavaScript, and other programming languages.
- Contact:
    - Phone: +254112670912
    - Email: agerclackson44@gmail.com
    - Social Media: @clacks
- Location:
    - Born: Kisumu Nyahera, Kenya
    - Lives: Nairobi, Kenya

SERVICES AVAILABLE:
- Printing & Photocopying
- High-speed Internet Access  
- Computer Services & Repairs
- Document Scanning
- Gaming Section
- Computer Training
- Mobile Services (Airtime, M-Pesa)
- Stationery & Supplies

RESPONSE STYLE:
- Be friendly and helpful
- Provide specific pricing when available
- Direct customers to staff for complex issues
- Always ask if they need help with anything else

Current customer: {user_name}
"""

ADMIN_SYSTEM_PROMPT = """You are the admin AI assistant for the E-C Digital Hub cyber cafe management system.
This system was created by Clackson Ager, the owner of E-C Digital Hub.

CREATOR & OWNER DETAILS:
- Name: Clackson Ager
- Education: JKUAT student, B.Sc. Statistics and programming, Software Engineering and Cybersecurity
- Professional Role: AI Engineer, Fullstack Engineer, IT Specialist
- Experience: 3 years
- Skills: Python, JavaScript, and more
- Contact: 
    - Phone: +254112670912
    - Email: agerclackson44@gmail.com
    - Social Media: @clacks
- Location:
    - Born: Kisumu Nyahera, Kenya
    - Lives: Nairobi, Kenya

You can help with:
- Customer service analytics
- Service usage reports  
- Staff management insights
- Revenue tracking
- Equipment status monitoring
- Summary of services: {services_context}

Be professional and provide detailed administrative insights.
Current admin: {user_name}
"""

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


# Database dependency injection
def get_db():
    if not SessionLocal:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available. Check DATABASE_URL."
        )
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_db_tables():
    if engine and Base:
        logging.info("📊 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logging.info("✅ Database tables created successfully")
    else:
        logging.warning("⚠️  Database URL not configured. Skipping table creation.")


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
    logging.info("🚀 Starting E-C Digital Hub AI Assistant Backend...")
    if DATABASE_URL:
        create_db_tables()
    logging.info("✅ Application startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("🔄 Shutting down E-C Digital Hub AI Assistant Backend...")
    if engine:
        engine.dispose()
    logging.info("✅ Shutdown complete!")


# ================================
# API ROUTES - MAIN ENDPOINTS
# ================================
@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Welcome to E-C Digital Hub AI Assistant Backend! 🚀",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["Health"])
async def health_check():
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
    if not customer_chain or not admin_chain:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is currently unavailable. Please check configuration."
        )
    
    try:
        conv_uuid = uuid.UUID(request.conversation_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid conversation ID format. Must be a valid UUID."
        )
    
    is_admin = verify_admin(request.user_name.lower())
    is_admin_str = "true" if is_admin else "false"
    
    conversation = db.query(Conversation).filter(Conversation.id == conv_uuid).first()
    
    if not conversation:
        conversation = Conversation(
            id=conv_uuid,
            user_name=request.user_name,
            is_admin=is_admin_str,
            last_activity=datetime.utcnow()
        )
        db.add(conversation)
        db.commit() 
        db.refresh(conversation)
    else:
        conversation.last_activity = datetime.utcnow()
        conversation.is_admin = is_admin_str
        db.commit()
        db.refresh(conversation)
    
    user_message = Message(
        conversation_id=conv_uuid,
        sender="user",
        text=request.message
    )
    db.add(user_message)

    db_messages = (
        db.query(Message)
        .filter(Message.conversation_id == conv_uuid)
        .order_by(Message.timestamp)
        .all()
    )
    
    chat_history = []
    for msg in db_messages:
        if msg.sender == "user":
            chat_history.append(HumanMessage(content=msg.text))
        elif msg.sender == "bot":
            chat_history.append(AIMessage(content=msg.text))
    
    try:
        start_time = time.time()
        
        # Add cyber cafe context to user input for the LLM
        context = knowledge_base.get_info(request.message)
        enhanced_input = request.message
        if context:
            enhanced_input = f"Available services context:\n{context}\n\nCustomer question: {request.message}"
        
        services_context = "\n".join([f"- {service}: {info}" for service, info in knowledge_base.services.items()])

        if is_admin:
            ai_response = await admin_chain.ainvoke({
                "input": enhanced_input,
                "user_name": request.user_name,
                "chat_history": chat_history,
                "services_context": services_context
            })
        else:
            ai_response = await customer_chain.ainvoke({
                "input": enhanced_input,
                "user_name": request.user_name,
                "chat_history": chat_history
            })
        
        response_time = time.time() - start_time
        logging.info(f"⏱️ Response generated in {response_time:.2f}s for {request.user_name}")
        
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
        logging.error(f"❌ Error generating AI response: {e}")
        db.rollback()
        
        # Fallback response
        fallback_response = "I'm sorry, I'm having technical difficulties right now. Please ask our staff for assistance with printing, internet, or other services."
        
        bot_message = Message(
            conversation_id=conv_uuid,
            sender="bot",
            text=fallback_response
        )
        db.add(bot_message)
        db.commit()
        
        return ChatResponse(
            response=fallback_response,
            is_admin=is_admin,
            username=request.user_name
        )

# ================================
# API ROUTES - ADMIN ENDPOINTS
# ================================

@app.post("/api/admin/login", response_model=AdminLoginResponse, tags=["Admin"])
async def admin_login(request: AdminLoginRequest):
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
    if not verify_admin(username.lower()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)
    
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
    if not verify_admin(username.lower()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
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
    if not verify_admin(username.lower()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid conversation ID format"
        )
    
    conversation = db.query(Conversation).filter(Conversation.id == conv_uuid).first()
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
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
    if not verify_admin(username.lower()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
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
    
@app.post("/api/admin/update-services", tags=["Admin"])
async def update_service_info(
    service_name: str = Query(..., description="Service to update"),
    new_info: str = Query(..., description="Updated service information"),
    username: str = Query(..., description="Admin username"),
    db: Session = Depends(get_db)
):
    """Update cyber cafe service information (Note: Changes are in-memory and will be lost on server restart)"""
    if not verify_admin(username.lower()):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    
    if hasattr(knowledge_base, 'services') and service_name in knowledge_base.services:
        old_info = knowledge_base.services[service_name]
        knowledge_base.services[service_name] = new_info
        
        service_update = ServiceUpdate(
            service_name=service_name,
            old_price=old_info[:50] + "..." if len(old_info) > 50 else old_info,
            new_price=new_info[:50] + "..." if len(new_info) > 50 else new_info,
            updated_by=username,
            notes="Service information updated via admin panel"
        )
        
        db.add(service_update)
        db.commit()
        
        return {
            "success": True,
            "message": f"✅ Service '{service_name}' updated successfully",
            "old_info": old_info,
            "new_info": new_info
        }
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Service '{service_name}' not found")


@app.get("/api/admin/recent-activity", tags=["Admin"])
async def get_recent_activity(
    username: str = Query(..., description="Admin username"),
    hours: int = Query(24, le=168, description="Hours to look back (max 168 = 1 week)"),
    db: Session = Depends(get_db)
):
    if not verify_admin(username.lower()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    since = datetime.utcnow() - timedelta(hours=hours)
    
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
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again later.",
        "support": "Contact support at +254-701-161779"
    }

# APPLICATION METADATA
if __name__ == "__main__":
    import uvicorn
    logging.info("🚀 Starting E-C Digital Hub AI Assistant Backend...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

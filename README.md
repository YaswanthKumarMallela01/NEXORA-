# 🧠 NEXORA — Multi-Agent AI Placement Readiness Platform

![Python](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green) ![LangChain](https://img.shields.io/badge/LangChain-0.3-orange) ![License](https://img.shields.io/badge/License-MIT-purple)

**NEXORA** is an AI-powered career placement platform that uses **4 autonomous LangChain agents** to mentor students through their placement journey — from resume analysis to mock interviews, personalized coaching, and risk detection. Built with a Neural Minimalism UI design system.

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        NEXORA Frontend                           │
│  Login → Dashboard → Coach Chat → Mock Interview → Roadmap → TPC│
│         (Neural Minimalism UI · HTML/CSS/JS)                     │
└────────────────────┬─────────────────────────────────────────────┘
                     │ REST API (FastAPI)
┌────────────────────┴─────────────────────────────────────────────┐
│                     API Layer (FastAPI)                           │
│  /auth/*  /api/resume/*  /api/coach/*  /api/interview/*          │
│  /api/dashboard/*  /api/alerts/*  /api/tasks/*                   │
│  + JWT Auth · Rate Limiting · CORS · Request Logging             │
└────────────────────┬─────────────────────────────────────────────┘
                     │ Agent Orchestration
┌────────────────────┴─────────────────────────────────────────────┐
│               Multi-Agent Orchestrator (LangChain)               │
│                                                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────┐ ┌────────┐│
│  │ ResumeAgent  │ │  CoachAgent  │ │ InterviewAgent │ │ Alert  ││
│  │ (Groq LLM)  │ │ (Groq+Memory)│ │ (Gemini Flash) │ │ Agent  ││
│  │              │ │              │ │                │ │(Groq)  ││
│  │ • PDF Parse  │ │ • Career Q&A │ │ • 5 Questions  │ │• Risk  ││
│  │ • Skills     │ │ • Roadmaps   │ │ • Evaluation   │ │  Eval  ││
│  │ • JD Match   │ │ • Tasks      │ │ • Scoring      │ │• Email ││
│  │ • Scoring    │ │ • Advice     │ │ • Feedback     │ │• Alert ││
│  └──────┬───────┘ └──────┬───────┘ └───────┬────────┘ └───┬────┘│
└─────────┼────────────────┼─────────────────┼──────────────┼─────┘
          │                │                 │              │
┌─────────┴────────────────┴─────────────────┴──────────────┴─────┐
│                     Infrastructure Layer                         │
│                                                                  │
│  ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌─────────────────┐ │
│  │  Supabase  │ │  Pinecone  │ │  Resend  │ │    LLM APIs     │ │
│  │  (Auth+DB) │ │ (Vectors)  │ │ (Email)  │ │ Groq · Gemini   │ │
│  │            │ │            │ │          │ │ Together · HF    │ │
│  └────────────┘ └────────────┘ └──────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 How the Multi-Agent System Works

NEXORA uses **4 specialized AI agents**, each with its own LLM and toolset, orchestrated by a central router:

### 1. 📄 ResumeAgent (`agents/resume_agent.py`)
- **LLM**: Groq (Llama 3.1 70B) — fast reasoning
- **Tools**: `pdf_parser`, `skill_extractor`, `jd_matcher`
- **What it does**: Accepts a Base64-encoded PDF resume, extracts text via PyMuPDF, identifies skills, matches against job descriptions from the RAG knowledge base (Pinecone), and calculates a readiness score
- **Output**: Structured skill profile + readiness percentage

### 2. 🤖 CoachAgent (`agents/coach_agent.py`)
- **LLM**: Groq (Llama 3.1 70B) + Together AI (Mixtral) fallback
- **Tools**: `assign_task`, `generate_roadmap`, `resource_recommender`
- **Memory**: `ConversationBufferWindowMemory` (last 20 messages per user)
- **What it does**: Stateful AI career mentor. Remembers past conversations, assigns personalized tasks, generates weekly study plans, recommends resources, and provides motivational guidance
- **Output**: Contextual career advice with action items

### 3. 🎤 InterviewAgent (`agents/interview_agent.py`)
- **LLM**: Google Gemini 1.5 Flash — large context window for evaluation
- **Tools**: `question_generator`, `answer_evaluator`
- **What it does**: Runs a full mock interview session (5 questions). Generates role-specific technical and behavioral questions, evaluates answers across 5 dimensions (technical depth, communication, problem-solving, specificity, confidence), provides detailed feedback and a running average score
- **Output**: Per-question score (1-10) + detailed feedback + session summary

### 4. 🚨 AlertAgent (`agents/alert_agent.py`)
- **LLM**: Groq (Llama 3.1 70B)
- **Tools**: `risk_evaluator`, `email_sender` (via Resend API)
- **Triggers**: `missed_tasks`, `low_score`, `deadline_approaching`, `inactivity`, `critical_gaps`
- **What it does**: Evaluates student risk based on activity data, generates graduated severity alerts (LOW → MEDIUM → HIGH → CRITICAL), sends personalized nudge emails to students and escalation emails to TPC administrators
- **Output**: Alert record saved to DB + email sent

### Multi-LLM Routing (`chains/orchestrator.py`)
The orchestrator intelligently routes requests:
- **Groq** (primary) — fast responses for resume analysis, coaching, alerts
- **Together AI** (fallback) — activates if Groq is rate-limited or unavailable
- **Gemini Flash** (interviews only) — uses its 1M-token context window for answer evaluation

---

## 📁 Project Structure

```
nexora-backend/
├── agents/
│   ├── resume_agent.py       # PDF analysis → skill extraction → JD matching
│   ├── coach_agent.py        # Stateful career mentor with memory
│   ├── interview_agent.py    # Gemini-powered mock interviews
│   └── alert_agent.py        # Risk evaluation → email notifications
├── rag/
│   ├── ingest.py             # Document loading → chunking → Pinecone upload
│   └── retriever.py          # Semantic search over placement knowledge base
├── chains/
│   └── orchestrator.py       # Multi-LLM routing (Groq → Together → Gemini)
├── auth/
│   └── supabase_auth.py      # OTP email auth + JWT verification
├── api/
│   ├── routes.py             # All REST endpoints (7 routers, 16 endpoints)
│   └── middleware.py          # Rate limiting, CORS, request logging
├── db/
│   ├── supabase_client.py    # Database CRUD helpers
│   └── schema.sql            # Supabase table definitions
├── frontend/
│   ├── login.html            # Landing + OTP authentication
│   ├── dashboard.html        # Student dashboard (real API data)
│   ├── coach.html            # AI coach chat interface
│   ├── interview.html        # Mock interview simulator
│   ├── roadmap.html          # Task pipeline & progress tracker
│   └── tpc.html              # TPC admin panel (role-gated)
├── config.py                 # Pydantic settings (env vars)
├── main.py                   # FastAPI app entry point
├── requirements.txt          # Python dependencies
├── .env                      # API keys & secrets (not committed)
├── .gitignore                # Git ignore rules
├── vercel.json               # Vercel deployment config
└── README.md                 # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A Supabase project (free tier works)
- API keys: Groq, Google AI (Gemini), Together AI, Pinecone, Resend, HuggingFace

### 1. Clone & Setup

```bash
git clone https://github.com/YaswanthKumarMallela01/NEXORA-.git
cd nexora-backend

# Create virtual environment (Python 3.10)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the root directory:

```env
# ── LLM APIs ──
GROQ_API_KEY="gsk_..."
GOOGLE_API_KEY="AIza..."
HUGGINGFACE_API_KEY="hf_..."
TOGETHER_API_KEY="key_..."

# ── Vector Store (Pinecone) ──
PINECONE_API_KEY="pcsk_..."
PINECONE_HOST="https://your-index.svc.pinecone.io"
PINECONE_INDEX_NAME="nexora-rag"

# ── Database + Auth (Supabase) ──
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_ANON_KEY="eyJ..."
SUPABASE_SERVICE_ROLE_KEY="eyJ..."

# ── Email (Resend) ──
RESEND_API_KEY="re_..."
FROM_EMAIL="onboarding@resend.dev"

# ── Automation ──
N8N_WEBHOOK_URL="https://your-n8n.app.n8n.cloud/webhook/..."
N8N_API_KEY="your-webhook-key"

# ── App Config ──
JWT_SECRET="your-secret-key"
ENVIRONMENT="development"
```

### 3. Setup Database

Run the SQL from `db/schema.sql` in your Supabase SQL Editor to create the required tables:
- `students` — User profiles with readiness scores
- `tasks` — AI-assigned tasks with status tracking
- `interview_sessions` — Mock interview records
- `alerts` — Risk alerts with severity levels

### 4. Ingest RAG Knowledge Base (Optional)

Place placement-related documents (PDFs, text files) in a `placement_resources/` folder, then run:

```bash
python -m rag.ingest
```

This chunks documents and uploads embeddings to Pinecone for the RAG retriever.

### 5. Start the Server

```bash
python -m uvicorn main:app --reload --port 8000
```

Open your browser:
- **App**: http://localhost:8000 → redirects to login
- **API Docs**: http://localhost:8000/docs (Swagger UI)

---

## 🌐 API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/auth/send-otp` | Public | Send OTP to email |
| `POST` | `/auth/verify-otp` | Public | Verify OTP → returns JWT |
| `POST` | `/api/resume/analyze` | JWT | Upload & analyze resume PDF |
| `POST` | `/api/coach/chat` | JWT | Chat with AI career coach |
| `GET` | `/api/coach/memory` | JWT | Get conversation history |
| `POST` | `/api/interview/start` | JWT | Begin mock interview session |
| `POST` | `/api/interview/answer` | JWT | Submit answer → get evaluation |
| `GET` | `/api/interview/summary/{id}` | JWT | Get session summary & scores |
| `GET` | `/api/dashboard/student` | JWT | Full student dashboard data |
| `GET` | `/api/dashboard/tpc` | TPC Role | TPC admin cohort dashboard |
| `POST` | `/api/alerts/trigger` | Webhook | Trigger alert evaluation |
| `POST` | `/api/alerts/acknowledge/{id}` | JWT | Acknowledge an alert |
| `GET` | `/api/tasks/` | JWT | List user's tasks |
| `PATCH` | `/api/tasks/{id}` | JWT | Update task status |
| `GET` | `/health` | Public | System health check |
| `GET` | `/api/info` | Public | API info & version |

---

## 🎨 Frontend Pages

| Page | URL | Description |
|------|-----|-------------|
| Login | `/login` | OTP email authentication |
| Dashboard | `/student` | Readiness score, tasks, skills, alerts |
| AI Coach | `/coach` | Real-time chat with Groq-powered coach |
| Interview | `/interview` | Gemini-powered mock interview simulator |
| Roadmap | `/roadmap` | Task pipeline with status tracking |
| TPC Admin | `/tpc` | Admin dashboard (role-gated) |

All frontend pages fetch **real data from the API** — no hardcoded information.

---

## 🔧 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Backend | FastAPI | REST API framework |
| Agents | LangChain 0.3 | Agent orchestration & tool execution |
| Primary LLM | Groq (Llama 3.1 70B) | Fast reasoning for most agents |
| Interview LLM | Google Gemini 1.5 Flash | Large-context answer evaluation |
| Fallback LLM | Together AI (Mixtral 8x7B) | Backup when Groq is rate-limited |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) | 384-dim vectors for RAG |
| Vector DB | Pinecone | Semantic search over knowledge base |
| Database | Supabase (PostgreSQL) | Users, tasks, sessions, alerts |
| Auth | Supabase Auth | OTP magic-link email authentication |
| Email | Resend | Alert notifications & nudges |
| Frontend | HTML/CSS/JS | Neural Minimalism design system |
| Design | Stitch (Google) | UI generation & design tokens |

---

## 📊 How the Flow Works (End-to-End)

```
Student Signs Up (Email OTP)
        │
        ▼
  ┌─────────────┐
  │  Dashboard   │ ◄── Shows readiness score, tasks, skills, alerts
  │  /student    │     (all data from Supabase via /api/dashboard/student)
  └──────┬──────┘
         │
    ┌────┴────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼
Upload    Chat with   Start Mock   View
Resume    AI Coach    Interview    Roadmap
    │         │          │          │
    ▼         ▼          ▼          ▼
ResumeAgent CoachAgent InterviewAgent Tasks API
(Groq)     (Groq+Mem) (Gemini)       (CRUD)
    │         │          │
    ▼         ▼          ▼
Skills     Tasks      Scores
saved to   assigned   saved to
Supabase   to DB      Supabase
    │         │          │
    └────┬────┴──────────┘
         ▼
    AlertAgent monitors
    all student activity
         │
         ▼
    Sends email if at-risk
    (via Resend API)
         │
         ▼
    TPC Admin sees all
    students in /tpc
```

---

## 🔐 Security

- **JWT Authentication**: All protected routes require a valid Supabase JWT
- **Role-Based Access**: TPC dashboard requires `tpc_admin` role
- **Rate Limiting**: 100 requests/min per IP
- **CORS**: Configurable allowed origins
- **Webhook Validation**: n8n webhooks require API key header

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

Built by **Yaswanth Kumar Mallela** as a placement readiness platform leveraging multi-agent AI architecture.

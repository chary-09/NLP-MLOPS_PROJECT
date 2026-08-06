# CareerCopilot AI

**AI Career Intelligence for Students**

CareerCopilot AI is a student-focused platform that brings resume analysis, ATS compatibility checks, skill gap detection, interview prep, and career guidance together in one intelligent experience.

---

## 🚀 Overview

### Purpose

CareerCopilot AI helps students prepare for placements and internships by improving resumes, matching candidate skills to job descriptions, and generating personalized career guidance.

### Problem Statement

- Students need a single platform for resume review, ATS testing, interview preparation, and career planning.
- Existing tools are often fragmented and provide generic or shallow advice.

### Our Solution

- Centralized career intelligence for resumes and job matching
- Personalized recommendations using resume, skills, and job descriptions
- Fast, actionable feedback on ATS readiness, interview readiness, and growth plans

### Target Users

- College students
- Recent graduates
- Internship seekers
- Placement cells

### Core Features

- Resume upload and parsing
- ATS scoring and job-match analysis
- Skill gap detection
- Career roadmap generation
- Interview question generation
- AI chat guidance

---

## 🏗️ Detailed Architecture

### System Architecture Diagram

```text
+-------------------+          +-------------------------+
|                   |          |                         |
|     Browser       |  HTTPS   |      React SPA          |
|                   |--------->|  (Resume UI, dashboard) |
+-------------------+          +-----------+-------------+
                                        |
                                        | REST / JSON
                                        v
                          +-------------+--------------+
                          |                            |
                          |      FastAPI Backend       |
                          |                            |
                          +------+------+--------------+
                                 |      |     |  
           +---------------------+      |     |   +--------------------+
           |                            |     |   |                    |
           v                            v     v   v                    v
+----------------+           +----------------+ +---------------+ +----------------+
|                |           |                | |               | |                |
|  PostgreSQL    |           |   AI Engine    | | Object Store  | | Metrics/Logs   |
|                |           | (NLP, scoring, | | (resumes,     | | (errors,       |
+----------------+           | embeddings,    | | artifacts)     | | usage, health) |
                              | generation)    | +---------------+ +----------------+
                              +----------------+
```

### Component Flow

```text
1. User opens the app in the browser
2. React SPA calls backend APIs
3. FastAPI authenticates and validates requests
4. Backend stores records in PostgreSQL
5. Backend sends resume/JD data to AI Engine
6. AI Engine returns scores, gaps, and recommendations
7. Backend persists results and returns them to the UI
8. UI renders dashboard, reports, and chat guidance
```

### Service Interaction Diagram

```text
+-------------+      +-----------------+      +-----------------+
|  Frontend   | ---> |   Backend API   | ---> |   AI Engine     |
| (React SPA) |      |  (FastAPI)      |      | (NLP, scoring)  |
+-------------+      +--------+--------+      +--------+--------+
                               |                        |
                               |                        |
                               v                        v
                         +-----------+            +------------+
                         | Database  |            | Object /   |
                         | PostgreSQL|            | file store |
                         +-----------+            +------------+
```

### AI Engine Pipeline Diagram

```text
+-----------------+    +-----------------+    +----------------------+    +----------------------+    +----------------------+
|                 |    |                 |    |                      |    |                      |    |                      |
|  Resume / JD    | -> |    Text         | -> |   Entity & Skill     | -> |  Embeddings & Match  | -> |Recommendations &     |
|  Input          |    |  Extraction     |    |   Extraction         |    |   Scoring            |    |   Output             |
|  (PDF/TXT/JSON) |    |  (OCR / parser) |    |  (skills, roles,     |    |  (semantic similarity|    |  (career roadmap,    |
|                 |    |                 |    |   experience, gaps)  |    |   + ATS score)       |    |   interview Qs)      |
+-----------------+    +-----------------+    +----------------------+    +----------------------+    +----------------------+
```

### File Structure Diagram

```text
NLP-PDNC project/
├── README.md
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── hooks/
│   │   ├── styles/
│   │   └── App.tsx
│   ├── package.json
│   ├── tsconfig.json
│   └── tailwind.config.js
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── utils/
│   │   └── main.py
│   ├── requirements.txt
│   ├── alembic/
│   └── Dockerfile
├── ai_engine/
│   ├── nlp/
│   ├── models/
│   ├── embeddings/
│   ├── inference.py
│   ├── train.py
│   └── requirements.txt
├── database/
│   ├── migrations/
│   ├── init.sql
│   └── seed/
├── tests/
│   ├── frontend/
│   ├── backend/
│   ├── ai_engine/
│   └── integration/
└── docker/
    ├── docker-compose.yml
    └── env.example
```

### Minimal Design Summary

- Browser → React SPA → FastAPI
- FastAPI → PostgreSQL, AI Engine, file storage, logging
- AI Engine provides resume parsing, matching, and guidance
- File structure separates frontend, backend, AI, database, tests, and deployment

### Updated M8N Design

- **Modular**: Each layer is separate and independently deployable
- **Manageable**: API contracts and versioned artifacts
- **Measurable**: Metrics and logs for latency, inference, and errors
- **Non-Functional**: Security, performance, reliability, and scalability are built into the design

This system design uses four M8N dimensions with eight focused architecture decisions that include non-functional requirements as first-class design criteria.

- **Modular**
  1. Separate frontend, backend, AI engine, and data services into independent components.
  2. Use clear service boundaries so frontend, API, and AI layers can evolve separately.

- **Manageable**
  3. Define API contracts for all backend and AI interactions.
  4. Version resumes, parsed artifacts, and model metadata to support rollback and audit.

- **Measurable**
  5. Collect metrics for API latency, AI inference time, and user engagement.
  6. Log resume processing stages, match results, recommendation generation, and errors.

- **Non-Functional**
  7. Design for security, performance, scalability, and reliability as primary architecture goals.
  8. Use deployment automation, health checks, and monitoring to make the system observable and operable.

### Non-Functional Design Considerations

- **Security**
  - HTTPS for all endpoints
  - JWT-based auth and role checks
  - Validation for uploaded documents and user input

- **Performance**
  - Cache common JD and resume embeddings
  - Batch AI inference where possible
  - Use async background tasks for long-running analysis

- **Scalability**
  - Containerize services for horizontal scaling
  - Separate CPU-bound backend from GPU-bound AI workloads
  - Use a managed database and optional vector store

- **Reliability**
  - Retry and circuit-breaker patterns for external model services
  - Health checks for backend and AI components
  - Persistent storage for analysis results

---

## 📁 Project Structure

### Expected Folder Layout

```text
frontend/
  ├─ public/
  ├─ src/
  │   ├─ components/
  │   ├─ pages/
  │   ├─ services/
  │   ├─ hooks/
  │   ├─ styles/
  │   └─ App.tsx
  ├─ package.json
  ├─ tsconfig.json
  └─ tailwind.config.js

backend/
  ├─ app/
  │   ├─ api/
  │   ├─ core/
  │   ├─ models/
  │   ├─ schemas/
  │   ├─ services/
  │   ├─ utils/
  │   └─ main.py
  ├─ requirements.txt
  ├─ alembic/
  └─ Dockerfile

ai_engine/
  ├─ nlp/
  ├─ models/
  ├─ embeddings/
  ├─ inference.py
  ├─ train.py
  └─ requirements.txt

database/
  ├─ migrations/
  ├─ init.sql
  └─ seed/

tests/
  ├─ frontend/
  ├─ backend/
  ├─ ai_engine/
  └─ integration/

docker/
  ├─ docker-compose.yml
  └─ env.example
```

### AI Architecture

- NLP: resume and job description parsing, entity extraction, semantic similarity, and skill normalization
- Deep Learning: resume quality scoring, ATS prediction, and match probability estimation
- Generative AI: interview question generation, chat guidance, and improvement suggestions

---

## 💡 Tech Stack & Roadmap

### Technologies

- Frontend: React, TypeScript, Tailwind CSS
- Backend: FastAPI, Python, SQLAlchemy, JWT
- Database: PostgreSQL
- NLP + AI: spaCy, PyTorch, Transformers, sentence embeddings
- DevOps: Docker, GitHub Actions

### What Makes It Special

- AI-powered career guidance tailored for students
- Resume intelligence and ATS readiness insights
- Personalized learning paths and interview preparation
- Scalable, modular architecture for rapid development

### Current Status

- Stage: Planning and design
- Next steps: define requirements, build prototype, implement resume parser, add ATS matching
- Future expansion: recruiter portal, mobile app, and multi-language support


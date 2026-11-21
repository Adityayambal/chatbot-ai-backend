# chatbot-ai-backend
AI doubt-solving chatbot. Answers students' questions, video transcripts and practice questions using BM25 retrieval and OpenAI GPT, with citations. FastAPI + React.

> AI doubt-solving chat for Indian K-12 students: answers com own lesson content, with citations, in simple grade-appropriate language.

### Highlights
- **Answers from course content:** retrieves excerpts from revision notes, video transcripts and practice questions with BM25 for the student's exact medium, grade, board and subject. Every answer cites its sources.
- **Two-model pipeline:** a router (`gpt-5.4-mini`) classifies each question (on-topic, off-topic, greeting, unsafe or test request) and rewrites follow-ups; the tutor (`gpt-5.5`) streams the answer.
- **Streaming chat:** answers stream over Server-Sent Events, with Stop and Retry, plus 👍/👎 feedback on each answer.
- **Doubt lifecycle:** one open chat per subject, chat history, "Did this solve your doubt?" (not asked during a test), an optional 1–5★ rating, and a "Talk to a teacher" hand-off.
- **Quiz mode:** students can ask to be tested. They get 3–5 questions, and their answers are checked and scored.
- **Cost-aware:** OpenAI prompt caching is preserved across turns. A typical answer costs about $0.04 and a greeting about $0.001, and each turn's cost is logged.
- **Tested offline:** 99 backend tests run without network access or OpenAI calls (a fixture bundle and a fake OpenAI client stand in).

### Tech stack
| Layer | Stack |
|---|---|
| Backend | Python 3.11+, FastAPI, SQLAlchemy 2 (async) + SQLite, rank-bm25, OpenAI SDK, sse-starlette |
| Frontend | React 18, Vite, TypeScript (strict), Tailwind CSS (light/dark), lucide-react |
| Tests | pytest + pytest-asyncio, respx, ruff; `tsc --noEmit` |

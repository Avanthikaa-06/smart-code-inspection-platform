# Smart Code Inspection Platform with Vulnerability Detection System

An AI-powered multi-agent platform for automated source code inspection, security vulnerability detection, severity assessment, remediation, and developer assistance.

## Overview

The Smart Code Inspection Platform combines static analysis, artificial intelligence, security knowledge retrieval, and automated remediation to provide developers with a centralized code inspection workflow.

The platform analyzes Python and Java source code, identifies code quality issues and security vulnerabilities, evaluates their severity, provides remediation guidance, and generates structured reports.

## Key Features

- Automated source code analysis for Python and Java
- Static code quality analysis
- Security vulnerability detection
- OWASP-based security analysis
- CWE classification for identified vulnerabilities
- Severity classification into Critical, High, Medium, and Low
- AI-powered remediation recommendations
- Before-and-after code comparison for applicable fixes
- RAG-based security knowledge retrieval
- Conversational code assistant for developer queries
- Multi-agent architecture for specialized analysis tasks
- Parallel agent execution using LangGraph
- PR Summary generation
- Code quality and security reporting
- PDF report generation
- Centralized LLM routing with automatic fallback
- Developer-focused analysis dashboard
- Analysis history and previous-result access
- GitHub-based authentication support

## System Architecture

The platform follows a multi-agent architecture in which different components perform specialized tasks.

### Core Components

- Code Analysis Agent
  - Analyzes source code quality and maintainability.
  - Uses static analysis tools to identify code quality issues.

- Security Agent
  - Detects security vulnerabilities in source code.
  - Integrates security analysis tools and security knowledge.

- Severity Agent
  - Evaluates the severity of detected findings.
  - Classifies issues according to their security and code-quality impact.

- Remediation Agent
  - Generates remediation recommendations.
  - Provides secure coding guidance and applicable code improvements.

- PR Summary Agent
  - Consolidates analysis results.
  - Generates a structured summary of findings and remediation information.

- Conversational Agent
  - Provides an interactive interface for developers.
  - Uses retrieval-augmented generation to answer questions related to analyzed code and security findings.

- Orchestrator
  - Coordinates the execution of multiple agents.
  - Uses LangGraph for agent workflow orchestration and parallel execution.

- RAG Engine
  - Retrieves relevant security knowledge from the project knowledge base.
  - Supports contextual responses for security-related queries.

- LLM Router
  - Provides centralized access to language models.
  - Uses a primary LLM and automatically falls back to a backup LLM when required.

## Analysis Technologies

### Python

The platform uses multiple analysis techniques for Python source code:

- Pylint
- Radon
- Python AST analysis
- Bandit
- Semgrep

### Java

Java source code analysis uses:

- PMD
- javalang AST analysis
- Semgrep

### AI and Knowledge Retrieval

- LangChain
- LangGraph
- Google Gemini
- Groq
- FAISS
- Sentence Transformers
- Retrieval-Augmented Generation

## Workflow

The general analysis workflow is:

1. User submits Python or Java source code.
2. The system detects the programming language.
3. Static analysis tools inspect the source code.
4. Code quality and security findings are collected.
5. Specialized agents process the findings.
6. The Severity Agent evaluates finding severity.
7. The RAG engine retrieves relevant security knowledge.
8. The Remediation Agent generates remediation guidance.
9. The PR Summary Agent consolidates the results.
10. The platform presents the results through the Streamlit dashboard.
11. A PDF report can be generated for the completed analysis.
12. Developers can use the Conversational Assistant to query the analysis.

## Technology Stack

| Category | Technologies |
|---|---|
| Frontend | Streamlit |
| Programming Languages | Python, Java |
| Static Analysis | Pylint, Radon, Bandit, PMD, Semgrep |
| AST Analysis | Python AST, javalang |
| AI Frameworks | LangChain, LangGraph |
| Primary LLM | Google Gemini |
| Backup LLM | Groq |
| Vector Database | FAISS |
| Embeddings | Sentence Transformers |
| Security Knowledge | OWASP, CWE, Secure Coding Practices |
| Report Generation | ReportLab |
| Visualization | Plotly |
| Authentication | GitHub OAuth |
| Deployment | Streamlit Community Cloud |

## Project Structure

```text
smart-code-inspection-platform/
│
├── app.py
├── backend.py
├── auth.py
├── analysis_history.py
├── language_detector.py
├── llm_router.py
├── findings_display.py
├── pr_summary_pdf.py
├── rag_engine.py
├── report_generator.py
├── schemas.py
│
├── codeanalysis.py
├── conversationalagent.py
├── orchestrator.py
├── prsummaryagent.py
├── remediationagent.py
├── securityagent.py
├── severityagent.py
│
├── Best_practices.txt
├── FAQ.txt
├── java_securecoding_practices.txt
├── OWASP_CODING_PRACTISES.txt
├── OWASP_Top10_2025.txt
├── python_securecoding_practices.txt
│
├── requirements.txt
├── LICENSE
└── README.md

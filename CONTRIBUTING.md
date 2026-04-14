# Contributing to Hands-On AI Engineering

First off, thank you for considering contributing to Hands-On AI Engineering! Your contributions help build a valuable resource for the AI engineering community.

This document provides guidelines for contributing to this repository. Please read it carefully to ensure a smooth contribution process.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Adding a New Project](#adding-a-new-project)
- [Project Structure Requirements](#project-structure-requirements)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold professional and respectful behavior. Please report any unacceptable behavior by opening an issue.

---

## How Can I Contribute?

There are many ways to contribute:

### Reporting Bugs

If you find a bug in any project:
1. **Check existing issues** to avoid duplicates
2. **Open a new issue** with:
   - Clear, descriptive title
   - Detailed description with steps to reproduce
   - Project name/directory
   - Error messages or logs
   - Your environment details (OS, Python version, etc.)

### Suggesting Enhancements

Have an idea for improvement?
1. **Open an issue** to discuss it first
2. Provide:
   - Clear description of the enhancement
   - Use case and benefits
   - Possible implementation approach

### Adding New Projects

This is the primary contribution type! See [Adding a New Project](#adding-a-new-project) below.

---

## Adding a New Project

We welcome new AI engineering projects! Follow these steps:

### Step 1: Create an Issue First

**Before starting any work**, create an issue describing your project:
- What the project does
- Which category it belongs to (or propose a new one)
- Technologies used
- Unique value it provides

This helps us:
- Avoid duplicate work
- Provide guidance on structure
- Track contributions

### Step 2: Choose the Right Category

**Current Categories:**

- [`rag_apps/`](rag_apps/)
- [`ai_agents/`](ai_agents/)
- [`fine_tuning/`](fine_tuning/)
- [`openclaw/`](openclaw/)

**Adding a New Category:**

Categories are added **progressively** as projects are contributed. 

**Potential Future Categories:**
- `mcp_agents/` - Model Context Protocol integrations
- `fine_tuning/` - Model fine-tuning examples
- `voice_agents/` - Voice-enabled AI applications
- `llm_apps/` - General LLM-powered applications
- `workflow_automation/` - AI workflow tools
- `computer_vision/` - CV and image processing

**Add your project in the right category folder:** Once you've chosen or proposed a category, create your project folder inside the appropriate category directory (e.g., `rag_apps/your_project_name/`).

### Step 3: Follow Naming Conventions

**Project Folder Naming:**
- Use **snake_case** (all lowercase with underscores)
- Be descriptive but concise
- Examples:
  - ✅ `agentic_rag_with_qwen_and_firecrawl`
  - ✅ `multi_agent_financial_analyst`
  - ✅ `voice_enabled_customer_support`
  - ❌ `my-project` (don't use kebab-case)
  - ❌ `Project1` (don't use numbers without context)
  - ❌ `agent` (too generic)

### Step 4: One Project Per Pull Request

**IMPORTANT**: Submit each new project in its own Pull Request.

✅ **Correct**:
- PR #1: Add agentic_rag_with_qwen
- PR #2: Add voice_customer_support

❌ **Incorrect**:
- PR #1: Add agentic_rag_with_qwen + voice_customer_support + finagent

This keeps reviews focused and makes it easier to merge/rollback if needed.

---

## Project Structure Requirements

Every project **MUST** follow this structure:

```
your_project_name/
├── README.md              # Required: Project documentation
├── requirements.txt       # Required: Python dependencies
├── .env.example          # Required: Example environment variables
├── app.py                # Main application file (name may vary)
├── utils.py              # Optional: Helper functions
├── tests/                # Encouraged: Test files
└── assets/               # Optional: Images, docs, etc.
```

### README.md Requirements

Your project **MUST** have a comprehensive README. Use our [template](.github/README_TEMPLATE.md) and include:

---


## Pull Request Process

### 1. Fork and Branch

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering

# Create a branch for your project
git checkout -b add-project-name
```

### 2. Make Your Changes

- Add your project in the appropriate category folder
- Follow all structure requirements
- Test thoroughly

### 3. Commit Guidelines

```bash
# Use clear, descriptive commit messages
git add .
git commit -m "Add: Agentic RAG with Qwen and FireCrawl project"

# Push to your fork
git push origin add-project-name
```

**Commit Message Format:**
- `Add: <project-name>` - New project
- `Fix: <description>` - Bug fix
- `Update: <description>` - Documentation or code update
- `Refactor: <description>` - Code restructuring

### 4. Create Pull Request

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill in the PR template:

```markdown
## Description
Brief description of your project

## Related Issue
Closes #<issue-number>

## Type of Change
- [x] New project
- [ ] Bug fix
- [ ] Documentation update

## Checklist
- [x] Project is in the correct category folder
- [x] README.md follows the template
- [x] requirements.txt is included
- [x] .env.example is included
- [x] No API keys or secrets committed
- [x] Project has been tested locally

## Category
`rag_apps/` or `ai_agents/` or propose new category

## Screenshots (if applicable)
Add screenshots of your project running
```

### 5. Review Process

- Maintainers will review your PR
- Address any feedback or requested changes
- Once approved, your PR will be merged!

---

## Project Approval Criteria

Your project will be approved if it:

✅ **Meets Requirements:**
- Complete README with all sections
- Proper folder structure
- Working code with no errors
- Dependencies documented
- Environment variables documented

✅ **Provides Value:**
- Demonstrates a real use case
- Well-documented and easy to understand
- Production-ready or educational value
- Unique or improves upon existing approaches

✅ **Follows Guidelines:**
- In correct category
- Uses snake_case naming
- One project per PR
- Linked to an issue

---

## Getting Help

**Questions?** Open an issue with the `question` label

**Need guidance?** Tag maintainers in your issue

**Found a bug?** Open an issue with detailed reproduction steps

---

Thank you for contributing to Hands-On AI Engineering!

Your work helps developers worldwide learn and build amazing AI applications.

---

**Next Steps:**
1. Star the repository
2. Create an issue for your project idea
3. Start building!
4. Submit your PR

We can't wait to see what you build!



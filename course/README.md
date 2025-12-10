# Course Learning Trail

This directory contains all the learning materials, notes, and experiments from building the battery heating controller system.

## Structure

### `/notes/` - Theory & Concepts
Reference documents explaining the theory behind each step:
- Clear explanations of concepts
- Why we chose certain approaches
- Trade-offs and alternatives
- Links to further reading

### `/notebooks/` - Exploratory Analysis
Jupyter notebooks for data exploration and experimentation:
- Data visualization and analysis
- Model interpretation
- Performance comparisons
- Experimental features

## Learning Path

Follow the main [syllabus.md](./syllabus.md) and reference the corresponding notes as you progress:

| Step | Topic | Reference Notes | Notebooks |
|------|-------|----------------|-----------|
| 1.1 | ML Workflow & Agent Basics | `notes/01_ml_workflow.md` | - |
| 1.2 | Data & Features | `notes/02_data_and_features.md` | - |
| 2.1-2.2 | Data Collection & EDA | `notes/03_data_collection.md` | `01_explore.ipynb` |
| 3.1-3.2 | Supervised Learning | `notes/04_supervised_learning.md` | - |
| 3.3 | Model Interpretability | `notes/05_interpretability.md` | `02_interpret_model.ipynb` |
| 4.1-4.2 | Control Systems & Energy | `notes/06_control_systems.md` | - |
| 4.3 | Tool-Based Architecture | `notes/07_tool_pattern.md` | - |
| 5.1-5.2 | ML-Assisted Control | `notes/08_ml_control.md` | - |
| 5.3 | Model Monitoring | `notes/09_monitoring_drift.md` | `03_model_performance.ipynb` |
| 6.1-6.2 | Agentic AI & Graphs | `notes/10_agents_and_graphs.md` | - |
| 6.3 | Distributed Systems | `notes/11_edge_cloud.md` | - |
| 7.1 | Evaluation & Testing | `notes/12_evaluation.md` | `04_controller_comparison.ipynb` |
| 7.3 | Reinforcement Learning | `notes/13_reinforcement_learning.md` | - |

## Notes Format

Each note document follows this structure:

1. **Core Concepts** - What you need to know
2. **Why It Matters** - Relevance to this project
3. **How It Works** - Technical explanation
4. **Trade-offs** - Alternative approaches and their pros/cons
5. **Key Takeaways** - Summary bullets
6. **Further Reading** - Optional deep dives

## Learning Workflow

### How to Progress Through the Course

For each step in the [syllabus](./syllabus.md):

1. **Theory First**
   - Announce: "I'm ready for Step X.X"
   - I'll create a theory note in `notes/` (e.g., `04_supervised_learning.md`)
   - We'll discuss the concepts - ask questions, clarify understanding
   - The note serves as your permanent reference

2. **Build Together**
   - Apply the theory to the project task
   - Write production code in `src/`
   - Add comments explaining key decisions
   - Reference theory notes in code when helpful

3. **Document Insights**
   - If we discover something important during implementation, we update the theory note
   - For exploratory work, we create notebooks in `notebooks/`
   - Notebooks include inline commentary on observations and insights

4. **Iterate**
   - Each step builds on the previous one
   - The system evolves continuously
   - Your understanding deepens with each iteration

### What Gets Created

**Theory Notes** (`notes/*.md`):
- Standalone reference documents
- Explain concepts, trade-offs, and rationale
- Can be read independently or reviewed later
- Follow consistent structure (see Notes Format above)

**Notebooks** (`notebooks/*.ipynb`):
- Data exploration and visualization
- Model analysis and interpretation
- Performance comparisons
- Experimental features

**Implementation Notes**:
- Inline code comments in `src/`
- Architecture decisions documented
- References back to theory when relevant

### The Learning Trail

As you progress, you'll build a complete record:
- **What** you built (code in `src/`)
- **Why** you built it that way (notes in `course/notes/`)
- **How** it performs (notebooks in `course/notebooks/`)
- **What** you learned (insights documented throughout)

This trail lets you:
- Review concepts anytime
- Understand past decisions
- Share your learning journey
- Build similar systems in the future

---

## Deleting This Directory

When you're done learning and ready to deploy, you can safely delete the entire `course/` directory. The production system in `src/` will be completely independent.

---

**Start here**: [syllabus.md](./syllabus.md)

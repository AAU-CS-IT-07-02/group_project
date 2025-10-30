# G 5 Sub-Issue Templates

This directory contains templates for creating sub-issues under Goal G 5 (Getting started with do_mpc).

## Overview

Four new sub-issues have been defined to support the development and experimentation work for Goal G 5:

1. **G 5.5**: Optimise the runtime of PySINDy
2. **G 5.6**: Identify all hyperparameters, their domain and which values we want to test  
3. **G 5.7**: Find a way to monitor the resources of a running SLURM experiment
4. **G 5.8**: Run experiments on the combinatorial explosion of parameters

## File Structure

```
g5_sub_issues/
├── README.md                               # This file
├── G5.5_optimize_pysindy_runtime.md       # Template for runtime optimization issue
├── G5.6_hyperparameter_identification.md  # Template for hyperparameter identification issue
├── G5.7_monitor_slurm_resources.md        # Template for SLURM monitoring issue
└── G5.8_combinatorial_experiments.md      # Template for combinatorial experiments issue
```

## How to Create Issues from These Templates

### Method 1: Manual Creation via GitHub UI

1. Navigate to the [repository issues page](https://github.com/AAU-CS-IT-07-02/group_project/issues)
2. Click "New Issue"
3. Open one of the template files in this directory
4. Copy the content (excluding the YAML frontmatter if present)
5. Paste into the issue description
6. Set the title from the template
7. Add labels: Check the frontmatter for recommended labels
8. Assign: `toita86` and `beltranaceves` (as specified in requirements)
9. Set milestone: M1_Modeling
10. In the description, link to parent issue #47 by mentioning "Parent issue: #47"
11. Click "Submit new issue"

### Method 2: Using GitHub CLI (gh)

If you have the GitHub CLI installed and authenticated:

```bash
# Navigate to the repository directory
cd /path/to/group_project

# Create G 5.5
gh issue create \
  --title "G 5.5 | Optimise the runtime of PySINDy" \
  --body-file docs/issue_templates/g5_sub_issues/G5.5_optimize_pysindy_runtime.md \
  --label "enhancement" \
  --assignee "toita86,beltranaceves" \
  --milestone "M1_Modeling"

# Create G 5.6
gh issue create \
  --title "G 5.6 | Identify all hyperparameters, their domain and which values we want to test" \
  --body-file docs/issue_templates/g5_sub_issues/G5.6_hyperparameter_identification.md \
  --label "documentation,enhancement" \
  --assignee "toita86,beltranaceves" \
  --milestone "M1_Modeling"

# Create G 5.7
gh issue create \
  --title "G 5.7 | Find a way to monitor the resources of a running SLURM experiment" \
  --body-file docs/issue_templates/g5_sub_issues/G5.7_monitor_slurm_resources.md \
  --label "enhancement,documentation" \
  --assignee "toita86,beltranaceves" \
  --milestone "M1_Modeling"

# Create G 5.8
gh issue create \
  --title "G 5.8 | Run experiments on the combinatorial explosion of parameters" \
  --body-file docs/issue_templates/g5_sub_issues/G5.8_combinatorial_experiments.md \
  --label "enhancement" \
  --assignee "toita86,beltranaceves" \
  --milestone "M1_Modeling"
```

### Method 3: Automated Script

A Python script can be created to automate issue creation using the GitHub API:

```python
import os
from github import Github

# Initialize GitHub client
token = os.environ.get('GITHUB_TOKEN')
g = Github(token)
repo = g.get_repo("AAU-CS-IT-07-02/group_project")

# Issue data
issues = [
    {
        "title": "G 5.5 | Optimise the runtime of PySINDy",
        "body_file": "G5.5_optimize_pysindy_runtime.md",
        "labels": ["enhancement"],
    },
    {
        "title": "G 5.6 | Identify all hyperparameters, their domain and which values we want to test",
        "body_file": "G5.6_hyperparameter_identification.md",
        "labels": ["documentation", "enhancement"],
    },
    {
        "title": "G 5.7 | Find a way to monitor the resources of a running SLURM experiment",
        "body_file": "G5.7_monitor_slurm_resources.md",
        "labels": ["enhancement", "documentation"],
    },
    {
        "title": "G 5.8 | Run experiments on the combinatorial explosion of parameters",
        "body_file": "G5.8_combinatorial_experiments.md",
        "labels": ["enhancement"],
    },
]

# Create issues
for issue_data in issues:
    with open(f"docs/issue_templates/g5_sub_issues/{issue_data['body_file']}", 'r') as f:
        body = f.read()
    
    issue = repo.create_issue(
        title=issue_data["title"],
        body=body,
        labels=issue_data["labels"],
        assignees=["toita86", "beltranaceves"],
        milestone=repo.get_milestone(1)  # M1_Modeling milestone number
    )
    print(f"Created issue #{issue.number}: {issue.title}")
```

## Issue Dependencies

The issues have the following dependency structure:

```
G 5 (Parent: Getting started with do_mpc)
├── G 5.1: Install do_mpc and get it running (existing)
├── G 5.2: Get multiple examples working (existing)
├── G 5.3: Implement a do_mpc controller for a PySINDy model (existing)
├── G 5.4: Learning materials and report explanations (existing)
├── G 5.5: Optimise the runtime of PySINDy (NEW)
├── G 5.6: Identify hyperparameters and test values (NEW)
├── G 5.7: Monitor SLURM experiment resources (NEW)
└── G 5.8: Run combinatorial parameter experiments (NEW)
     └── Depends on: G 5.5, G 5.6, G 5.7
```

## Recommended Creation Order

To respect dependencies, create issues in this order:

1. G 5.5 (Runtime optimization - no dependencies)
2. G 5.6 (Hyperparameter identification - no dependencies)
3. G 5.7 (SLURM monitoring - no dependencies)
4. G 5.8 (Combinatorial experiments - depends on 5.5, 5.6, 5.7)

After creating each issue:
1. Note the issue number
2. Link it to parent issue #47
3. For G 5.8, add links to G 5.5, G 5.6, and G 5.7 in the description

## Assignees

All four issues should be assigned to:
- **toita86**
- **beltranaceves**

## Milestone

All four issues should be added to the **M1_Modeling** milestone.

## Additional Notes

- These issues are focused on experimental work with PySINDy optimization
- They complement the existing G 5 sub-issues which focus on do_mpc
- The combination of both sets of issues supports the integration of PySINDy models with do_mpc controllers
- Resource monitoring (G 5.7) is crucial for efficient use of the SLURM cluster
- Hyperparameter optimization (G 5.8) will be a large effort and should be started only after the prerequisite issues are completed

## Contact

For questions about these issue templates, contact the assigned team members:
- toita86
- beltranaceves

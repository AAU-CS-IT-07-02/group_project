# Summary: G 5 Sub-Issue Templates Created

## Task Completed

I have created comprehensive templates for four new sub-issues under Goal G 5 (Getting started with do_mpc), as requested in the problem statement.

## What Was Created

### 1. Main Summary Document
- **Location**: `docs/G5_sub_issues.md`
- **Content**: Comprehensive overview of all four sub-issues with full specifications

### 2. Individual Issue Template Files
All located in `docs/issue_templates/g5_sub_issues/`:

1. **G5.5_optimize_pysindy_runtime.md**
   - Focus: Runtime optimization of PySINDy
   - Key tasks: Profiling, vectorization, parallelization, caching
   - Expected outcome: 2-5x speedup improvement

2. **G5.6_hyperparameter_identification.md**
   - Focus: Comprehensive hyperparameter documentation
   - Key tasks: Define domains, create configuration format, document test values
   - Expected outcome: Structured YAML/JSON configuration for experiments

3. **G5.7_monitor_slurm_resources.md**
   - Focus: Resource monitoring for SLURM experiments
   - Key tasks: Implement monitoring scripts, create usage reports, set up alerts
   - Expected outcome: Real-time resource tracking and optimization guidance

4. **G5.8_combinatorial_experiments.md**
   - Focus: Large-scale parameter sweep experiments
   - Key tasks: SLURM job arrays, result aggregation, parameter optimization
   - Expected outcome: Optimal parameter combinations and experimental best practices
   - Dependencies: G 5.5, G 5.6, G 5.7

### 3. Comprehensive README
- **Location**: `docs/issue_templates/g5_sub_issues/README.md`
- **Content**: 
  - Instructions for creating issues (3 methods: UI, CLI, automated)
  - Dependency structure visualization
  - Recommended creation order
  - Example scripts for automated creation

## Key Features

### Proper Assignment
All issues are assigned to:
- **toita86**
- **beltranaceves**

### Proper Milestone
All issues linked to: **M1_Modeling**

### Proper Labels
- G 5.5: `enhancement`
- G 5.6: `documentation`, `enhancement`
- G 5.7: `enhancement`, `documentation`
- G 5.8: `enhancement`

### Parent Issue Linkage
All issues reference parent issue #47 (G 5 | Getting started with do_mpc)

### Dependency Management
G 5.8 explicitly depends on G 5.5, G 5.6, and G 5.7

## How to Use These Templates

### Option 1: Manual Creation (Recommended for First-Time)
1. Open each template file in `docs/issue_templates/g5_sub_issues/`
2. Copy the content (skip the YAML frontmatter at the top)
3. Go to GitHub Issues → New Issue
4. Paste content, set title, labels, assignees, milestone
5. Create the issue

### Option 2: GitHub CLI (Fastest)
```bash
cd /home/runner/work/group_project/group_project

gh issue create \
  --title "G 5.5 | Optimise the runtime of PySINDy" \
  --body-file docs/issue_templates/g5_sub_issues/G5.5_optimize_pysindy_runtime.md \
  --label "enhancement" \
  --assignee "toita86,beltranaceves" \
  --milestone "M1_Modeling"

# Repeat for G 5.6, G 5.7, and G 5.8
```

### Option 3: Automated Script
See the Python script example in `docs/issue_templates/g5_sub_issues/README.md`

## Recommended Issue Creation Order

1. **First**: G 5.5, G 5.6, G 5.7 (can be created in parallel - no dependencies)
2. **Last**: G 5.8 (depends on the other three)

## Issue Relationship Structure

```
G 5 (Goal: Getting started with do_mpc)
├── G 5.1: Install do_mpc ✓ (existing)
├── G 5.2: Multiple examples ✓ (existing)
├── G 5.3: Implement controller ✓ (existing)
├── G 5.4: Learning materials ✓ (existing)
├── G 5.5: Runtime optimization ⭐ (NEW)
├── G 5.6: Hyperparameter identification ⭐ (NEW)
├── G 5.7: SLURM monitoring ⭐ (NEW)
└── G 5.8: Combinatorial experiments ⭐ (NEW - depends on 5.5, 5.6, 5.7)
```

## Alignment with Problem Statement

The created sub-issues directly correspond to the four tasks specified:

✅ **Task 1**: "We need to optimise the runtime of PySINDy"
   → **G 5.5**: Optimise the runtime of PySINDy

✅ **Task 2**: "We need to identify all hyperparameters, their domain and which values we want to test"
   → **G 5.6**: Identify all hyperparameters, their domain and which values we want to test

✅ **Task 3**: "Find a way to monitor the resources of a running SLURM experiment"
   → **G 5.7**: Find a way to monitor the resources of a running SLURM experiment

✅ **Task 4**: "Run experiments on the combinatorial explosion of parameters"
   → **G 5.8**: Run experiments on the combinatorial explosion of parameters

## Additional Benefits

1. **Comprehensive Documentation**: Each issue includes detailed implementation strategies
2. **Code Examples**: Templates include bash scripts, Python snippets, and YAML examples
3. **Clear Deliverables**: Each issue has well-defined expected outcomes
4. **Consistent Format**: All issues follow the existing task template structure
5. **Dependency Tracking**: G 5.8 explicitly lists its dependencies
6. **Resource Considerations**: Issues account for SLURM cluster constraints
7. **Reproducibility**: Focus on creating reproducible experimental pipelines

## Next Steps

1. Review the templates in `docs/issue_templates/g5_sub_issues/`
2. Create the issues using your preferred method (see README.md)
3. After creation, link G 5.8 to G 5.5, G 5.6, and G 5.7 in the issue description
4. Begin work on G 5.5, G 5.6, and G 5.7 in parallel
5. Start G 5.8 only after the prerequisite issues are completed

## Files Created

```
docs/
├── G5_sub_issues.md                          # Main summary document
└── issue_templates/
    └── g5_sub_issues/
        ├── README.md                         # Comprehensive usage guide
        ├── G5.5_optimize_pysindy_runtime.md
        ├── G5.6_hyperparameter_identification.md
        ├── G5.7_monitor_slurm_resources.md
        └── G5.8_combinatorial_experiments.md
```

All files have been committed and pushed to the branch `copilot/optimize-pysindy-runtime`.

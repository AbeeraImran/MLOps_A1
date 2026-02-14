# MLOps Automation: Titanic Dataset

[cite_start]This project implements an end-to-end machine learning pipeline[cite: 6]. 
[cite_start]All tasks are executed exclusively through a Makefile to ensure reproducibility[cite: 7, 8].

## Workflow
1. [cite_start]**Setup**: Installs dependencies[cite: 19].
2. [cite_start]**Data**: Downloads Titanic dataset to `data/raw`[cite: 20, 28].
3. [cite_start]**Preprocess**: Cleans data and saves to `data/processed`[cite: 21, 31].
4. [cite_start]**Features**: Engineering new features in `features/`[cite: 33, 34].
5. [cite_start]**Train**: Trains a Random Forest model[cite: 36, 37].
6. [cite_start]**Evaluate**: Saves metrics in `results/`[cite: 40].

## Commands
Run the entire pipeline:
```bash
make all
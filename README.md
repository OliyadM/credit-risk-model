Credit Risk Probability Model for Alternative Data

Project Overview
You are building a credit scoring model for Bati Bank using alternative behavioral data from an eCommerce platform. The goal is to develop a proxy target variable for credit risk, perform exploratory data analysis (EDA), engineer features, build and deploy machine learning models that assign risk probability and credit scores, and automate model deployment with CI/CD.

This project follows best practices in machine learning engineering including a well-structured codebase, documentation, reproducibility, and testing.

Credit Scoring Business Understanding
1. Influence of Basel II Accord on Model Requirements
The Basel II Capital Accord requires financial institutions to accurately measure and manage credit risk to maintain adequate capital reserves. This regulation emphasizes the need for credit risk models that are interpretable, transparent, and well-documented to ensure regulatory compliance, risk control, and stakeholder trust. Therefore, our credit scoring model must be explainable and auditable, facilitating internal validation and regulatory review.

2. Necessity and Risks of Creating a Proxy Variable
Since the dataset lacks a direct "default" label indicating loan repayment failure, it is necessary to create a proxy variable that approximates credit risk by analyzing customer behavior patterns such as Recency, Frequency, and Monetary value (RFM). This proxy enables supervised learning for risk prediction. However, using a proxy can introduce business risks such as misclassification, where customers are incorrectly labeled high or low risk, potentially leading to financial losses or missed lending opportunities.

3. Trade-Offs Between Simple and Complex Models
Simple models like Logistic Regression with Weight of Evidence (WoE) encoding are highly interpretable, easier to validate, and meet regulatory requirements more straightforwardly. They allow clear explanations of risk factors but may sacrifice some predictive performance.

Complex models like Gradient Boosting can capture nonlinear relationships and interactions, often resulting in higher accuracy. However, they are less interpretable ("black-box") and may face challenges in regulatory approval and stakeholder acceptance due to limited transparency.

Balancing interpretability and predictive performance is critical in regulated financial contexts.


project structure

credit-risk-model/
├── .github/workflows/ci.yml       # CI/CD pipeline for testing and linting
├── data/                          # Data directory (raw and processed, excluded from Git)
│   ├── raw/
│   └── processed/
├── notebooks/                     # Jupyter notebooks for EDA and analysis
│   └── 1.0-eda.ipynb
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_processing.py         # Feature engineering and data prep
│   ├── train.py                   # Model training script
│   ├── predict.py                 # Inference script
│   └── api/
│       ├── main.py                # FastAPI backend for model serving
│       └── pydantic_models.py     # Pydantic schemas for API validation
├── tests/                        # Unit tests
│   └── test_data_processing.py
├── Dockerfile                    # Containerization setup
├── docker-compose.yml            # Compose file for local deployment
├── requirements.txt              # Python dependencies
├── .gitignore                    # Files/folders to ignore in Git
└── README.md                     # Project documentation (this file)

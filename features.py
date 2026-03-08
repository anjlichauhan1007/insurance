# Features list for the loan approval model
# Based on the dataset structure

FEATURES = [
    'person_age',                      # float64
    'person_gender',                   # str (encoded to int)
    'person_education',                # str (encoded to int)
    'person_income',                   # float64
    'person_emp_exp',                  # int64
    'person_home_ownership',           # str (encoded to int)
    'loan_amnt',                       # float64
    'loan_intent',                     # str (encoded to int)
    'loan_int_rate',                   # float64
    'loan_percent_income',             # float64
    'cb_person_cred_hist_length',      # float64
    'credit_score',                    # int64
    'previous_loan_defaults_on_file'   # str (encoded to int)
]

# Encoding dictionaries
ENCODINGS = {
    'person_gender': {'female': 1, 'male': 0},
    'person_education': {
        'Master': 1, 'High School': 2, 'Bachelor': 3,
        'Associate': 4, 'Doctorate': 5
    },
    'person_home_ownership': {
        'RENT': 1, 'OWN': 2, 'MORTGAGE': 3, 'OTHER': 4
    },
    'loan_intent': {
        'PERSONAL': 1, 'EDUCATION': 2, 'MEDICAL': 3,
        'VENTURE': 4, 'HOMEIMPROVEMENT': 5, 'DEBTCONSOLIDATION': 6
    },
    'previous_loan_defaults_on_file': {'No': 1, 'Yes': 2}
}

# Data types
DTYPES = {
    'person_age': 'float64',
    'person_gender': 'str',
    'person_education': 'str',
    'person_income': 'float64',
    'person_emp_exp': 'int64',
    'person_home_ownership': 'str',
    'loan_amnt': 'float64',
    'loan_intent': 'str',
    'loan_int_rate': 'float64',
    'loan_percent_income': 'float64',
    'cb_person_cred_hist_length': 'float64',
    'credit_score': 'int64',
    'previous_loan_defaults_on_file': 'str',
    'loan_status': 'int64'
}
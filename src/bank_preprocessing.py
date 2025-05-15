import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder 
from sklearn.model_selection import train_test_split  
from typing import List, Dict, Optional, Tuple 

def map_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    if df['y'].dtype == 'object':
        df['y'] = df['y'].map({'yes': 1, 'no': 0})
    return df


def handle_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace 'unknown' values with the mode in selected columns:
    - 'job', 'marital', 'education': replace 'unknown' with mode
    - 'default', 'housing', 'loan': keep 'unknown' as a separate category
    """
    replace_with_mode = ['job', 'marital', 'education']
    
    for col in replace_with_mode:
        if col in df.columns and 'unknown' in df[col].values:
            mode = df.loc[df[col] != 'unknown', col].mode(dropna=True)
            if not mode.empty:
                df[col] = df[col].replace('unknown', mode[0])
    
    return df

def simplify_education(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge similar education categories into broader groups and apply ordinal encoding.
    """
    # Merge basic education levels into a single category
    df['education'] = df['education'].replace({
        'basic.4y': 'basic',
        'basic.6y': 'basic',
        'basic.9y': 'basic'
    })
    
    # Define ordinal encoding for education levels
    edu_mapping = {
        "illiterate": 0,
        "basic": 1,
        "high.school": 2,
        "professional.course": 3,
        "university.degree": 4    
    }
    
    df['education'] = df['education'].map(edu_mapping)
    
    return df


def clip_outliers(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Clip outliers above 99th percentile for selected columns.
    """
    for col in cols:
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=upper)
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate features from existing columns.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with additional engineered features and macroeconomic ratios.
    """
    # Binary flag if client was contacted before
    df['contacted_before'] = (df['pdays'] != 999).astype(int)
    
    # Define "many_contacts" based on the upper quartile of campaign calls
    campaign_threshold = df['campaign'].quantile(0.75)
    df['many_contacts'] = (df['campaign'] > campaign_threshold).astype(int)

    # Contact ratio: campaign contacts relative to the total contacts (without creating a total_contacts column)
    df['contact_ratio'] = df['campaign'] / (df['campaign'] + df['previous']).replace(0, np.nan)

    # Flag if previous campaign resulted in success
    df['previous_outcome_success'] = (df['poutcome'] == 'success').astype(int)
    df = df.drop(columns='poutcome')

    # Creating macroeconomic interaction features
    eps = 1e-5  # Small epsilon to avoid division by zero
    df['euribor3m_to_emp.var.rate'] = df['euribor3m'] / (df['emp.var.rate'] + eps)
    df['cons.conf.idx_to_euribor3m'] = df['cons.conf.idx'] / (df['euribor3m'] + eps)

    return df

   
def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Splits the dataset into training and validation sets.

    Args:
        df (pd.DataFrame): The full dataset.
        target_col (str): The name of the target variable.
        test_size (float): Proportion of data to use for validation.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple: Training features, validation features, training targets, validation targets, and list of input column names.
    """
    # Drop 'duration' column if it exists (optional or leakage-related)
    df = df.drop(columns=['duration'], errors='ignore')
    
    # Stratified split to maintain target distribution
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target_col]
    )
    
    # Identify feature columns (excluding the target)
    input_cols = [col for col in df.columns if col != target_col]
    
    return train_df[input_cols], val_df[input_cols], train_df[target_col], val_df[target_col], input_cols

def encode_categorical_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    Encodes categorical features using OneHotEncoder, excluding the 'education' column.

    Args:
        train_inputs (pd.DataFrame): Training dataset.
        val_inputs (pd.DataFrame): Validation dataset.

    Returns:
        Tuple: Transformed training and validation datasets along with the fitted encoder.
    """
    # Identify categorical columns, excluding 'education'
    categorical_cols = train_inputs.select_dtypes(include=['object']).columns.tolist()
    if 'education' in categorical_cols:
        categorical_cols.remove('education')

    # Initialize and fit the encoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_inputs[categorical_cols])

    # Transform training and validation data
    train_encoded = pd.DataFrame(
        encoder.transform(train_inputs[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=train_inputs.index
    )
    val_encoded = pd.DataFrame(
        encoder.transform(val_inputs[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=val_inputs.index
    )

    # Replace original categorical columns with encoded versions
    train_inputs = pd.concat([train_inputs.drop(columns=categorical_cols), train_encoded], axis=1)
    val_inputs = pd.concat([val_inputs.drop(columns=categorical_cols), val_encoded], axis=1)

    return train_inputs, val_inputs, encoder


def scale_numeric_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    scale: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[MinMaxScaler]]:
    """
    Scales numeric features using MinMaxScaler, if specified.

    Args:
        train_inputs (pd.DataFrame): Training feature set.
        val_inputs (pd.DataFrame): Validation feature set.
        scale (bool): Whether to apply scaling.

    Returns:
        Tuple: Scaled training and validation features, and the scaler (or None if scaling not applied).
    """
    numeric_cols = train_inputs.select_dtypes(include=[np.number]).columns.tolist()
    scaler = None
    if scale:
        scaler = MinMaxScaler()
        train_inputs[numeric_cols] = scaler.fit_transform(train_inputs[numeric_cols])
        val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    return train_inputs, val_inputs, scaler

def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    scale_numeric: bool = True,
    create_features_flag: bool = True  # to control feature creation
) -> Dict[str, any]:
    """
    Full data preprocessing: cleaning, feature creation, scaling, and encoding.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Name of the target variable.
        scale_numeric (bool): Whether to apply feature scaling.
        create_features_flag (bool): Whether to apply feature creation.

    Returns:
        Dict[str, any]: Dictionary containing prepared datasets and transformers.
    """
    df = map_target_variable(df)
    df = handle_unknowns(df)
    df = simplify_education(df)
        
    # Conditionally call create_features
    if create_features_flag:
        df = create_features(df)
    
    df = clip_outliers(df, ['campaign'])

    X_train, X_val, y_train, y_val, input_cols = split_data(df, target_col)
    X_train, X_val, scaler = scale_numeric_features(X_train, X_val, scale_numeric)
    X_train, X_val, encoder = encode_categorical_features(X_train, X_val)

    return {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'scaler': scaler,
        'encoder': encoder,
        'input_cols': input_cols
    }

def preprocess_new_data(
    df: pd.DataFrame,
    scaler: Optional[MinMaxScaler] = None,
    encoder: Optional[OneHotEncoder] = None,
    create_features_flag: bool = True
) -> pd.DataFrame:
    """
    Preprocess new, unseen data using previously fitted transformers for scaling and encoding.
    Does not create new features or handle target variables, only applies transformations.

    Args:
        df (pd.DataFrame): The new data to preprocess.
        scaler (Optional[MinMaxScaler], optional): Fitted scaler to apply to the new data.
        encoder (Optional[OneHotEncoder], optional): Fitted encoder to apply to the new data.
        create_features_flag (bool): Flag to indicate whether to create features or not.

    Returns:
        pd.DataFrame: The preprocessed new data.
    """
    # Handle unknowns and simplify education
    df = handle_unknowns(df)
    df = simplify_education(df)
    
    # Apply feature creation if needed
    if create_features_flag:
        df = create_features(df)
        
    
    # Scale numeric features using previously fitted scaler
    if scaler:
        df_scaled = scaler.transform(df.select_dtypes(include=[np.number]))
        df[df.select_dtypes(include=[np.number]).columns] = df_scaled
    
    # Encode categorical features using previously fitted encoder
    if encoder:
        categorical_cols = df.select_dtypes(include=['object']).drop(columns=['education']).columns.tolist()
        df_encoded = encoder.transform(df[categorical_cols])
        encoded_columns = encoder.get_feature_names_out(categorical_cols)
        df_encoded = pd.DataFrame(df_encoded, columns=encoded_columns, index=df.index)
        df = pd.concat([df, df_encoded], axis=1).drop(columns=categorical_cols)
    
    return df



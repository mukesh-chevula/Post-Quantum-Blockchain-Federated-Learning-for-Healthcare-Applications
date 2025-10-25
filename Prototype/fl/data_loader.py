"""
Healthcare Data Loader for Federated Learning
Loads and preprocesses Synthea patient and condition data for FL training
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

logger = logging.getLogger(__name__)


class HealthcareDataLoader:
    """
    Load and preprocess Synthea healthcare data for federated learning
    Task: Predict chronic conditions based on patient demographics
    """
    
    def __init__(self, data_dir: str, target_conditions: Optional[List[str]] = None):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing patients.csv and conditions.csv
            target_conditions: List of condition codes to predict (default: common chronic conditions)
        """
        self.data_dir = Path(data_dir)
        self.target_conditions = target_conditions or [
            'Diabetes mellitus type 2 (disorder)',
            'Essential hypertension (disorder)',
            'Chronic sinusitis (disorder)',
            'Prediabetes (finding)',
            'Body mass index 30+ - obesity (finding)',
            'Anemia (disorder)'
        ]
        
        self.patients_df = None
        self.conditions_df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load patients and conditions CSV files"""
        logger.info("Loading healthcare data...")
        
        # Load patients
        patients_path = self.data_dir / 'patients.csv'
        if not patients_path.exists():
            raise FileNotFoundError(f"Patients file not found: {patients_path}")
        
        self.patients_df = pd.read_csv(patients_path)
        logger.info(f"Loaded {len(self.patients_df)} patients")
        
        # Load conditions
        conditions_path = self.data_dir / 'conditions.csv'
        if not conditions_path.exists():
            raise FileNotFoundError(f"Conditions file not found: {conditions_path}")
        
        self.conditions_df = pd.read_csv(conditions_path)
        logger.info(f"Loaded {len(self.conditions_df)} condition records")
        
        return self.patients_df, self.conditions_df
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess patient data and create features
        
        Returns:
            DataFrame with patient features and target labels
        """
        if self.patients_df is None or self.conditions_df is None:
            self.load_data()
        
        logger.info("Preprocessing patient data...")
        
        # Calculate patient age
        current_year = datetime.now().year
        self.patients_df['BIRTHDATE'] = pd.to_datetime(self.patients_df['BIRTHDATE'])
        self.patients_df['AGE'] = current_year - self.patients_df['BIRTHDATE'].dt.year
        
        # Handle death date
        self.patients_df['IS_DECEASED'] = self.patients_df['DEATHDATE'].notna().astype(int)
        
        # Encode categorical features
        self.patients_df['GENDER_ENCODED'] = self._encode_column('GENDER', self.patients_df['GENDER'])
        self.patients_df['RACE_ENCODED'] = self._encode_column('RACE', self.patients_df['RACE'])
        self.patients_df['ETHNICITY_ENCODED'] = self._encode_column('ETHNICITY', self.patients_df['ETHNICITY'])
        self.patients_df['MARITAL_ENCODED'] = self._encode_column('MARITAL', self.patients_df['MARITAL'].fillna('S'))
        
        # Extract financial features
        self.patients_df['INCOME'] = pd.to_numeric(self.patients_df['INCOME'], errors='coerce').fillna(0)
        self.patients_df['HEALTHCARE_EXPENSES'] = pd.to_numeric(self.patients_df['HEALTHCARE_EXPENSES'], errors='coerce').fillna(0)
        self.patients_df['HEALTHCARE_COVERAGE'] = pd.to_numeric(self.patients_df['HEALTHCARE_COVERAGE'], errors='coerce').fillna(0)
        
        # Create target labels from conditions
        logger.info("Creating target labels...")
        for condition in self.target_conditions:
            label_col = self._condition_to_column_name(condition)
            self.patients_df[label_col] = self.patients_df['Id'].apply(
                lambda patient_id: self._has_condition(patient_id, condition)
            )
        
        logger.info(f"Preprocessed {len(self.patients_df)} patient records")
        return self.patients_df
    
    def _encode_column(self, col_name: str, data: pd.Series) -> np.ndarray:
        """Encode categorical column using LabelEncoder"""
        if col_name not in self.label_encoders:
            self.label_encoders[col_name] = LabelEncoder()
            return self.label_encoders[col_name].fit_transform(data.fillna('Unknown'))
        else:
            return self.label_encoders[col_name].transform(data.fillna('Unknown'))
    
    def _has_condition(self, patient_id: str, condition_description: str) -> int:
        """Check if patient has ever had a specific condition"""
        patient_conditions = self.conditions_df[
            self.conditions_df['PATIENT'] == patient_id
        ]
        has_condition = (patient_conditions['DESCRIPTION'] == condition_description).any()
        return 1 if has_condition else 0
    
    def _condition_to_column_name(self, condition: str) -> str:
        """Convert condition description to column name"""
        return 'HAS_' + condition.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').upper()
    
    def create_federated_splits(
        self,
        n_clients: int,
        stratify_by: str = 'STATE'
    ) -> Dict[int, pd.DataFrame]:
        """
        Split data into federated client datasets
        
        Args:
            n_clients: Number of federated clients
            stratify_by: Column to stratify by (default: STATE for geographic distribution)
        
        Returns:
            Dictionary mapping client_id to their data subset
        """
        if self.patients_df is None:
            self.preprocess_data()
        
        logger.info(f"Creating {n_clients} federated data splits...")
        
        client_data = {}
        
        if stratify_by in self.patients_df.columns:
            # Stratified split by location/attribute
            unique_strata = self.patients_df[stratify_by].dropna().unique()
            
            # Assign strata to clients (round-robin)
            strata_to_client = {}
            for idx, stratum in enumerate(unique_strata):
                strata_to_client[stratum] = idx % n_clients
            
            # Split data
            for client_id in range(n_clients):
                assigned_strata = [k for k, v in strata_to_client.items() if v == client_id]
                client_data[client_id] = self.patients_df[
                    self.patients_df[stratify_by].isin(assigned_strata)
                ].copy()
                
                logger.info(f"Client {client_id}: {len(client_data[client_id])} patients from {len(assigned_strata)} regions")
        else:
            # Random split if stratification column not available
            shuffled_df = self.patients_df.sample(frac=1, random_state=42).reset_index(drop=True)
            chunk_size = len(shuffled_df) // n_clients
            
            for client_id in range(n_clients):
                start_idx = client_id * chunk_size
                end_idx = start_idx + chunk_size if client_id < n_clients - 1 else len(shuffled_df)
                client_data[client_id] = shuffled_df.iloc[start_idx:end_idx].copy()
                
                logger.info(f"Client {client_id}: {len(client_data[client_id])} patients")
        
        return client_data
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for model training"""
        return [
            'AGE',
            'GENDER_ENCODED',
            'RACE_ENCODED',
            'ETHNICITY_ENCODED',
            'MARITAL_ENCODED',
            'IS_DECEASED',
            'INCOME',
            'HEALTHCARE_EXPENSES',
            'HEALTHCARE_COVERAGE'
        ]
    
    def get_target_columns(self) -> List[str]:
        """Get list of target condition columns"""
        return [self._condition_to_column_name(cond) for cond in self.target_conditions]
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_condition_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare X, y for training
        
        Args:
            df: DataFrame with features and targets
            target_condition_idx: Index of target condition to predict (0-5)
        
        Returns:
            X (features), y (labels)
        """
        feature_cols = self.get_feature_columns()
        target_cols = self.get_target_columns()
        
        # Select features
        X = df[feature_cols].values
        
        # Select target (single condition for binary classification)
        y = df[target_cols[target_condition_idx]].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        return X, y
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict:
        """Get statistics about the dataset"""
        target_cols = self.get_target_columns()
        
        stats = {
            'n_samples': len(df),
            'n_features': len(self.get_feature_columns()),
            'n_conditions': len(target_cols),
            'condition_prevalence': {}
        }
        
        for col in target_cols:
            if col in df.columns:
                prevalence = df[col].sum() / len(df)
                stats['condition_prevalence'][col] = prevalence
        
        return stats


def demonstrate_data_loading():
    """Demonstration of data loading and preprocessing"""
    import os
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Get data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir  # patients.csv and conditions.csv are in fl/ directory
    
    print("=" * 70)
    print("PQBFL Healthcare Data Loader - Demonstration")
    print("=" * 70)
    
    # Initialize loader
    loader = HealthcareDataLoader(data_dir)
    
    # Load and preprocess
    loader.load_data()
    processed_df = loader.preprocess_data()
    
    print(f"\n✓ Loaded {len(processed_df)} patients")
    print(f"✓ Age range: {processed_df['AGE'].min()}-{processed_df['AGE'].max()} years")
    print(f"✓ Feature columns: {len(loader.get_feature_columns())}")
    print(f"✓ Target conditions: {len(loader.get_target_columns())}")
    
    # Show condition prevalence
    print("\nCondition Prevalence:")
    for col in loader.get_target_columns():
        count = processed_df[col].sum()
        pct = (count / len(processed_df)) * 100
        print(f"  {col}: {count}/{len(processed_df)} ({pct:.1f}%)")
    
    # Create federated splits
    print("\nCreating Federated Splits:")
    client_splits = loader.create_federated_splits(n_clients=3, stratify_by='STATE')
    
    for client_id, data in client_splits.items():
        stats = loader.get_data_statistics(data)
        print(f"\nClient {client_id}:")
        print(f"  Samples: {stats['n_samples']}")
        print(f"  Diabetes prevalence: {stats['condition_prevalence'].get('HAS_DIABETES_MELLITUS_TYPE_2_DISORDER', 0):.2%}")
        print(f"  Hypertension prevalence: {stats['condition_prevalence'].get('HAS_ESSENTIAL_HYPERTENSION_DISORDER', 0):.2%}")
    
    # Prepare sample training data
    print("\nPreparing Training Data (Diabetes Prediction):")
    X, y = loader.prepare_training_data(processed_df, target_condition_idx=0)
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Positive samples: {y.sum()}/{len(y)} ({y.sum()/len(y):.2%})")
    
    print("\n" + "=" * 70)
    print("Data Loading Complete!")
    print("=" * 70)


if __name__ == '__main__':
    demonstrate_data_loading()

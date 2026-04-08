"""
preprocess.py - Data Cleaning and Preprocessing Module
AI-Based Intrusion Detection System (IDS)

Handles: loading, cleaning, encoding, and normalizing the NSL-KDD dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# NSL-KDD column definitions
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

# Attack category mappings (NSL-KDD)
ATTACK_CATEGORY_MAP = {
    "normal": "Normal",
    # DoS attacks
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS",
    # Probe attacks
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L attacks
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R attacks
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
LABEL_COLUMN = "label"
CATEGORY_COLUMN = "attack_category"


def generate_synthetic_nslkdd(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate a synthetic NSL-KDD-like dataset for demonstration purposes
    when no real dataset is uploaded. Mirrors real feature distributions.
    """
    np.random.seed(42)
    logger.info(f"Generating synthetic NSL-KDD dataset with {n_samples} samples...")

    protocols = ["tcp", "udp", "icmp"]
    services = ["http", "ftp", "smtp", "ssh", "dns", "ftp_data", "telnet", "pop_3", "domain_u", "auth"]
    flags = ["SF", "S0", "REJ", "RSTO", "SH", "RSTR", "S1", "S2", "S3", "OTH"]

    categories = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    category_weights = [0.53, 0.23, 0.12, 0.08, 0.04]

    attack_cat = np.random.choice(categories, size=n_samples, p=category_weights)

    data = {
        "duration": np.random.exponential(5, n_samples).astype(int),
        "protocol_type": np.random.choice(protocols, n_samples),
        "service": np.random.choice(services, n_samples),
        "flag": np.random.choice(flags, n_samples, p=[0.6, 0.1, 0.1, 0.05, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02]),
        "src_bytes": np.where(attack_cat == "DoS",
                              np.random.randint(0, 100, n_samples),
                              np.random.randint(0, 50000, n_samples)),
        "dst_bytes": np.random.randint(0, 100000, n_samples),
        "land": np.random.randint(0, 2, n_samples),
        "wrong_fragment": np.random.randint(0, 4, n_samples),
        "urgent": np.random.randint(0, 2, n_samples),
        "hot": np.random.randint(0, 30, n_samples),
        "num_failed_logins": np.where(attack_cat == "R2L",
                                      np.random.randint(1, 6, n_samples),
                                      np.random.randint(0, 2, n_samples)),
        "logged_in": np.random.randint(0, 2, n_samples),
        "num_compromised": np.where(attack_cat == "U2R",
                                    np.random.randint(1, 10, n_samples),
                                    np.zeros(n_samples, dtype=int)),
        "root_shell": np.where(attack_cat == "U2R",
                               np.random.randint(0, 2, n_samples),
                               np.zeros(n_samples, dtype=int)),
        "su_attempted": np.random.randint(0, 2, n_samples),
        "num_root": np.random.randint(0, 5, n_samples),
        "num_file_creations": np.random.randint(0, 5, n_samples),
        "num_shells": np.random.randint(0, 3, n_samples),
        "num_access_files": np.random.randint(0, 5, n_samples),
        "num_outbound_cmds": np.zeros(n_samples, dtype=int),
        "is_host_login": np.random.randint(0, 2, n_samples),
        "is_guest_login": np.random.randint(0, 2, n_samples),
        "count": np.where(attack_cat == "DoS",
                          np.random.randint(200, 512, n_samples),
                          np.random.randint(1, 200, n_samples)),
        "srv_count": np.random.randint(1, 512, n_samples),
        "serror_rate": np.where(attack_cat == "DoS",
                                np.random.uniform(0.8, 1.0, n_samples),
                                np.random.uniform(0, 0.2, n_samples)),
        "srv_serror_rate": np.random.uniform(0, 1, n_samples),
        "rerror_rate": np.random.uniform(0, 1, n_samples),
        "srv_rerror_rate": np.random.uniform(0, 1, n_samples),
        "same_srv_rate": np.random.uniform(0, 1, n_samples),
        "diff_srv_rate": np.random.uniform(0, 1, n_samples),
        "srv_diff_host_rate": np.random.uniform(0, 1, n_samples),
        "dst_host_count": np.random.randint(1, 256, n_samples),
        "dst_host_srv_count": np.random.randint(1, 256, n_samples),
        "dst_host_same_srv_rate": np.random.uniform(0, 1, n_samples),
        "dst_host_diff_srv_rate": np.random.uniform(0, 1, n_samples),
        "dst_host_same_src_port_rate": np.random.uniform(0, 1, n_samples),
        "dst_host_srv_diff_host_rate": np.random.uniform(0, 1, n_samples),
        "dst_host_serror_rate": np.random.uniform(0, 1, n_samples),
        "dst_host_srv_serror_rate": np.random.uniform(0, 1, n_samples),
        "dst_host_rerror_rate": np.random.uniform(0, 1, n_samples),
        "dst_host_srv_rerror_rate": np.random.uniform(0, 1, n_samples),
        "attack_category": attack_cat,
        "label": [cat.lower() if cat == "Normal" else cat.lower() + "_attack" for cat in attack_cat],
    }

    df = pd.DataFrame(data)
    logger.info("Synthetic dataset generated successfully.")
    return df


def load_nslkdd(filepath: str) -> pd.DataFrame:
    """Load NSL-KDD dataset from a .csv or .txt file."""
    logger.info(f"Loading dataset from: {filepath}")
    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext in [".txt", ".data", ".csv"]:
            # Try with NSL-KDD columns first
            try:
                df = pd.read_csv(filepath, header=None, names=NSL_KDD_COLUMNS)
            except Exception:
                df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        raise

    logger.info(f"Loaded dataset shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values: fill numerics with median, categoricals with mode."""
    logger.info("Handling missing values...")
    missing_before = df.isnull().sum().sum()

    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            fill_val = df[col].mode()[0] if not df[col].mode().empty else "unknown"
            df[col] = df[col].fillna(fill_val)

    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values: {missing_before} → {missing_after}")
    return df


def map_attack_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Map granular attack labels to 5 high-level categories."""
    logger.info("Mapping attack labels to categories...")
    if LABEL_COLUMN in df.columns:
        df[CATEGORY_COLUMN] = df[LABEL_COLUMN].str.strip().map(
            lambda x: ATTACK_CATEGORY_MAP.get(x.lower(), x if x in ["Normal","DoS","Probe","R2L","U2R"] else "Unknown")
        )
        unknown_count = (df[CATEGORY_COLUMN] == "Unknown").sum()
        if unknown_count > 0:
            logger.warning(f"{unknown_count} samples mapped to 'Unknown' category.")
    elif CATEGORY_COLUMN not in df.columns:
        raise ValueError("Dataset must contain a 'label' or 'attack_category' column.")
    return df


def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical features and return encoders."""
    logger.info("Encoding categorical features...")
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logger.info(f"  Encoded '{col}' → {len(le.classes_)} classes")
    return df, encoders


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Encode attack category labels to integers."""
    logger.info("Encoding target labels...")
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df[CATEGORY_COLUMN])
    logger.info(f"  Label classes: {list(le.classes_)}")
    return df, le


def scale_features(df: pd.DataFrame, method: str = "standard",
                   scaler=None) -> tuple[pd.DataFrame, object]:
    """Normalize numerical features using StandardScaler or MinMaxScaler."""
    logger.info(f"Scaling features using {method}Scaler...")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target and encoded label columns
    exclude = ["label_encoded", "difficulty"]
    numerical_cols = [c for c in numerical_cols if c not in exclude]

    if scaler is None:
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        df[numerical_cols] = scaler.transform(df[numerical_cols])

    logger.info(f"  Scaled {len(numerical_cols)} numerical features.")
    return df, scaler


def split_data(df: pd.DataFrame, test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """Split dataset into train/test sets."""
    feature_cols = [c for c in df.columns
                    if c not in [LABEL_COLUMN, CATEGORY_COLUMN, "label_encoded",
                                 "difficulty", "attack_category"]]
    X = df[feature_cols]
    y = df["label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def full_preprocess_pipeline(df: pd.DataFrame, scaler_method: str = "standard",
                              test_size: float = 0.2) -> dict:
    """
    Run the complete preprocessing pipeline end-to-end.

    Returns a dict with: X_train, X_test, y_train, y_test,
                          label_encoder, feature_encoders, scaler, df_processed
    """
    logger.info("=== Starting Full Preprocessing Pipeline ===")
    df = handle_missing_values(df)
    df = map_attack_categories(df)
    df, feature_encoders = encode_categorical(df)
    df, label_encoder = encode_labels(df)

    # Drop difficulty column if present (NSL-KDD specific)
    if "difficulty" in df.columns:
        df.drop(columns=["difficulty"], inplace=True)

    df, scaler = scale_features(df, method=scaler_method)
    X_train, X_test, y_train, y_test = split_data(df, test_size=test_size)

    logger.info("=== Preprocessing Complete ===")
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": label_encoder,
        "feature_encoders": feature_encoders,
        "scaler": scaler,
        "df_processed": df,
        "feature_cols": list(X_train.columns),
    }
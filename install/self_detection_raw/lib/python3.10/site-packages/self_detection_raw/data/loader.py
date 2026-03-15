"""
Robust parser for robot_data_*.txt files.

Handles:
- Comment lines (#)
- Token cleaning (remove non-numeric chars)
- Legacy 31-column rows (missing joint velocities)
- Base 37-column rows
- Extended 53-column rows with trailing comp/pred values
- Streaming parsing for large files
"""

import re
import numpy as np
from typing import List, Tuple, Optional, Iterator
import logging

# Setup logging if not already configured
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Supported column counts
LEGACY_COLS_NO_JV = 31
EXPECTED_COLS = 37
EXTENDED_COLS = 53

# Column indices (0-based)
IDX_TIMESTAMP = 0
IDX_J1 = 1
IDX_J6 = 6
IDX_JV1 = 7
IDX_JV6 = 12
IDX_PROX1 = 13
IDX_PROX8 = 20
IDX_RAW1 = 21
IDX_RAW8 = 28
IDX_TOF1 = 29
IDX_TOF8 = 36

_extended_format_logged = False


def clean_token(token: str) -> Optional[float]:
    """
    Clean token and try to parse as float.
    
    Args:
        token: String token
        
    Returns:
        Parsed float or None if failed
    """
    # Remove non-numeric characters (keep digits, e, E, +, -, .)
    cleaned = re.sub(r'[^0-9eE\+\-\.]', '', token)
    
    try:
        return float(cleaned)
    except (ValueError, OverflowError):
        return None


def parse_line(line: str, line_num: int) -> Optional[np.ndarray]:
    """
    Parse a single line from robot_data file.
    
    Args:
        line: Raw line string
        line_num: Line number (for error reporting)
        
    Returns:
        (37,) array or None if parsing failed
    """
    # Skip comments
    if line.lstrip().startswith('#'):
        return None
    
    # Strip and split
    line = line.strip()
    if not line:
        return None
    
    tokens = line.split(',')
    
    # Parse tokens
    values = []
    for i, token in enumerate(tokens):
        token = token.strip()
        
        # First column (timestamp) can be date format, skip if not numeric
        if i == 0:
            # Try to parse as float, if fails use 0.0 (timestamp not needed for training)
            try:
                val = float(token)
            except ValueError:
                # Timestamp in date format, use 0.0
                val = 0.0
        else:
            try:
                val = float(token)
            except ValueError:
                # Try cleaning
                val = clean_token(token)
                if val is None:
                    logger.warning(f"Line {line_num}, col {i}: Failed to parse '{token}', dropping line")
                    return None
        values.append(val)
    
    # Handle legacy/base/extended formats.
    if len(values) == LEGACY_COLS_NO_JV:
        # Insert 6 zeros for jv1..jv6 after j6 (index 7).
        values = values[:7] + [0.0] * 6 + values[7:]
        logger.debug(f"Line {line_num}: Inserted missing jv columns (zeros)")
    elif len(values) == EXTENDED_COLS:
        global _extended_format_logged
        if not _extended_format_logged:
            logger.info(
                "Detected extended %d-column format; using the first %d columns "
                "(timestamp/joints/velocity/prox/raw/tof) and ignoring trailing comp/pred columns",
                EXTENDED_COLS,
                EXPECTED_COLS,
            )
            _extended_format_logged = True
        values = values[:EXPECTED_COLS]
    elif len(values) < EXPECTED_COLS:
        logger.warning(f"Line {line_num}: Expected {EXPECTED_COLS} or {EXTENDED_COLS} columns, got {len(values)}, dropping")
        return None
    elif len(values) > EXPECTED_COLS:
        logger.warning(
            f"Line {line_num}: Expected {EXPECTED_COLS} or {EXTENDED_COLS} columns, got {len(values)}, truncating"
        )
        values = values[:EXPECTED_COLS]
    
    return np.array(values, dtype=np.float32)


def load_file_streaming(filepath: str) -> Iterator[np.ndarray]:
    """
    Load file line-by-line (streaming).
    
    Args:
        filepath: Path to robot_data file
        
    Yields:
        (37,) arrays for each valid line
    """
    dropped_count = 0
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            if line.lstrip().startswith('#') or not line.strip():
                continue
            parsed = parse_line(line, line_num)
            if parsed is not None:
                yield parsed
            else:
                dropped_count += 1
    
    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count} lines from {filepath}")


def load_file(filepath: str) -> np.ndarray:
    """
    Load entire file into memory.
    
    Args:
        filepath: Path to robot_data file
        
    Returns:
        (N, 37) array
    """
    rows = list(load_file_streaming(filepath))
    if not rows:
        raise ValueError(f"No valid data rows found in {filepath}")
    return np.stack(rows, axis=0)


def extract_features(data: np.ndarray, use_vel: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract input features (X) and targets (Y) from parsed data.
    
    Args:
        data: (N, 37) parsed data array
        use_vel: Whether to use joint velocities (DEPRECATED: always False, joint velocities removed)
        
    Returns:
        (X, Y) tuple:
            X: (N, 12) features [sin(j1..j6), cos(j1..j6)] - joint velocities removed
            Y: (N, 8) targets [raw1..raw8]
    """
    # Joint positions (degrees) -> radians
    j_pos_deg = data[:, IDX_J1:IDX_J6+1]  # (N, 6)
    j_pos_rad = np.deg2rad(j_pos_deg)
    
    # sin/cos transformation
    sin_j = np.sin(j_pos_rad)  # (N, 6)
    cos_j = np.cos(j_pos_rad)  # (N, 6)
    
    # Joint velocities removed - only use sin/cos
    # Concatenate: [sin, cos] -> (N, 12)
    X = np.concatenate([sin_j, cos_j], axis=1)
    
    # Targets: raw1..raw8
    Y = data[:, IDX_RAW1:IDX_RAW8+1]  # (N, 8)
    
    return X, Y


def load_and_extract(filepath: str, use_vel: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load file and extract features/targets.
    
    Args:
        filepath: Path to robot_data file
        use_vel: Whether to use joint velocities
        
    Returns:
        (X, Y) tuple
    """
    data = load_file(filepath)
    return extract_features(data, use_vel=use_vel)


def load_multiple_files(filepaths: List[str], use_vel: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load multiple files and concatenate.
    
    Args:
        filepaths: List of file paths (can be empty)
        use_vel: Whether to use joint velocities
        
    Returns:
        (X, Y) tuple with concatenated data
    """
    if not filepaths:
        # Return empty arrays with correct shape
        return np.array([]).reshape(0, 18), np.array([]).reshape(0, 8)
    
    all_X = []
    all_Y = []
    
    for filepath in filepaths:
        X, Y = load_and_extract(filepath, use_vel=use_vel)
        all_X.append(X)
        all_Y.append(Y)
    
    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)
    
    return X, Y


def split_files_train_val(
    filepaths: List[str],
    val_ratio: float = 0.2,
    split_mode: str = 'file',
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Split files into train/val sets.
    
    Args:
        filepaths: List of file paths
        val_ratio: Validation ratio
        split_mode: 'file' = file-level split (time-ordered), 'random' = random split
        seed: Random seed
        
    Returns:
        (train_files, val_files) tuple
    """
    import random
    import logging
    logger = logging.getLogger(__name__)
    
    random.seed(seed)
    np.random.seed(seed)
    
    # 파일이 1개일 때는 경고하고 train에 할당
    if len(filepaths) == 1:
        logger.warning(
            f"파일이 1개만 있습니다. 파일 내부에서 train/val split을 하려면 "
            f"--val_split within 옵션을 사용하거나, 더 많은 파일을 추가하세요. "
            f"현재는 모든 데이터를 train으로 사용합니다."
        )
        return filepaths, []
    
    if split_mode == 'file':
        # File-level split: sort by filename (time order), take last N% as val
        sorted_files = sorted(filepaths)
        n_val = max(1, int(len(sorted_files) * val_ratio))
        # 최소 1개는 train에 남겨둠
        if n_val >= len(sorted_files):
            n_val = len(sorted_files) - 1
            logger.warning(
                f"val_ratio가 너무 커서 train 파일이 없어질 수 있습니다. "
                f"val 파일 수를 {n_val}개로 조정합니다."
            )
        val_files = sorted_files[-n_val:]
        train_files = sorted_files[:-n_val]
    else:
        # Random file split
        indices = list(range(len(filepaths)))
        random.shuffle(indices)
        n_val = max(1, int(len(filepaths) * val_ratio))
        # 최소 1개는 train에 남겨둠
        if n_val >= len(filepaths):
            n_val = len(filepaths) - 1
            logger.warning(
                f"val_ratio가 너무 커서 train 파일이 없어질 수 있습니다. "
                f"val 파일 수를 {n_val}개로 조정합니다."
            )
        val_indices = set(indices[:n_val])
        train_files = [filepaths[i] for i in range(len(filepaths)) if i not in val_indices]
        val_files = [filepaths[i] for i in range(len(filepaths)) if i in val_indices]
    
    return train_files, val_files

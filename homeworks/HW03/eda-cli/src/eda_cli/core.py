# src/eda_cli/core.py
import pandas as pd
from typing import Dict, Any, List

def compute_quality_flags(
    df: pd.DataFrame, 
    min_missing_share: float = 0.3,
    high_cardinality_threshold: int = 50
) -> Dict[str, Any]:
    """
    Вычисляет флаги качества данных с учетом параметров.
    
    Args:
        df: DataFrame для анализа
        min_missing_share: порог для проблемных пропусков
        high_cardinality_threshold: порог для высокой кардинальности
    
    Returns:
        Словарь с флагами и метриками качества
    """
    n_rows, n_cols = df.shape
    
    # Базовые метрики
    total_cells = n_rows * n_cols
    missing_cells = df.isnull().sum().sum()
    missing_share = missing_cells / total_cells if total_cells > 0 else 0
    duplicate_rows = df.duplicated().sum()
    
    # Списки проблемных колонок на основе параметров
    problematic_missing_cols = []
    for col in df.columns:
        missing_ratio = df[col].isnull().sum() / n_rows
        if missing_ratio > min_missing_share:
            problematic_missing_cols.append({
                'column': col,
                'missing_ratio': missing_ratio
            })
    
    # НОВЫЕ ЭВРИСТИКИ
    # 1. Константные колонки
    constant_cols = []
    for col in df.columns:
        if df[col].nunique(dropna=True) == 1:
            constant_cols.append(col)
    
    # 2. Категориальные с высокой кардинальностью
    high_cardinality_cols = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_count = df[col].nunique()
        if unique_count > high_cardinality_threshold:
            high_cardinality_cols[col] = unique_count
    
    # 3. Проверка ID на уникальность
    suspicious_id_duplicates = {}
    # Проверяем колонки, которые могут быть ID
    possible_id_cols = [col for col in df.columns 
                       if 'id' in col.lower() or col.lower() in ['id', 'index', 'key']]
    
    for col in possible_id_cols:
        duplicate_count = df[col].duplicated().sum()
        if duplicate_count > 0:
            suspicious_id_duplicates[col] = duplicate_count
    
    # Расчет интегрального score с учетом новых факторов
    base_score = 1.0
    
    # Штрафы
    penalties = {
        'missing': missing_share * 0.5,  # до 0.5 за пропуски
        'duplicates': min(0.2, duplicate_rows / n_rows * 0.5),
        'constant': len(constant_cols) * 0.1,
        'high_cardinality': len(high_cardinality_cols) * 0.05,
        'id_duplicates': len(suspicious_id_duplicates) * 0.15
    }
    
    quality_score = base_score - sum(penalties.values())
    quality_score = max(0.0, min(1.0, quality_score))
    
    return {
        # Базовые метрики
        'n_rows': n_rows,
        'n_cols': n_cols,
        'missing_share': missing_share,
        'has_missing': missing_cells > 0,
        'duplicate_rows': duplicate_rows,
        'has_duplicates': duplicate_rows > 0,
        'quality_score': quality_score,
        
        # Параметры анализа
        'min_missing_share': min_missing_share,
        'high_cardinality_threshold': high_cardinality_threshold,
        
        # Списки проблемных колонок
        'problematic_missing_cols': problematic_missing_cols,
        
        # Новые флаги
        'has_constant_columns': len(constant_cols) > 0,
        'constant_columns_list': constant_cols,
        
        'has_high_cardinality_categoricals': len(high_cardinality_cols) > 0,
        'high_cardinality_columns': high_cardinality_cols,
        
        'has_suspicious_id_duplicates': len(suspicious_id_duplicates) > 0,
        'suspicious_id_duplicates': suspicious_id_duplicates,
        
        # Дополнительные метрики (опционально)
        'numeric_columns_count': len(df.select_dtypes(include=['number']).columns),
        'categorical_columns_count': len(df.select_dtypes(include=['object', 'category']).columns),
        'date_columns_count': len(df.select_dtypes(include=['datetime']).columns)
    }

def generate_overview(df: pd.DataFrame) -> str:
    """Генерирует краткую текстовую сводку."""
    flags = compute_quality_flags(df)
    
    overview_lines = [
        "📊 ОБЗОР ДАННЫХ",
        "=" * 50,
        f"Размер: {flags['n_rows']} строк × {flags['n_cols']} колонок",
        f"Пропуски: {flags['missing_share']:.1%}",
        f"Дубликаты строк: {flags['duplicate_rows']}",
        f"Качество данных (score): {flags['quality_score']:.2f}/1.00",
        "",
        "ПРОБЛЕМНЫЕ КОЛОНКИ:"
    ]
    
    # Добавляем проблемные колонки по пропускам
    if flags['problematic_missing_cols']:
        overview_lines.append("  • С высоким % пропусков (>30%):")
        for item in flags['problematic_missing_cols']:
            overview_lines.append(f"    - {item['column']}: {item['missing_ratio']:.1%}")
    
    # Добавляем константные колонки
    if flags['has_constant_columns']:
        overview_lines.append(f"  • Константные: {', '.join(flags['constant_columns_list'])}")
    
    # Добавляем колонки с высокой кардинальностью
    if flags['has_high_cardinality_categoricals']:
        overview_lines.append("  • Высокая кардинальность:")
        for col, count in flags['high_cardinality_columns'].items():
            overview_lines.append(f"    - {col}: {count} уникальных значений")
    
    return "\n".join(overview_lines)
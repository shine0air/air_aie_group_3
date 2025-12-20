import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import numpy as np

def save_histograms(
    df: pd.DataFrame, 
    out_dir: Path, 
    max_columns: int = 5
) -> List[str]:
    """
    Сохраняет гистограммы для числовых колонок.
    
    Args:
        max_columns: максимальное количество гистограмм для построения
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    saved_images = []
    
    cols_to_plot = numeric_cols[:max_columns]
    
    for i, col in enumerate(cols_to_plot):
        plt.figure(figsize=(10, 6))
        
        data = df[col].dropna()
        
        if len(data) > 0:
            plt.hist(data, bins=min(30, len(data)//10 + 1), edgecolor='black', alpha=0.7)
            plt.title(f'Распределение: {col}')
            plt.xlabel(col)
            plt.ylabel('Частота')
            
            stats_text = f'n={len(data)}\nmean={data.mean():.2f}\nstd={data.std():.2f}'
            plt.text(0.7, 0.7, stats_text, transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            filename = out_dir / f'hist_{col.replace(" ", "_")}.png'
            plt.tight_layout()
            plt.savefig(filename, dpi=100)
            plt.close()
            
            saved_images.append(filename.name)
        else:
            plt.close()
    
    if len(numeric_cols) > max_columns:
        print(f"⚠️ Построено только {max_columns} из {len(numeric_cols)} числовых колонок.")
        print(f"   Используйте --max-hist-columns для увеличения лимита.")
    
    return saved_images

def create_report(
    df: pd.DataFrame,
    quality_flags: Dict[str, Any],
    out_dir: Path,
    title: str = "EDA Report",
    max_hist_columns: int = 5,
    top_k_categories: int = 10,
    min_missing_share: float = 0.3,
    high_cardinality_threshold: int = 50
) -> None:
    """
    Создает полный отчет в формате Markdown.
    """
    report_lines = []
    
    report_lines.append(f"# {title}")
    report_lines.append(f"*Сгенерировано с помощью eda-cli*\n")
    
    report_lines.append("## ⚙️ Параметры анализа")
    report_lines.append(f"- **Макс. гистограмм:** {max_hist_columns}")
    report_lines.append(f"- **Топ-K категорий:** {top_k_categories}")
    report_lines.append(f"- **Порог проблемных пропусков:** {min_missing_share:.0%}")
    report_lines.append(f"- **Порог высокой кардинальности:** {high_cardinality_threshold}")
    report_lines.append("")
    report_lines.append("## 📊 Общая информация")
    report_lines.append(f"- **Размер данных:** {quality_flags['n_rows']} строк × {quality_flags['n_cols']} колонок")
    report_lines.append(f"- **Числовые колонки:** {quality_flags.get('numeric_columns_count', 0)}")
    report_lines.append(f"- **Категориальные колонки:** {quality_flags.get('categorical_columns_count', 0)}")
    report_lines.append(f"- **Колонки с датами:** {quality_flags.get('date_columns_count', 0)}")
    report_lines.append("")
    

    report_lines.append("## 🔍 Качество данных")
    report_lines.append(f"- **Интегральный score:** `{quality_flags['quality_score']:.2f}/1.00`")
    
    missing_info = []
    if quality_flags['problematic_missing_cols']:
        for item in quality_flags['problematic_missing_cols'][:10]:
            missing_info.append(f"  - `{item['column']}`: {item['missing_ratio']:.1%}")
    
    if missing_info:
        report_lines.append(f"- **⚠️ Колонки с >{min_missing_share:.0%} пропусков:**")
        report_lines.extend(missing_info)
        if len(quality_flags['problematic_missing_cols']) > 10:
            report_lines.append(f"  *... и еще {len(quality_flags['problematic_missing_cols']) - 10} колонок*")
    else:
        report_lines.append(f"- **✓ Нет колонок с >{min_missing_share:.0%} пропусков**")
    report_lines.append("")
    
    if quality_flags['has_duplicates']:
        report_lines.append(f"- **⚠️ Обнаружены дубликаты строк:** {quality_flags['duplicate_rows']}")
    else:
        report_lines.append("- **✓ Нет дубликатов строк**")
    report_lines.append("")
    
    if quality_flags['has_constant_columns']:
        report_lines.append(f"- **⚠️ Константные колонки:**")
        for col in quality_flags['constant_columns_list']:
            report_lines.append(f"  - `{col}`")
    else:
        report_lines.append("- **✓ Нет константных колонок**")
    report_lines.append("")
    
    if quality_flags['has_high_cardinality_categoricals']:
        report_lines.append(f"- **⚠️ Высокая кардинальность (> {high_cardinality_threshold} уникальных значений):**")
        for col, count in quality_flags['high_cardinality_columns'].items():
            report_lines.append(f"  - `{col}`: {count} уникальных значений")
    else:
        report_lines.append("- **✓ Нет категориальных признаков с высокой кардинальностью**")
    report_lines.append("")
    
    if quality_flags['has_suspicious_id_duplicates']:
        report_lines.append("- **⚠️ Возможные проблемы с уникальностью ID:**")
        for col, count in quality_flags['suspicious_id_duplicates'].items():
            report_lines.append(f"  - `{col}`: {count} дубликатов")
    report_lines.append("")
    
    report_lines.append(f"## 📈 Топ-{top_k_categories} значений по категориальным признакам")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) > 0:
        for col in categorical_cols[:10]:
            value_counts = df[col].value_counts()
            total = len(df[col].dropna())
            
            if total > 0:
                report_lines.append(f"### `{col}`")
                report_lines.append(f"Всего значений: {total} | Уникальных: {value_counts.shape[0]}")
                
                top_values = value_counts.head(top_k_categories)
                for value, count in top_values.items():
                    percentage = count / total * 100
                    report_lines.append(f"- `{value}`: {count} ({percentage:.1f}%)")
                
                if len(value_counts) > top_k_categories:
                    other_count = value_counts.iloc[top_k_categories:].sum()
                    other_pct = other_count / total * 100
                    report_lines.append(f"- *... и еще {len(value_counts) - top_k_categories} значений: {other_count} ({other_pct:.1f}%)*")
                
                report_lines.append("")
    else:
        report_lines.append("Категориальные признаки отсутствуют.\n")
    
    report_lines.append(f"## 📊 Гистограммы числовых признаков")
    report_lines.append(f"*Построено гистограмм: {max_hist_columns} из {quality_flags.get('numeric_columns_count', 0)} числовых колонок*")
    
    saved_images = save_histograms(df, out_dir, max_columns=max_hist_columns)
    
    if saved_images:
        for img in saved_images:
            report_lines.append(f"![{img}]({img})")
        report_lines.append("")
    else:
        report_lines.append("Числовые признаки для гистограмм отсутствуют.\n")
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        report_lines.append("## 📐 Статистика по числовым колонкам")
        report_lines.append("| Колонка | Среднее | Медиана | Std | Min | Max |")
        report_lines.append("|---------|---------|---------|-----|-----|-----|")
        
        for col in numeric_cols[:15]:
            stats = df[col].describe()
            report_lines.append(
                f"| `{col}` | {stats.get('mean', 'NA'):.2f} | {stats.get('50%', 'NA'):.2f} | "
                f"{stats.get('std', 'NA'):.2f} | {stats.get('min', 'NA'):.2f} | {stats.get('max', 'NA'):.2f} |"
            )
        
        if len(numeric_cols) > 15:
            report_lines.append(f"| *... и еще {len(numeric_cols) - 15} колонок* |")
        report_lines.append("")

    report_lines.append("## 🎯 Выводы")
    score = quality_flags['quality_score']
    
    if score > 0.8:
        report_lines.append("✅ **Отличное качество данных.** Можно сразу переходить к моделированию.")
    elif score > 0.6:
        report_lines.append("⚠️ **Среднее качество данных.** Рекомендуется очистка данных.")
    elif score > 0.4:
        report_lines.append("⚠️ **Низкое качество данных.** Требуется значительная предобработка.")
    else:
        report_lines.append("❌ **Критическое качество данных.** Возможно, требуется поиск нового датасета.")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append(f"*Отчет сгенерирован с параметрами: max_hist_columns={max_hist_columns}, "
                       f"top_k_categories={top_k_categories}, min_missing_share={min_missing_share:.0%}*")
    
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    
    import json
    summary = {
        "title": title,
        "parameters": {
            "max_hist_columns": max_hist_columns,
            "top_k_categories": top_k_categories,
            "min_missing_share": min_missing_share,
            "high_cardinality_threshold": high_cardinality_threshold
        },
        "basic_metrics": {
            "n_rows": quality_flags['n_rows'],
            "n_cols": quality_flags['n_cols'],
            "quality_score": quality_flags['quality_score']
        },
        "problematic_columns": {
            "high_missing": [item['column'] for item in quality_flags['problematic_missing_cols']],
            "constant": quality_flags['constant_columns_list'],
            "high_cardinality": list(quality_flags['high_cardinality_columns'].keys())
        }
    }
    
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
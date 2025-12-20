# src/eda_cli/cli.py
import typer
from pathlib import Path
import pandas as pd
from eda_cli import core, viz
from typing import Optional

app = typer.Typer(help="CLI для анализа данных")

@app.command()
def overview(file_path: Path):
    """Краткая сводка о датасете."""
    df = pd.read_csv(file_path)
    print(core.generate_overview(df))

@app.command()
def report(
    file_path: Path,
    out_dir: Path = Path("reports"),
    # СТАРЫЕ ПАРАМЕТРЫ (если есть)
    # НОВЫЕ ПАРАМЕТРЫ:
    max_hist_columns: int = typer.Option(
        5,
        help="Максимальное количество числовых колонок для построения гистограмм."
    ),
    top_k_categories: int = typer.Option(
        10,
        help="Количество топ-значений для вывода у категориальных признаков."
    ),
    title: str = typer.Option(
        "EDA Report",
        help="Заголовок отчета."
    ),
    min_missing_share: float = typer.Option(
        0.3,
        help="Порог доли пропусков для пометки колонки как проблемной (от 0 до 1)."
    ),
    # Можно добавить и другие параметры
    high_cardinality_threshold: int = typer.Option(
        50,
        help="Порог для определения высокой кардинальности категориальных признаков."
    )
):
    """
    Генерация полного отчета EDA.
    
    Примеры:
    eda-cli report data.csv --title "Мой анализ" --top-k-categories 5
    eda-cli report data.csv --min-missing-share 0.2 --max-hist-columns 3
    """
    # Загрузка данных
    df = pd.read_csv(file_path)
    
    # Создание выходной директории
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Вычисление метрик с учетом новых параметров
    quality_flags = core.compute_quality_flags(
        df, 
        min_missing_share=min_missing_share,
        high_cardinality_threshold=high_cardinality_threshold
    )
    
    # Генерация отчета с учетом всех параметров
    viz.create_report(
        df=df,
        quality_flags=quality_flags,
        out_dir=out_dir,
        title=title,
        max_hist_columns=max_hist_columns,
        top_k_categories=top_k_categories,
        min_missing_share=min_missing_share,
        high_cardinality_threshold=high_cardinality_threshold
    )
    
    print(f"✅ Отчет сохранен в {out_dir}/")
    print(f"   Основной файл: {out_dir}/report.md")

if __name__ == "__main__":
    app()
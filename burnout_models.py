import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def main():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 150

    # 1. Завантаження датасету
    df = pd.read_csv("employee_burnout_attrition.csv")
    print(f"Розмір датасету: {df.shape[0]} рядків, {df.shape[1]} колонок")

    # Видаляємо ідентифікатор та обидві цільові/результуючі колонки,
    # щоб уникнути витоку даних (target leakage) через attrition_12m
    X = df.drop(columns=["employee_id", "burnout_score", "attrition_12m"])
    y = df["burnout_score"]

    # 2. Розбиття на навчальну (80%) та тестову (20%) вибірки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Препроцесинг: стандартизація числових ознак та One-Hot для категоріальних
    cat_cols = ["department", "level"]
    num_cols = [col for col in X.columns if col not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                cat_cols,
            ),
        ]
    )

    # 4. Три моделі різної складності
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
        ),
        "MLP (Neural Net)": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1.0,
            batch_size=64,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        ),
    }

    results = []
    predictions = {}
    fitted_pipelines = {}

    # 5. Навчання, замір часу та оцінка метрик
    for name, model in models.items():
        pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred_test = pipe.predict(X_test)
        pred_time_ms = (time.perf_counter() - t1) * 1000

        y_pred_train = pipe.predict(X_train)

        # 5-кратна крос-валідація на train для перевірки стабільності
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)

        predictions[name] = y_pred_test
        fitted_pipelines[name] = pipe

        results.append(
            {
                "Модель": name,
                "Час навчання (с)": fit_time,
                "Час прогнозу (мс)": pred_time_ms,
                "Train MAE": mean_absolute_error(y_train, y_pred_train),
                "Test MAE": mean_absolute_error(y_test, y_pred_test),
                "Train RMSE": root_mean_squared_error(y_train, y_pred_train),
                "Test RMSE": root_mean_squared_error(y_test, y_pred_test),
                "Train R2": r2_score(y_train, y_pred_train),
                "Test R2": r2_score(y_test, y_pred_test),
                "CV R2 (mean)": cv_r2.mean(),
                "CV R2 (std)": cv_r2.std(),
            }
        )

    res_df = pd.DataFrame(results)
    print("\n=== Порівняльна таблиця моделей ===")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    res_df.to_csv("burnout_model_comparison.csv", index=False)

    # 6. Графік 1: Порівняння якості (Train vs Test R2), похибок (MAE/RMSE) та швидкості
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    short_names = ["Linear\nRegression", "Random\nForest", "MLP\n(Neural Net)"]
    x = np.arange(len(short_names))
    w = 0.35

    # (a) Train vs Test R2
    b1 = axes[0].bar(
        x - w / 2, res_df["Train R2"], w, label="Train R²", color="#4C72B0"
    )
    b2 = axes[0].bar(x + w / 2, res_df["Test R2"], w, label="Test R²", color="#DD8452")
    axes[0].set_title("Якість R²: Train vs Test", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names)
    axes[0].set_ylim(0.7, 1.03)
    axes[0].set_ylabel("R²")
    axes[0].legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.006,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # (b) Test MAE & RMSE
    b3 = axes[1].bar(
        x - w / 2, res_df["Test MAE"], w, label="Test MAE", color="#55A868"
    )
    b4 = axes[1].bar(
        x + w / 2, res_df["Test RMSE"], w, label="Test RMSE", color="#C44E52"
    )
    axes[1].set_title("Похибки на тестовій вибірці (менше = краще)", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names)
    axes[1].set_ylim(0, 9.5)
    axes[1].set_ylabel("Бали burnout_score 0-100")
    axes[1].legend()
    for bars in (b3, b4):
        for bar in bars:
            h = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.15,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # (c) Час навчання (лог-шкала для наочності)
    colors_time = ["#4C72B0", "#8172B3", "#CCB974"]
    b5 = axes[2].bar(x, res_df["Час навчання (с)"], width=0.5, color=colors_time)
    axes[2].set_yscale("log")
    axes[2].set_title("Час навчання (с, логарифмічна шкала)", fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_names)
    axes[2].set_ylabel("Секунди (log)")
    axes[2].set_ylim(0.01, max(res_df["Час навчання (с)"]) * 3)
    for bar in b5:
        h = bar.get_height()
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.18,
            f"{h:.3f} с",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("model_comparison_metrics.png", bbox_inches="tight")
    plt.close()

    # 7. Графік 2: Фактичні vs Прогнозовані значення burnout_score
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    scatter_colors = ["#4C72B0", "#8172B3", "#DD8452"]

    for ax, (name, y_pred), col in zip(axes, predictions.items(), scatter_colors):
        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        ax.scatter(y_test, y_pred, alpha=0.28, s=14, color=col, edgecolors="none")
        ax.plot([0, 100], [0, 100], "k--", lw=1.5, label="Ідеальний прогноз y = x")
        ax.set_title(
            f"{name}\nTest R² = {r2:.3f} | RMSE = {rmse:.2f}", fontweight="bold"
        )
        ax.set_xlabel("Фактичний burnout_score")
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.legend(loc="upper left", fontsize=9)

    axes[0].set_ylabel("Прогнозований burnout_score")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", bbox_inches="tight")
    plt.close()

    # 8. Графік 3: Вплив ознак (коефіцієнти Linear Regression та важливість у Random Forest)
    lin_pipe = fitted_pipelines["Linear Regression"]
    feature_names = num_cols + list(
        lin_pipe.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(cat_cols)
    )
    lin_coefs = pd.Series(lin_pipe.named_steps["model"].coef_, index=feature_names)
    top_lin = lin_coefs.reindex(
        lin_coefs.abs().sort_values(ascending=False).index
    ).head(8)[::-1]

    rf_pipe = fitted_pipelines["Random Forest"]
    rf_imp = pd.Series(
        rf_pipe.named_steps["model"].feature_importances_, index=feature_names
    )
    top_rf = rf_imp.sort_values(ascending=False).head(8)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    lin_colors = ["#C44E52" if v > 0 else "#4C72B0" for v in top_lin.values]
    axes[0].barh(top_lin.index, top_lin.values, color=lin_colors)
    axes[0].axvline(0, color="black", lw=0.9)
    axes[0].set_title(
        "Linear Regression: стандартизовані коефіцієнти. Топ 8", fontweight="bold"
    )
    axes[0].set_xlabel(
        "Зміна burnout_score на +1 SD ознаки\n(червоний — підвищує вигорання, синій — знижує)"
    )

    axes[1].barh(top_rf.index, top_rf.values, color="#55A868")
    axes[1].set_title("Random Forest: важливість ознак. Топ 8", fontweight="bold")
    axes[1].set_xlabel("Відносна важливість")

    plt.tight_layout()
    plt.savefig("feature_importance.png", bbox_inches="tight")
    plt.close()

    print(
        "\nЗбережено файли: burnout_model_comparison.csv, model_comparison_metrics.png, actual_vs_predicted.png, feature_importance.png"
    )


if __name__ == "__main__":
    main()

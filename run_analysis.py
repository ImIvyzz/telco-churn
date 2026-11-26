"""Run the full churn analysis end to end and regenerate everything in outputs/.

    python run_analysis.py                     # default scenario: keep 20% of high-risk churners
    python run_analysis.py --save-rate 0.3     # test another retention assumption
"""
import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render charts to files, no window needed
import seaborn as sns  # noqa: E402

from churn import analysis, config, data, impact, model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Telco customer churn: analysis, model and business impact.")
    p.add_argument("--data", type=Path, default=config.DATA_PATH, help="input CSV")
    p.add_argument("--out", type=Path, default=config.OUTPUT_DIR, help="output folder")
    p.add_argument("--save-rate", type=float, default=config.SAVE_RATE,
                   help="assumed share of contacted churners the offer keeps (default 0.20)")
    p.add_argument("--band", choices=config.RISK_LABELS, default=config.TARGET_BAND,
                   help="risk band that receives the retention offer (default High)")
    return p.parse_args()


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid")
    args.out.mkdir(parents=True, exist_ok=True)

    section("1. Data")
    raw = data.load_raw(args.data)
    df = data.clean(raw)
    print(f"{len(df):,} customers, {df.shape[1] - 1} attributes")

    section("2. The business problem")
    f = analysis.headline_figures(df)
    print(f"Churn rate: {f['churn_rate']:.1%}")
    print(f"Monthly revenue lost with churned customers: ${f['lost_monthly_revenue']:,.0f} "
          f"of ${f['monthly_revenue']:,.0f} ({f['lost_monthly_revenue'] / f['monthly_revenue']:.1%})")
    print(f"Churn by contract: month-to-month {f['churn_month_to_month']:.1%}, "
          f"one year {f['churn_one_year']:.1%}, two years {f['churn_two_year']:.1%}")
    print(f"Churn in the first 12 months: {f['churn_first_12_months']:.1%} "
          f"(after month 24: {f['churn_after_24_months']:.1%})")
    analysis.plot_segments(data.add_tenure_group(df), args.out / "churn_by_segment.png")

    tests = analysis.chi_square_tests(df)
    unrelated = tests.loc[~tests["related_to_churn"], "variable"].tolist()
    print(f"Not related to churn (chi-square p >= 0.05): {', '.join(unrelated)}")

    section("3. Model")
    features, target = model.features_and_target(df)
    s = model.split(features, target)
    pipes = model.build_pipelines(features)
    results = model.evaluate(pipes, s)
    results.to_csv(args.out / "model_results.csv")
    print(results.round(3).to_string())
    best_name = model.select_model(results)
    best = pipes[best_name]
    print(f"Selected: {best_name}")
    model.plot_evaluation(pipes, best_name, s, args.out / "model_evaluation.png")
    model.plot_drivers(model.odds_ratios(pipes[model.LOGIT]), args.out / "churn_drivers.png")
    print("\nThreshold trade-off (test set):")
    print(model.threshold_table(best, s).round(3).to_string())
    print("\nTop drivers by permutation importance:")
    print(model.importance(best, s, top=5).round(4).to_string(index=False))

    section("4. Business impact")
    scored = impact.score_customers(best, raw, df, features, target, model.cv_splitter())
    scored.to_csv(args.out / "customer_risk_scores.csv", index=False)
    print(impact.risk_bands(scored).round(3).to_string())
    sc = impact.retention_scenario(scored, args.save_rate, args.band)
    print(f"\nContact the {args.band} band: {sc['customers_contacted']:,} customers "
          f"({sc['share_of_base']:.0%} of the base), reaching {sc['share_of_churners']:.0%} of all churners.")
    print(f"If the offer keeps {args.save_rate:.0%} of them: ${sc['saved_monthly']:,.0f}/month "
          f"(~${sc['saved_yearly']:,.0f}/year) protected, before the cost of the offer.")

    print(f"\nOutputs written to {args.out.resolve()}")


if __name__ == "__main__":
    main()

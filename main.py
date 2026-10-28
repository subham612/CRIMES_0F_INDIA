from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Keep all project paths anchored to this file so the script runs from any cwd.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "Table_1A.1_2016.csv"
OUTPUT_DIR = PROJECT_DIR / "output"
PROCESSED_DIR = PROJECT_DIR / "processed"

print("=" * 60)
print("CRIME ANALYSIS OF INDIA (2014–2016)")
print("=" * 60)

# Load the source table and verify its expected structure before renaming columns.
raw_df = pd.read_csv(DATA_PATH)
if raw_df.shape[1] != 11:
    raise ValueError(f"Expected 11 source columns, found {raw_df.shape[1]}.")
serial_values = pd.to_numeric(raw_df.iloc[:, 0], errors="coerce")
excluded_rows = raw_df.loc[serial_values.isna()]
expected_summary_rows = {"Total (States)", "Total (UTs)", "Total (All-India)"}
excluded_labels = set(excluded_rows.iloc[:, 2].astype(str).str.strip())
if excluded_labels != expected_summary_rows:
    raise ValueError(f"Unexpected non-region rows excluded: {sorted(excluded_labels)}")
print(f"\nExcluded {len(excluded_rows)} non-region summary rows: {', '.join(sorted(excluded_labels))}")
df = raw_df.copy()
df.columns = [
    "serial_no", "category", "state_ut", "crimes_2014", "crimes_2015",
    "crimes_2016", "percentage_share_2016", "incidence_rank_2016",
    "population_lakhs_2016", "crime_rate_2016", "crime_rate_rank_2016",
]
df = df[serial_values.notna()].copy()
df["serial_no"] = pd.to_numeric(df["serial_no"], errors="raise").astype(int)
df["category"] = df["category"].astype(str).str.strip()
df["state_ut"] = df["state_ut"].astype(str).str.strip()
numeric_columns = [
    "crimes_2014", "crimes_2015", "crimes_2016", "percentage_share_2016",
    "incidence_rank_2016", "population_lakhs_2016", "crime_rate_2016",
    "crime_rate_rank_2016",
]
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Validate the cleaned source records before deriving any new fields.
print(f"\nAnalyzed regions: {len(df)}")
print("Missing values by column:")
missing_counts = df.isna().sum()
print(missing_counts.to_string())
if missing_counts.any():
    raise ValueError("Missing values remain after cleaning; review the source data and cleaning steps.")
if df["serial_no"].duplicated().any():
    raise ValueError("Duplicate serial numbers found in region records.")
if df["state_ut"].duplicated().any():
    duplicates = df.loc[df["state_ut"].duplicated(keep=False), "state_ut"].tolist()
    raise ValueError(f"Duplicate State/UT names found: {duplicates}")
if not set(df["category"]).issubset({"State", "UT"}):
    raise ValueError(f"Unexpected region categories: {sorted(set(df['category']))}")
if (df[["crimes_2014", "crimes_2015", "crimes_2016", "population_lakhs_2016", "crime_rate_2016"]] < 0).any().any():
    raise ValueError("Negative count, population, or crime-rate values found.")
print("Data-quality validation passed: no missing values, duplicate region identifiers, or invalid categories.")

# Calculate absolute and percentage changes. A percentage change from a zero baseline is undefined.
def safe_percentage_change(change, baseline):
    baseline_values = baseline.to_numpy(dtype=float)
    return np.divide(
        change.to_numpy(dtype=float) * 100,
        baseline_values,
        out=np.full(len(baseline_values), np.nan, dtype=float),
        where=baseline_values != 0,
    )

df["change_2014_2015"] = df["crimes_2015"] - df["crimes_2014"]
df["percentage_change_2014_2015"] = safe_percentage_change(df["change_2014_2015"], df["crimes_2014"])
df["change_2015_2016"] = df["crimes_2016"] - df["crimes_2015"]
df["percentage_change_2015_2016"] = safe_percentage_change(df["change_2015_2016"], df["crimes_2015"])
df["change_2014_2016"] = df["crimes_2016"] - df["crimes_2014"]
df["percentage_change"] = safe_percentage_change(df["change_2014_2016"], df["crimes_2014"])

print("\nSummary statistics for recorded crime counts:")
print(df[["crimes_2014", "crimes_2015", "crimes_2016"]].describe().round(2))

year_columns = ["crimes_2014", "crimes_2015", "crimes_2016"]
years = np.array([2014, 2015, 2016])
yearly_totals = df[year_columns].sum()
total_crimes = yearly_totals.to_numpy(dtype=float)
print("\nTOTAL RECORDED IPC CRIMES BY YEAR")
for year, total in zip(years, total_crimes):
    print(f"{year}: {total:,.0f}")
print("Year-over-year total changes:")
for previous_year, current_year, previous_total, current_total in zip(years[:-1], years[1:], total_crimes[:-1], total_crimes[1:]):
    absolute_change = current_total - previous_total
    percentage_change = absolute_change / previous_total * 100
    print(f"{previous_year} to {current_year}: {absolute_change:+,.0f} crimes ({percentage_change:+.2f}%)")

print("\nTOP 10 STATES/UTs BY RECORDED CRIME COUNT (2016)")
print(df.nlargest(10, "crimes_2016")[["state_ut", "crimes_2016"]].to_string(index=False))
print("\nTOP 10 STATES/UTs BY CRIME RATE (2016; per 100,000 population)")
print(df.nlargest(10, "crime_rate_2016")[["state_ut", "crime_rate_2016"]].to_string(index=False))
print("\nBOTTOM 10 STATES/UTs BY CRIME RATE (2016; per 100,000 population)")
print(df.nsmallest(10, "crime_rate_2016")[["state_ut", "crime_rate_2016"]].to_string(index=False))
print("\nLARGEST 2014–2016 INCREASES (absolute crime counts)")
print(df.nlargest(10, "change_2014_2016")[["state_ut", "change_2014_2016", "percentage_change"]].round(2).to_string(index=False))
print("\nLARGEST 2014–2016 DECREASES (absolute crime counts)")
print(df.nsmallest(10, "change_2014_2016")[["state_ut", "change_2014_2016", "percentage_change"]].round(2).to_string(index=False))

rate_summary_by_category = df.groupby("category")["crime_rate_2016"].agg(
    region_count="count", mean="mean", median="median", standard_deviation="std", minimum="min", maximum="max"
)
rate_summary_by_category["q1"] = df.groupby("category")["crime_rate_2016"].quantile(.25)
rate_summary_by_category["q3"] = df.groupby("category")["crime_rate_2016"].quantile(.75)
rate_summary_by_category["iqr"] = rate_summary_by_category["q3"] - rate_summary_by_category["q1"]
print("\n2016 CRIME-RATE DISTRIBUTION BY REGION CATEGORY (per 100,000 population)")
print(rate_summary_by_category.round(2).to_string())

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 10})

# Chart 1: Annual totals. Three observed years are shown without interpolation.
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(years, total_crimes, color="#245b8f", marker="o", linewidth=2.6, markersize=7)
ax.set(title="Total Recorded IPC Crimes Across States/UTs (2014–2016)",
       xlabel="Year", ylabel="Recorded IPC crimes (count)")
ax.set_xticks(years)
ax.set_ylim(0, total_crimes.max() * 1.08)
ax.grid(axis="y", alpha=.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "yearly_crime_trend.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Chart 2: Compact share-of-total donut: leading 8 regions plus all remaining regions.
ranked = df.sort_values("crimes_2016", ascending=False)
leading = ranked.head(8).set_index("state_ut")["crimes_2016"]
others = ranked.iloc[8:]["crimes_2016"].sum()
share_values = leading.tolist() + [others]
share_labels = leading.index.tolist() + ["Others (remaining regions)"]
fig, ax = plt.subplots(figsize=(9, 6.5))
wedges, _, _ = ax.pie(
    share_values, startangle=90, counterclock=False,
    autopct=lambda pct: f"{pct:.1f}%" if pct >= 4 else "",
    pctdistance=.78, wedgeprops={"width": .42, "edgecolor": "white", "linewidth": 1.2},
    colors=plt.get_cmap("Blues")(np.linspace(.35, .9, len(share_values))),
)
ax.text(0, .08, "2016", ha="center", va="center", fontsize=15, weight="bold")
ax.text(0, -.16, "recorded crimes", ha="center", va="center", fontsize=9)
ax.set_title("Share of Recorded IPC Crimes by State/UT (2016)", pad=16)
ax.legend(wedges, share_labels, title="Region", loc="center left", bbox_to_anchor=(1.0, .5), frameon=False)
fig.text(.5, .025, "Percentages show each region’s share of the 36-region total; Others combines the remaining 28.", ha="center", fontsize=9)
fig.tight_layout(rect=[0, .06, .78, 1])
fig.savefig(OUTPUT_DIR / "crime_share_2016.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Chart 3: Restore the original top-10 count ranking as a separate bar chart.
top_10 = df.nlargest(10, "crimes_2016").sort_values("crimes_2016")
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top_10["state_ut"], top_10["crimes_2016"], color="#3977a8")
ax.set(title="Top 10 States/UTs by Recorded IPC Crimes (2016)",
       xlabel="Recorded IPC crimes (count)", ylabel="State/UT")
ax.grid(axis="x", alpha=.22)
ax.set_axisbelow(True)
for y, value in enumerate(top_10["crimes_2016"]):
    ax.text(value, y, f"  {value:,.0f}", va="center", ha="left", fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "top_10_crimes_2016.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Chart 4: Distribution of the recorded 2016 crime rate across regions.
rate_data = df["crime_rate_2016"].dropna()
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.boxplot(rate_data, orientation="vertical", patch_artist=True, widths=.42,
           boxprops={"facecolor": "#9ecae1", "edgecolor": "#245b8f"},
           medianprops={"color": "#a63603", "linewidth": 2},
           whiskerprops={"color": "#245b8f"}, capprops={"color": "#245b8f"},
           flierprops={"marker": "o", "markerfacecolor": "#d95f0e", "markeredgecolor": "none", "alpha": .7})
ax.set(title="Distribution of Recorded IPC Crime Rates Across States/UTs (2016)",
       ylabel="Recorded IPC crime rate (per 100,000 population)")
ax.set_xticks([1], [f"{len(rate_data)} states/UTs"])
ax.grid(axis="y", alpha=.25)
fig.text(.5, .015, "Points beyond the whiskers are statistical outliers, not evidence of incorrect data.", ha="center", fontsize=9)
fig.tight_layout(rect=[0, .05, 1, 1])
fig.savefig(OUTPUT_DIR / "highest_crime_rates.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Chart 5: A single diverging chart shows both directions around a clear zero baseline.
extremes = pd.concat([df.nlargest(10, "change_2014_2016"), df.nsmallest(10, "change_2014_2016")])
extremes = extremes.drop_duplicates("state_ut").sort_values("change_2014_2016")
colors = ["#c44e52" if value < 0 else "#4c956c" for value in extremes["change_2014_2016"]]
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(extremes["state_ut"], extremes["change_2014_2016"], color=colors)
ax.axvline(0, color="#333333", linewidth=1)
ax.set(title="Largest State/UT Changes in Recorded IPC Crimes (2014–2016)",
       xlabel="Change in recorded IPC crimes (2016 minus 2014; count)", ylabel="State/UT")
ax.grid(axis="x", alpha=.22)
ax.set_axisbelow(True)
span = max(abs(extremes["change_2014_2016"].min()), abs(extremes["change_2014_2016"].max()))
for y, value in enumerate(extremes["change_2014_2016"]):
    ax.text(value + (span * .015 if value >= 0 else -span * .015), y, f"{value:+,.0f}",
            va="center", ha="left" if value >= 0 else "right", fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "largest_crime_increases.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Chart 6: Population and rate association; correlation is descriptive only.
valid = df[["population_lakhs_2016", "crime_rate_2016"]].dropna()
correlation = valid["population_lakhs_2016"].corr(valid["crime_rate_2016"])
print(f"\nPopulation/crime-rate Pearson correlation: {correlation:.3f}")
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(valid["population_lakhs_2016"], valid["crime_rate_2016"],
           s=52, alpha=.8, color="#3977a8", edgecolor="white", linewidth=.5)
ax.set(title="Population vs Recorded IPC Crime Rate by State/UT (2016)",
       xlabel="Projected population (lakhs)", ylabel="Recorded IPC crime rate (per 100,000 population)")
ax.grid(alpha=.22)
ax.text(.03, .97, f"Pearson r = {correlation:.2f}\nAssociation does not imply causation",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "alpha": .85, "edgecolor": "#cccccc"})
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "population_vs_crime_rate.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Chart 7: State/UT rate ranking makes individual comparisons visible alongside the box plot.
rate_ranking = df.dropna(subset=["crime_rate_2016"]).sort_values("crime_rate_2016")
fig, ax = plt.subplots(figsize=(10, 12))
ax.hlines(rate_ranking["state_ut"], 0, rate_ranking["crime_rate_2016"], color="#9ecae1", linewidth=1.5)
ax.scatter(rate_ranking["crime_rate_2016"], rate_ranking["state_ut"], color="#245b8f", s=28, zorder=3)
ax.set(title="Recorded IPC Crime-Rate Ranking by State/UT (2016)",
       xlabel="Recorded IPC crime rate (per 100,000 population)", ylabel="State/UT")
ax.grid(axis="x", alpha=.22)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "crime_rate_rankings_2016.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Remove obsolete duplicate standalone charts now represented by the combined chart and box plot.
for obsolete_chart in ("largest_crime_decreases.png", "lowest_crime_rates.png"):
    obsolete_path = OUTPUT_DIR / obsolete_chart
    if obsolete_path.exists():
        obsolete_path.unlink()
# Export cleaned data and existing analysis tables.
cleaned_path = PROCESSED_DIR / "cleaned_crime_data.csv"
df.to_csv(cleaned_path, index=False)
data_processed_dir = PROJECT_DIR / "data" / "processed"
data_processed_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(data_processed_dir / "cleaned_crime_data.csv", index=False)
analysis_tables = {
    "top_10_crimes_2016.csv": df.nlargest(10, "crimes_2016"),
    "largest_increases_2014_2016.csv": df.nlargest(10, "change_2014_2016"),
    "largest_decreases_2014_2016.csv": df.nsmallest(10, "change_2014_2016"),
    "highest_crime_rates_2016.csv": df.nlargest(10, "crime_rate_2016"),
    "lowest_crime_rates_2016.csv": df.nsmallest(10, "crime_rate_2016"),
}
for filename, table in analysis_tables.items():
    table.to_csv(OUTPUT_DIR / filename, index=False)
yearly_summary = pd.DataFrame({"year": years, "total_recorded_crimes": total_crimes})
yearly_summary["change_from_previous_year"] = [np.nan, total_crimes[1] - total_crimes[0], total_crimes[2] - total_crimes[1]]
yearly_summary["percentage_change_from_previous_year"] = [
    np.nan,
    (total_crimes[1] / total_crimes[0] - 1) * 100,
    (total_crimes[2] / total_crimes[1] - 1) * 100,
]
yearly_summary.to_csv(OUTPUT_DIR / "yearly_crime_totals.csv", index=False)
rate_summary_by_category.to_csv(OUTPUT_DIR / "crime_rate_summary_by_category.csv")

# Flag Tukey 1.5-IQR observations for review; these are not evidence of bad data.
rate_q1 = df["crime_rate_2016"].quantile(.25)
rate_q3 = df["crime_rate_2016"].quantile(.75)
rate_iqr = rate_q3 - rate_q1
rate_lower_fence = rate_q1 - 1.5 * rate_iqr
rate_upper_fence = rate_q3 + 1.5 * rate_iqr
rate_outliers = df.loc[
    (df["crime_rate_2016"] < rate_lower_fence) | (df["crime_rate_2016"] > rate_upper_fence),
    ["state_ut", "category", "crime_rate_2016"],
].sort_values("crime_rate_2016", ascending=False)
rate_outliers.to_csv(OUTPUT_DIR / "crime_rate_iqr_observations_2016.csv", index=False)

# Written findings report retains the project's established findings and limitations.
highest_count_region = df.loc[df["crimes_2016"].idxmax()]
highest_rate_region = df.loc[df["crime_rate_2016"].idxmax()]
largest_increase_region = df.loc[df["change_2014_2016"].idxmax()]
largest_decrease_region = df.loc[df["change_2014_2016"].idxmin()]
with (OUTPUT_DIR / "project_findings.txt").open("w", encoding="utf-8") as report:
    report.write("CRIME ANALYSIS OF INDIA (2014–2016)\n" + "=" * 45 + "\n\n")
    report.write(f"Number of analyzed regions: {len(df)}\n")
    report.write(f"States: {(df['category'] == 'State').sum()}\n")
    report.write(f"Union Territories: {(df['category'] == 'UT').sum()}\n\n")
    report.write("TOTAL RECORDED CRIMES BY YEAR\n")
    for position, (year, total) in enumerate(zip(years, total_crimes)):
        if position == 0:
            report.write(f"{year}: {total:,.0f} (baseline year)\n")
        else:
            absolute = total - total_crimes[position - 1]
            percent = absolute / total_crimes[position - 1] * 100
            report.write(f"{year}: {total:,.0f} ({absolute:+,.0f}; {percent:+.2f}% vs {years[position - 1]})\n")
    report.write("\n2016 CRIME-RATE DISTRIBUTION BY REGION CATEGORY (per 100,000 population)\n")
    for category, row in rate_summary_by_category.iterrows():
        report.write(f"{category}: n={int(row['region_count'])}, median={row['median']:.1f}, IQR={row['q1']:.1f}–{row['q3']:.1f}\n")
    report.write("\nKEY FINDINGS\n")
    report.write(f"Highest 2016 crime count: {highest_count_region['state_ut']} ({highest_count_region['crimes_2016']:,.0f})\n")
    report.write(f"Highest 2016 recorded crime rate: {highest_rate_region['state_ut']} ({highest_rate_region['crime_rate_2016']})\n")
    report.write(f"Largest increase, 2014–2016: {largest_increase_region['state_ut']} (+{largest_increase_region['change_2014_2016']:,.0f}; {largest_increase_region['percentage_change']:+.2f}%)\n")
    report.write(f"Largest decrease, 2014–2016: {largest_decrease_region['state_ut']} ({largest_decrease_region['change_2014_2016']:,.0f}; {largest_decrease_region['percentage_change']:+.2f}%)\n")
    report.write(f"Population/crime-rate Pearson correlation: {correlation:.3f}; this is a very weak linear association in these 36 regions, not evidence of causation.\n")
    if len(rate_outliers):
        flagged = ", ".join(f"{row.state_ut} ({row.crime_rate_2016:.1f})" for row in rate_outliers.itertuples())
        report.write(f"2016 rate values beyond the 1.5-IQR fences ({rate_lower_fence:.1f}–{rate_upper_fence:.1f}): {flagged}. These are statistical flags for context, not evidence of incorrect data.\n")
    else:
        report.write("No 2016 crime rates fall beyond the 1.5-IQR fences.\n")
    report.write("\n")
    report.write("METHODOLOGY\nPandas was used for loading, cleaning, summarizing, and exporting. NumPy was used for safe percentage-change calculations. Matplotlib was used for visualizations.\n\n")
    report.write("LIMITATIONS\nRecorded IPC crimes do not necessarily represent all crimes committed. Reporting practices, population differences, and the small number of years affect interpretation. State/UT group summaries describe distributions of regions and are not population-weighted. Correlation does not establish causation. Crime-rate outliers are statistical observations, not proof of incorrect data.\n")

# Preserve the project's verified key-result checks.
assert len(df) == 36, "Unexpected number of analyzed regions!"
assert df["state_ut"].nunique() == 36, "Duplicate region names found!"
expected_totals = {"crimes_2014": 2851563, "crimes_2015": 2949400, "crimes_2016": 2975711}
for column, expected in expected_totals.items():
    actual = int(df[column].sum())
    print(f"{column}: {actual:,}")
    assert actual == expected, f"Mismatch in {column}: expected {expected}, got {actual}"
assert highest_count_region["state_ut"] == "Uttar Pradesh" and int(highest_count_region["crimes_2016"]) == 282171
assert largest_increase_region["state_ut"] == "Delhi" and int(largest_increase_region["change_2014_2016"]) == 53865
assert largest_decrease_region["state_ut"] == "Rajasthan" and int(largest_decrease_region["change_2014_2016"]) == -30020
print("\nAll existing result checks passed.")

# Rebuild the dashboard from the revised chart images, replacing its prior collection.
graph_files = [
    ("Yearly total trend", "yearly_crime_trend.png"),
    ("Top 10 recorded crime counts", "top_10_crimes_2016.png"),
    ("2016 share of total", "crime_share_2016.png"),
    ("2016 crime-rate distribution", "highest_crime_rates.png"),
    ("2016 crime-rate rankings", "crime_rate_rankings_2016.png"),
    ("2014–2016 increases and decreases", "largest_crime_increases.png"),
    ("Population vs crime rate", "population_vs_crime_rate.png"),
]
fig, axes = plt.subplots(4, 2, figsize=(20, 26))
for ax, (title, filename) in zip(axes.flat, graph_files):
    ax.imshow(plt.imread(OUTPUT_DIR / filename))
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
for ax in axes.flat[len(graph_files):]:
    ax.axis("off")
fig.suptitle("Crimes of India: Analysis Dashboard (2014–2016)", fontsize=20, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .975])
fig.savefig(OUTPUT_DIR / "all_graphs_dashboard.png", dpi=200, bbox_inches="tight")
print("Opening the dashboard window. Close it to finish the program.")
plt.show(block=True)
plt.close(fig)

print(f"\nCleaned data saved: {cleaned_path}")
print(f"Revised charts and dashboard saved under: {OUTPUT_DIR}")
print("CRIME ANALYSIS COMPLETED SUCCESSFULLY")





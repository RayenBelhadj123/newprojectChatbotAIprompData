from __future__ import annotations

from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "housing_intelligence_25_slides.pdf"

W, H = 1600, 900
BG = (245, 248, 251)
CARD = (255, 255, 255)
NAVY = (18, 37, 63)
DEEP = (10, 28, 52)
TEAL = (0, 138, 150)
ORANGE = (223, 107, 34)
MUTED = (78, 94, 116)
LINE = (218, 226, 236)
PALE_TEAL = (222, 243, 244)
PALE_ORANGE = (255, 236, 220)
WHITE = (255, 255, 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


F_TAG = font(20, True)
F_TITLE = font(44, True)
F_SUB = font(30, True)
F_BULLET = font(27)
F_SMALL = font(22)
F_METRIC = font(58, True)
F_METRIC_LABEL = font(24, True)
F_NUM = font(34, True)


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], max_width: int, fnt, fill, line_gap: int = 8) -> int:
    x, y = xy
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=fnt)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def draw_bullets(draw: ImageDraw.ImageDraw, bullets: list[str], x: int, y: int, max_width: int, fnt=F_BULLET, fill=MUTED) -> None:
    for item in bullets:
        draw.text((x, y), "•", font=fnt, fill=TEAL)
        y = draw_wrapped(draw, item, (x + 34, y), max_width - 34, fnt, fill, 7) + 14


def draw_card(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill=CARD, outline=LINE, radius: int = 28) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=2)


def draw_small_label(draw: ImageDraw.ImageDraw, text: str, x: int, y: int) -> None:
    draw.rounded_rectangle((x, y, x + 225, y + 38), radius=18, fill=PALE_TEAL)
    draw.text((x + 18, y + 8), text.upper(), font=F_TAG, fill=TEAL)


def draw_title_header(draw: ImageDraw.ImageDraw, title: str, section: str = "FINAL DEFENSE") -> None:
    draw_small_label(draw, section, 58, 36)
    draw_wrapped(draw, title, (58, 88), 1180, F_TITLE, NAVY, 4)
    draw.rectangle((58, 154, 1540, 160), fill=TEAL)


def fit_image(path: Path, box: tuple[int, int, int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    x, y, w, h = box
    img.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), WHITE)
    ox = (w - img.width) // 2
    oy = (h - img.height) // 2
    canvas.paste(img, (ox, oy))
    return canvas


def base_slide(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    slide = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(slide)
    draw_title_header(draw, title)
    return slide, draw


def title_slide(title: str, bullets: list[str]) -> Image.Image:
    slide = Image.new("RGB", (W, H), DEEP)
    draw = ImageDraw.Draw(slide)
    draw.rounded_rectangle((1060, 0, 1600, 900), radius=0, fill=(8, 83, 100))
    draw.rounded_rectangle((1120, 90, 1510, 810), radius=42, fill=(15, 113, 128), outline=(40, 160, 171), width=3)
    draw.text((78, 72), "IT BUSINESS SCHOOL · TUNISIA", font=F_TAG, fill=(190, 239, 239))
    draw.text((78, 112), "Data Mining & Machine Learning · 2025-2026", font=F_SMALL, fill=(221, 232, 240))
    draw_wrapped(draw, title, (78, 215), 900, font(68, True), WHITE, 8)
    draw_bullets(draw, bullets, 100, 460, 820, font(32), (221, 232, 240))
    metric_specs = [("242", "Observations"), ("27", "Features"), ("20yr", "Period")]
    for idx, (value, label) in enumerate(metric_specs):
        x = 1148
        y = 160 + idx * 190
        draw.text((x, y), value, font=font(54, True), fill=WHITE)
        draw.text((x, y + 70), label, font=F_SMALL, fill=(221, 232, 240))
    draw.text((78, 805), "Ayoub Naouech · Rayen Belhadj", font=F_SMALL, fill=(221, 232, 240))
    return slide


def bullets_slide(title: str, bullets: list[str]) -> Image.Image:
    slide, draw = base_slide(title)
    draw_card(draw, (85, 210, 1515, 790))
    draw_bullets(draw, bullets, 135, 255, 1300)
    return slide


def agenda_slide(title: str, items: list[tuple[str, str]]) -> Image.Image:
    slide, draw = base_slide(title)
    y = 205
    for idx, (head, sub) in enumerate(items, start=1):
        x = 100 if idx <= 5 else 850
        yy = y + ((idx - 1) % 5) * 118
        draw_card(draw, (x, yy, x + 620, yy + 86))
        draw.rounded_rectangle((x + 22, yy + 18, x + 82, yy + 78), radius=16, fill=TEAL if idx % 2 else ORANGE)
        draw.text((x + 37, yy + 29), f"{idx:02d}", font=font(20, True), fill=WHITE)
        draw.text((x + 105, yy + 16), head, font=font(24, True), fill=NAVY)
        draw_wrapped(draw, sub, (x + 105, yy + 48), 460, F_SMALL, MUTED, 2)
    return slide


def metrics_slide(title: str, bullets: list[str]) -> Image.Image:
    slide, draw = base_slide(title)
    metrics = [("242", "Monthly observations", TEAL), ("27", "Dataset columns", ORANGE), ("2004-2024", "Historical period", NAVY)]
    for idx, (value, label, color) in enumerate(metrics):
        x = 110 + idx * 495
        draw_card(draw, (x - 18, 195, x + 398, 450))
        draw.rounded_rectangle((x, 220, x + 380, 370), radius=28, fill=color)
        bbox = draw.textbbox((0, 0), value, font=F_METRIC)
        draw.text((x + 190 - (bbox[2] - bbox[0]) / 2, 258), value, font=F_METRIC, fill=WHITE)
        bbox = draw.textbbox((0, 0), label, font=F_METRIC_LABEL)
        draw.text((x + 190 - (bbox[2] - bbox[0]) / 2, 398), label, font=F_METRIC_LABEL, fill=NAVY)
    draw_card(draw, (110, 505, 1490, 790))
    draw_bullets(draw, bullets, 145, 540, 1260)
    return slide


def image_full_slide(title: str, image: Path) -> Image.Image:
    slide, draw = base_slide(title)
    x, y, w, h = 100, 175, 1400, 650
    draw_card(draw, (x - 14, y - 14, x + w + 14, y + h + 14), radius=28)
    slide.paste(fit_image(image, (x, y, w, h)), (x, y))
    return slide


def image_left_slide(title: str, image: Path, bullets: list[str]) -> Image.Image:
    slide, draw = base_slide(title)
    x, y, w, h = 80, 180, 850, 620
    draw_card(draw, (x - 14, y - 14, x + w + 14, y + h + 14), radius=28)
    slide.paste(fit_image(image, (x, y, w, h)), (x, y))
    draw_card(draw, (975, 180, 1520, 800), fill=WHITE, radius=28)
    draw_bullets(draw, bullets, 1010, 225, 470, font(23))
    return slide


def model_cards_slide(title: str) -> Image.Image:
    slide, draw = base_slide(title)
    cards = [
        ("Linear Regression", "Best generated test result", "R2 0.999", TEAL),
        ("Ridge Regression", "Stable interpretable baseline", "R2 0.984", ORANGE),
        ("Tree Models", "Useful but weak extrapolation", "Gap risk", NAVY),
    ]
    for idx, (name, desc, metric, color) in enumerate(cards):
        x = 120 + idx * 485
        draw_card(draw, (x, 230, x + 400, 660))
        draw.rounded_rectangle((x + 30, 265, x + 370, 360), radius=22, fill=color)
        draw.text((x + 55, 292), metric, font=font(34, True), fill=WHITE)
        draw.text((x + 30, 405), name, font=font(28, True), fill=NAVY)
        draw_wrapped(draw, desc, (x + 30, 455), 330, font(23), MUTED, 6)
    draw.text((120, 725), "Key message: model evaluation must be chronological and diagnostic, not based on popularity.", font=font(24, True), fill=TEAL)
    return slide


def main() -> None:
    img = {
        "home": ROOT / "app_screenshots" / "app_home.png",
        "modeling": ROOT / "app_screenshots" / "app_modeling.png",
        "evaluation_app": ROOT / "app_screenshots" / "app_evaluation.png",
        "olap_app": ROOT / "app_screenshots" / "app_olap.png",
        "paper_app": ROOT / "app_screenshots" / "app_paper_review.png",
        "governance": ROOT / "app_screenshots" / "app_governance.png",
        "pipeline": ROOT / "figures" / "project_pipeline.png",
        "corr": ROOT / "figures" / "correlation_results.png",
        "eval": ROOT / "figures" / "model_evaluation_results.png",
        "diag": ROOT / "figures" / "fit_diagnostics_results.png",
        "pred": ROOT / "figures" / "actual_vs_predicted_results.png",
        "olap": ROOT / "figures" / "olap_period_results.png",
    }

    slides = [
        title_slide("US Housing Intelligence Platform", ["Final report and Streamlit app defense", "Data Mining and Machine Learning project", "Ayoub Naouech and Rayen Belhadj"]),
        agenda_slide("Presentation Outline", [
            ("Project Motivation & Context", "Why housing markets matter"),
            ("Dataset & Features", "20 years, 242 observations, 27 variables"),
            ("CRISP-DM Methodology", "Full 6-phase data science workflow"),
            ("Streamlit Intelligence App", "Interactive dashboard screenshots"),
            ("Model Evaluation", "Metrics, diagnostics, and prediction proof"),
            ("OLAP & Paper Review", "Segmentation and theory validation"),
            ("Business Impact", "Stakeholders and decision support"),
            ("Limitations & Future Work", "Next development steps"),
            ("Final Conclusion", "Main defense message"),
        ]),
        bullets_slide("Project Context", ["The U.S. housing market connects household wealth, credit conditions, inflation, construction, and employment.", "Housing prices react to demand, supply, income, mortgage rates, expectations, and timing.", "The project converts raw housing and macroeconomic data into an interactive intelligence platform."]),
        bullets_slide("Problem Statement", ["Main question: how can data mining and machine learning analyze and predict U.S. housing-market trends?", "Analytical challenge: housing variables interact in non-linear and time-dependent ways.", "Communication challenge: model results must be understandable for non-technical users."]),
        bullets_slide("Project Objectives", ["Analyze historical evolution of the Home Price Index.", "Identify variables associated with U.S. housing-price behavior.", "Train and compare machine learning models using reproducible metrics.", "Build a Streamlit app for visualization, modeling, OLAP, export, and interpretation."]),
        metrics_slide("Dataset Overview", ["Target variable: Home_Price_Index.", "Main variables: mortgage rate, interest rate, unemployment, inflation, permits, income, population, and sentiment.", "The dataset supports both prediction and interpretation."]),
        bullets_slide("Data Preparation", ["Parse and sort the date column chronologically.", "Convert numeric columns into model-ready values.", "Check missing values and duplicate rows before modeling.", "Create administration labels, lag features, rolling features, ratios, and percentage changes."]),
        bullets_slide("CRISP-DM Methodology", ["Business understanding defines the housing problem.", "Data understanding checks quality and temporal structure.", "Data preparation builds usable analytical features.", "Modeling and evaluation compare algorithms.", "Deployment is delivered through Streamlit."]),
        image_left_slide("System Architecture", img["pipeline"], ["Data layer: loading, cleaning, validation, and features.", "Analysis layer: EDA, correlations, OLAP, and period comparison.", "Modeling layer: regression, classification, forecasting, and diagnostics.", "Interpretation layer: paper review, conclusion, and export."]),
        image_full_slide("Streamlit App: Home And Workflow", img["home"]),
        image_full_slide("Streamlit App: ML Lab", img["modeling"]),
        image_full_slide("Streamlit App: Evaluation", img["evaluation_app"]),
        image_full_slide("Streamlit App: OLAP And Export", img["olap_app"]),
        image_left_slide("Streamlit App: Paper Review", img["paper_app"], ["Compares dashboard evidence with housing theory.", "Explains rates, supply, demand, timing, and affordability together.", "Makes the project academically stronger than a purely technical dashboard."]),
        image_full_slide("Streamlit App: Governance", img["governance"]),
        image_left_slide("Correlation Results", img["corr"], ["Strong relationships appear around lagged HPI, smoothed HPI, and Median_Income.", "Correlation supports interpretation but does not prove causality.", "Housing prices are persistent and linked to broader economic context."]),
        image_left_slide("Model Evaluation Results", img["eval"], ["Compared models include Linear Regression, Ridge, Random Forest, Gradient Boosting, Decision Tree, KNN, and SVR.", "Best generated metrics: R2 = 0.999, RMSE = 0.797, MAE = 0.661."]),
        model_cards_slide("Regression Metrics Summary"),
        image_left_slide("Fit Diagnostics", img["diag"], ["Diagnostics compare train and test performance.", "Large train-test gaps indicate weak generalization or overfitting risk.", "Tree models struggle when the final period moves beyond the training range."]),
        image_left_slide("Actual Versus Predicted", img["pred"], ["The best model follows the chronological test period closely.", "Lagged and smoothed price features explain the strong predictive result.", "The visual is easier to defend than metrics alone."]),
        image_left_slide("OLAP Period Results", img["olap"], ["Biden period has the highest average Home Price Index.", "Biden period also has the highest average mortgage-rate pressure.", "Prices and affordability pressure can rise together when supply and timing matter."]),
        bullets_slide("Paper Review Message", ["Housing prices are shaped by demand, financing conditions, supply, construction, income, and timing.", "Higher mortgage rates reduce affordability, but prices may react slowly when supply is constrained.", "The strongest conclusion is multi-factor interpretation rather than one-variable explanation."]),
        bullets_slide("Business Impact", ["Analytical impact: users understand which indicators move with housing prices.", "Decision-support impact: users compare models, periods, OLAP segments, and scenarios.", "Educational impact: the project demonstrates EDA, ML, forecasting, OLAP, paper review, and governance."]),
        bullets_slide("Limitations And Future Work", ["National-level data hides regional differences between states and cities.", "Forecasts are educational and should not be treated as financial advice.", "Future work: regional datasets, live APIs, SHAP explainability, better chatbot grounding, monitoring, and authentication."]),
        bullets_slide("Final Conclusion", ["The report, poster, and app now tell one consistent story.", "U.S. housing prices require theory, correlations, model evaluation, diagnostics, OLAP, and market context.", "The Streamlit dashboard is a housing intelligence platform, not only a collection of charts."]),
    ]
    slides[0].save(OUT, save_all=True, append_images=slides[1:], resolution=150.0)
    print(f"Created {OUT} with {len(slides)} pages")


if __name__ == "__main__":
    main()

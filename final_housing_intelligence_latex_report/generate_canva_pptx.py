from __future__ import annotations

import html
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "canva_housing_intelligence_presentation.pptx"
WORK = ROOT / "_pptx_build"

SLIDE_W = 13_333_333
SLIDE_H = 7_500_000

BG = "F7FAFC"
NAVY = "17324D"
TEAL = "087F8C"
ORANGE = "C35F1D"
MUTED = "53657A"
LINE = "D8E1EA"
WHITE = "FFFFFF"


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def emu(inches: float) -> int:
    return int(inches * 914400)


def box(x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
    return emu(x), emu(y), emu(w), emu(h)


def shape_xml(
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    font_size: int = 20,
    color: str = NAVY,
    bold: bool = False,
    fill: str | None = None,
    line: str | None = None,
    align: str = "l",
    radius: bool = False,
) -> str:
    bx, by, bw, bh = box(x, y, w, h)
    fill_xml = "<a:noFill/>" if fill is None else f"<a:solidFill><a:srgbClr val=\"{fill}\"/></a:solidFill>"
    line_xml = "<a:ln><a:noFill/></a:ln>" if line is None else f"<a:ln w=\"9525\"><a:solidFill><a:srgbClr val=\"{line}\"/></a:solidFill></a:ln>"
    bold_attr = " b=\"1\"" if bold else ""
    shape_type = "roundRect" if radius else "rect"
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="1" name="Text"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{bx}" y="{by}"/><a:ext cx="{bw}" cy="{bh}"/></a:xfrm><a:prstGeom prst="{shape_type}"><a:avLst/></a:prstGeom>{fill_xml}{line_xml}</p:spPr>
  <p:txBody>
    <a:bodyPr wrap="square" anchor="mid"/>
    <a:lstStyle/>
    <a:p><a:pPr algn="{align}"/><a:r><a:rPr lang="en-US" sz="{font_size * 100}"{bold_attr}><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="Aptos"/></a:rPr><a:t>{esc(text)}</a:t></a:r></a:p>
  </p:txBody>
</p:sp>"""


def bullets_xml(items: list[str], x: float, y: float, w: float, h: float, font_size: int = 17) -> str:
    bx, by, bw, bh = box(x, y, w, h)
    paragraphs = []
    for item in items:
        paragraphs.append(
            f"""
    <a:p>
      <a:pPr marL="342900" indent="-228600"><a:buChar char="•"/></a:pPr>
      <a:r><a:rPr lang="en-US" sz="{font_size * 100}"><a:solidFill><a:srgbClr val="{MUTED}"/></a:solidFill><a:latin typeface="Aptos"/></a:rPr><a:t>{esc(item)}</a:t></a:r>
    </a:p>"""
        )
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="2" name="Bullets"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{bx}" y="{by}"/><a:ext cx="{bw}" cy="{bh}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
  <p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{''.join(paragraphs)}</p:txBody>
</p:sp>"""


def image_xml(rid: str, x: float, y: float, w: float, h: float) -> str:
    bx, by, bw, bh = box(x, y, w, h)
    return f"""
<p:pic>
  <p:nvPicPr><p:cNvPr id="3" name="Picture"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
  <p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
  <p:spPr><a:xfrm><a:off x="{bx}" y="{by}"/><a:ext cx="{bw}" cy="{bh}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:ln w="9525"><a:solidFill><a:srgbClr val="{LINE}"/></a:solidFill></a:ln></p:spPr>
</p:pic>"""


def slide_xml(title: str, bullets: list[str] | None = None, image: str | None = None, layout: str = "bullets") -> str:
    shapes = [
        shape_xml("", 0, 0, 13.333, 7.5, fill=BG),
        shape_xml("US Housing Intelligence", 0.55, 0.28, 3.5, 0.32, 10, TEAL, True),
        shape_xml(title, 0.55, 0.62, 12.1, 0.7, 24, NAVY, True),
        shape_xml("", 0.55, 1.36, 12.1, 0.03, fill=TEAL),
    ]
    if layout == "title":
        shapes = [
            shape_xml("", 0, 0, 13.333, 7.5, fill=NAVY),
            shape_xml("US Housing Intelligence", 0.65, 0.6, 5.3, 0.45, 14, "BDEFEF", True),
            shape_xml(title, 0.65, 1.35, 11.2, 1.4, 34, WHITE, True),
            bullets_xml(bullets or [], 0.78, 3.05, 9.8, 2.4, 18),
            shape_xml("Data Mining and Machine Learning | Final Defense", 0.65, 6.45, 8.4, 0.35, 12, "DDE8F0"),
        ]
    elif layout == "image_full" and image:
        shapes.append(image_xml("rId2", 1.0, 1.62, 11.35, 5.35))
    elif layout == "image_left" and image:
        shapes.append(image_xml("rId2", 0.65, 1.65, 7.05, 5.35))
        shapes.append(bullets_xml(bullets or [], 8.0, 1.75, 4.75, 5.1, 16))
    elif layout == "image_right" and image:
        shapes.append(bullets_xml(bullets or [], 0.75, 1.75, 4.75, 5.1, 16))
        shapes.append(image_xml("rId2", 5.75, 1.65, 6.95, 5.35))
    elif layout == "metrics":
        shapes.extend(
            [
                shape_xml("242", 0.9, 1.9, 3.2, 1.3, 38, WHITE, True, TEAL, TEAL, "ctr", True),
                shape_xml("Monthly observations", 0.9, 3.15, 3.2, 0.45, 13, NAVY, True),
                shape_xml("27", 5.05, 1.9, 3.2, 1.3, 38, WHITE, True, ORANGE, ORANGE, "ctr", True),
                shape_xml("Dataset columns", 5.05, 3.15, 3.2, 0.45, 13, NAVY, True),
                shape_xml("2004-2024", 9.2, 1.9, 3.2, 1.3, 30, WHITE, True, NAVY, NAVY, "ctr", True),
                shape_xml("Historical period", 9.2, 3.15, 3.2, 0.45, 13, NAVY, True),
                bullets_xml(bullets or [], 1.1, 4.25, 11.0, 2.2, 16),
            ]
        )
    else:
        shapes.append(bullets_xml(bullets or [], 0.95, 1.75, 11.2, 5.25, 18))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    {''.join(shapes)}
  </p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def rels_xml(has_image: bool, image_target: str | None = None) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
    ]
    if has_image and image_target:
        rels.append(
            f'<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{image_target}"/>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>"""


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def package_parts(slides: list[dict[str, object]], media: dict[str, Path]) -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / "ppt" / "slides" / "_rels").mkdir(parents=True)
    (WORK / "ppt" / "slideLayouts" / "_rels").mkdir(parents=True)
    (WORK / "ppt" / "slideMasters" / "_rels").mkdir(parents=True)
    (WORK / "ppt" / "theme").mkdir(parents=True)
    (WORK / "_rels").mkdir()
    (WORK / "docProps").mkdir()
    (WORK / "ppt" / "media").mkdir()

    for media_name, source in media.items():
        shutil.copyfile(source, WORK / "ppt" / "media" / media_name)

    slide_ids = []
    for i, slide in enumerate(slides, start=1):
        image_name = slide.get("image_name")
        write_file(WORK / "ppt" / "slides" / f"slide{i}.xml", slide_xml(**slide["content"]))
        write_file(WORK / "ppt" / "slides" / "_rels" / f"slide{i}.xml.rels", rels_xml(bool(image_name), image_name if isinstance(image_name, str) else None))
        slide_ids.append(f'<p:sldId id="{255 + i}" r:id="rId{i + 5}"/>')

    pres_rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>',
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/printerSettings" Target="printerSettings/printerSettings1.bin"/>',
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps" Target="presProps.xml"/>',
        '<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps" Target="viewProps.xml"/>',
        '<Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>',
    ]
    for i in range(1, len(slides) + 1):
        pres_rels.append(f'<Relationship Id="rId{i + 5}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>')

    overrides = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Default Extension="png" ContentType="image/png"/>',
        '<Default Extension="bin" ContentType="application/vnd.openxmlformats-officedocument.presentationml.printerSettings"/>',
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>',
        '<Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    overrides += [
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, len(slides) + 1)
    ]

    write_file(WORK / "[Content_Types].xml", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">{"".join(overrides)}</Types>')
    write_file(WORK / "_rels" / ".rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>')
    write_file(WORK / "ppt" / "_rels" / "presentation.xml.rels", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{"".join(pres_rels)}</Relationships>')
    write_file(WORK / "ppt" / "presentation.xml", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" saveSubsetFonts="1">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{"".join(slide_ids)}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>''')
    write_file(WORK / "ppt" / "slideLayouts" / "slideLayout1.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')
    write_file(WORK / "ppt" / "slideLayouts" / "_rels" / "slideLayout1.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>')
    write_file(WORK / "ppt" / "slideMasters" / "slideMaster1.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst></p:sldMaster>')
    write_file(WORK / "ppt" / "slideMasters" / "_rels" / "slideMaster1.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>')
    write_file(WORK / "ppt" / "theme" / "theme1.xml", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="HousingIQ"><a:themeElements><a:clrScheme name="HousingIQ"><a:dk1><a:srgbClr val="{NAVY}"/></a:dk1><a:lt1><a:srgbClr val="{WHITE}"/></a:lt1><a:dk2><a:srgbClr val="{MUTED}"/></a:dk2><a:lt2><a:srgbClr val="{BG}"/></a:lt2><a:accent1><a:srgbClr val="{TEAL}"/></a:accent1><a:accent2><a:srgbClr val="{ORANGE}"/></a:accent2><a:accent3><a:srgbClr val="2563EB"/></a:accent3><a:accent4><a:srgbClr val="7C3AED"/></a:accent4><a:accent5><a:srgbClr val="10B981"/></a:accent5><a:accent6><a:srgbClr val="F59E0B"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink></a:clrScheme><a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme><a:fmtScheme name="Default"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>')
    write_file(WORK / "ppt" / "presProps.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentationPr xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>')
    write_file(WORK / "ppt" / "viewProps.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:viewPr xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>')
    write_file(WORK / "docProps" / "core.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>US Housing Intelligence Presentation</dc:title><dc:creator>Codex</dc:creator></cp:coreProperties>')
    write_file(WORK / "docProps" / "app.xml", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Codex</Application><Slides>{len(slides)}</Slides></Properties>')
    (WORK / "ppt" / "printerSettings").mkdir()
    (WORK / "ppt" / "printerSettings" / "printerSettings1.bin").write_bytes(b"")

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in WORK.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(WORK).as_posix())


def main() -> None:
    media_sources = {
        "app_home.png": ROOT / "app_screenshots" / "app_home.png",
        "app_modeling.png": ROOT / "app_screenshots" / "app_modeling.png",
        "app_evaluation.png": ROOT / "app_screenshots" / "app_evaluation.png",
        "app_olap.png": ROOT / "app_screenshots" / "app_olap.png",
        "app_paper_review.png": ROOT / "app_screenshots" / "app_paper_review.png",
        "app_governance.png": ROOT / "app_screenshots" / "app_governance.png",
        "correlation_results.png": ROOT / "figures" / "correlation_results.png",
        "model_evaluation_results.png": ROOT / "figures" / "model_evaluation_results.png",
        "fit_diagnostics_results.png": ROOT / "figures" / "fit_diagnostics_results.png",
        "actual_vs_predicted_results.png": ROOT / "figures" / "actual_vs_predicted_results.png",
        "olap_period_results.png": ROOT / "figures" / "olap_period_results.png",
        "project_pipeline.png": ROOT / "figures" / "project_pipeline.png",
    }

    def s(title: str, bullets: list[str] | None = None, image_name: str | None = None, layout: str = "bullets") -> dict[str, object]:
        return {"content": {"title": title, "bullets": bullets or [], "image": image_name, "layout": layout}, "image_name": image_name}

    slides = [
        s("US Housing Intelligence Platform", ["Final report and Streamlit app defense", "Data Mining and Machine Learning project", "Ayoub Naouech and Rayen Belhadj"], layout="title"),
        s("Presentation Roadmap", ["Context, problem, and objectives", "Dataset, preparation, and CRISP-DM", "Streamlit app screenshots and modules", "Evaluation, diagnostics, OLAP, and paper-review evidence", "Limitations, future work, and conclusion"]),
        s("Project Context", ["Housing prices connect household wealth, credit conditions, inflation, construction, and employment", "A useful analysis combines economic theory with data evidence", "The project converts a macroeconomic dataset into an interactive intelligence platform"]),
        s("Problem Statement", ["How can data mining and machine learning analyze and predict U.S. housing-market trends?", "The market is time-dependent and influenced by several interacting variables", "The final result must be understandable for non-technical users"]),
        s("Project Objectives", ["Analyze historical Home Price Index behavior", "Identify indicators associated with housing-price movement", "Compare models with reproducible metrics", "Build a Streamlit app for visualization, modeling, OLAP, export, and interpretation"]),
        s("Dataset Overview", ["Target variable: Home_Price_Index", "Main variables: mortgage rate, interest rate, unemployment, inflation, permits, income, population, and sentiment", "The data supports both prediction and interpretation"], layout="metrics"),
        s("Data Preparation", ["Parse and sort the date column chronologically", "Convert numeric columns into model-ready values", "Check missing values and duplicates", "Create administration labels, lags, rolling features, ratios, and percentage changes"]),
        s("CRISP-DM Methodology", ["Business understanding defines the housing problem", "Data understanding checks quality and temporal structure", "Preparation builds usable analytical features", "Modeling and evaluation compare algorithms", "Deployment is delivered through Streamlit"]),
        s("System Architecture", ["Data layer, analysis layer, modeling layer, interpretation layer, governance layer", "The app connects raw data, models, charts, paper review, and reporting"], "project_pipeline.png", "image_left"),
        s("Streamlit App: Home And Workflow", ["The dashboard presents the project as a housing intelligence product", "Sidebar navigation guides the user across analysis, modeling, reporting, and governance"], "app_home.png", "image_full"),
        s("Streamlit App: ML Lab", ["Users select target, features, scaling method, and model type", "The app supports interactive supervised learning rather than static code"], "app_modeling.png", "image_full"),
        s("Streamlit App: Evaluation", ["Models are compared with R2, MAE, RMSE, and classification metrics when applicable", "The page explains why the best model wins"], "app_evaluation.png", "image_full"),
        s("Streamlit App: OLAP And Export", ["OLAP pivot tables summarize data by period and dimensions", "3D cube and heatmap visuals help explain segment differences", "Export functions reuse results in reports"], "app_olap.png", "image_full"),
        s("Streamlit App: Paper Review", ["Compares dashboard evidence with housing theory and market interpretation", "Explains rates, supply, demand, timing, and affordability together"], "app_paper_review.png", "image_left"),
        s("Streamlit App: Governance", ["Production readiness checks include validity, missingness, duplicates, target availability, model cards, and registry notes", "This makes the project feel closer to a professional data product"], "app_governance.png", "image_full"),
        s("Correlation Results", ["Strong relationships appear around lagged HPI, smoothed HPI, and Median_Income", "Correlation supports interpretation but does not prove causality", "Housing prices are persistent and linked to broader economic context"], "correlation_results.png", "image_left"),
        s("Model Evaluation Results", ["Compared models include Linear Regression, Ridge, Random Forest, Gradient Boosting, Decision Tree, KNN, and SVR", "Best generated metrics: R2 = 0.999, RMSE = 0.797, MAE = 0.661"], "model_evaluation_results.png", "image_left"),
        s("Regression Metrics Table", ["Linear Regression: MAE 0.661, RMSE 0.797, R2 0.999, CV R2 0.994", "Ridge Regression: MAE 3.715, RMSE 4.162, R2 0.984", "Tree-based models perform weaker on the final chronological test period", "Evaluation must be time-aware, not based only on model popularity"]),
        s("Fit Diagnostics", ["Diagnostics compare train and test performance", "Large train-test gaps indicate weak generalization or overfitting risk", "Tree models struggle when the final period moves beyond the training range"], "fit_diagnostics_results.png", "image_left"),
        s("Actual Versus Predicted", ["The best model follows the chronological test period closely", "Lagged and smoothed price features explain the strong predictive result", "The visual is easier to defend than metrics alone"], "actual_vs_predicted_results.png", "image_left"),
        s("OLAP Period Results", ["Biden period has the highest average Home Price Index", "Biden period also has the highest average mortgage-rate pressure", "Prices and affordability pressure can rise together when supply and timing matter"], "olap_period_results.png", "image_left"),
        s("Paper Review Message", ["Housing prices are shaped by demand, financing conditions, supply, construction, income, and timing", "Higher mortgage rates reduce affordability, but prices may react slowly when supply is constrained", "The strongest conclusion is multi-factor interpretation"]),
        s("Business Impact", ["Analytical impact: users understand which indicators move with housing prices", "Decision-support impact: users compare models, periods, OLAP segments, and scenarios", "Educational impact: the project demonstrates EDA, ML, forecasting, OLAP, paper review, and governance"]),
        s("Limitations And Future Work", ["National-level data hides regional differences", "Forecasts are educational and not financial advice", "Future work: regional data, live APIs, SHAP explainability, better chatbot grounding, monitoring, and authentication"]),
        s("Final Conclusion", ["The report, poster, and app now tell one consistent story", "U.S. housing prices require theory, correlations, model evaluation, diagnostics, OLAP, and market context", "The Streamlit dashboard is a housing intelligence platform, not only a collection of charts"]),
    ]
    package_parts(slides, media_sources)
    print(f"Created {OUT}")


if __name__ == "__main__":
    main()

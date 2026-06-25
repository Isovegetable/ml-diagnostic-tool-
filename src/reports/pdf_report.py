"""
PDF 报告生成模块。

使用 fpdf2 生成正式报告 PDF。
支持中文字体（自动探测 Windows/Mac/Linux 系统字体）。
"""

import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

from fpdf import FPDF

# 自动探测系统可用的中文字体
_CJK_FONT_PATHS = [
    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/deng.ttf",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    # Linux (apt install fonts-noto-cjk)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    # Linux (apt install fonts-wqy-microhei)
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]
_CJK_FONT = None
for _p in _CJK_FONT_PATHS:
    if os.path.exists(_p):
        _CJK_FONT = _p
        break


class _ReportPDF(FPDF):
    """自定义 PDF 样式（支持中文）。"""

    def __init__(self):
        super().__init__()
        if _CJK_FONT:
            self.add_font("CJK", "", _CJK_FONT, uni=True)
            # 尝试加粗版本
            bold_path = _CJK_FONT.replace(".ttc", "bd.ttc").replace(".ttf", "bd.ttf")
            if os.path.exists(bold_path):
                self.add_font("CJK", "B", bold_path, uni=True)
            else:
                self.add_font("CJK", "B", _CJK_FONT, uni=True)
        self._use_cjk = _CJK_FONT is not None

    def _font(self, style="", size=10):
        """返回可用字体名（CJK 不支持 italic 时降级为常规）。"""
        if not self._use_cjk:
            return ("Helvetica", style, size)
        safe_style = style if style in ("", "B") else ""
        return ("CJK", safe_style, size)

    def header(self):
        if self.page_no() > 1:
            f = self._font("I", 8); self.set_font(*f)
            self.set_text_color(150, 150, 150)
            self.cell(0, 6, "Materials ML Diagnostic Report" if not self._use_cjk else "材料机器学习诊断报告", align="L")
            self.cell(0, 6, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
            self.line(10, 14, 200, 14)
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        f = self._font("", 7); self.set_font(*f)
        self.set_text_color(180, 180, 180)
        self.cell(0, 10,
            "Disclaimer: For research reference only" if not self._use_cjk
            else "本报告由材料机器学习自动诊断工具生成，仅供参考",
            align="C")

    def section_title(self, title_cn: str, title_en: str = ""):
        title = title_en if not self._use_cjk else title_cn
        f = self._font("B", 13); self.set_font(*f)
        self.set_text_color(31, 119, 180)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(31, 119, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def body_text(self, text: str):
        f = self._font("", 10); self.set_font(*f)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def key_value(self, key_cn: str, value: str, key_en: str = ""):
        key = key_en if not self._use_cjk else key_cn
        f = self._font("", 10); self.set_font(*f)
        self.set_text_color(50, 50, 50)
        self.cell(60, 7, key, align="R")
        f = self._font("B", 10); self.set_font(*f)
        self.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def metric_grid(self, metrics: list[tuple[str, str, str]]):
        """三列指标卡片。"""
        x_start = self.get_x()
        y_start = self.get_y()
        self.set_fill_color(240, 244, 248)
        col_w = 57
        for i, (label, value, note) in enumerate(metrics):
            x = x_start + i * (col_w + 4)
            self.set_xy(x, y_start)
            self.rect(x, y_start, col_w, 22, style="DF")
            self.set_xy(x + 2, y_start + 2)
            f = self._font("B", 14); self.set_font(*f)
            self.set_text_color(31, 119, 180)
            self.cell(col_w - 4, 7, value)
            self.set_xy(x + 2, y_start + 10)
            f = self._font("", 8); self.set_font(*f)
            self.set_text_color(100, 100, 100)
            self.cell(col_w - 4, 5, label)
            if note:
                self.set_xy(x + 2, y_start + 15)
                f = self._font("", 7); self.set_font(*f)
                self.set_text_color(150, 150, 150)
                self.cell(col_w - 4, 4, note)
        self.set_xy(x_start, y_start + 26)


def _level_label(level: str) -> str:
    labels = {"A": "适合建模", "B": "可探索", "C": "高风险", "D": "不建议"}
    return labels.get(level, "—")


def generate_pdf_report(
    original_shape: tuple,
    sample_size: int,
    feature_count: int,
    target_column: str,
    best_model_name: str,
    test_r2: float,
    train_r2: float,
    cv_r2_mean: float,
    cv_r2_std: float,
    mae: float,
    rmse: float,
    suggestions: list[str],
    diagnosis: dict | None = None,
    data_quality: dict | None = None,
) -> bytes:
    """
    生成 PDF 格式诊断报告。

    Args: 参数含义与 generate_report_text 一致。

    Returns:
        bytes: PDF 文件字节流
    """
    pdf = _ReportPDF()
    pdf.add_page()

    # ========== 封面 ==========
    pdf.ln(20)
    f = pdf._font("B", 24); pdf.set_font(*f)
    pdf.set_text_color(31, 119, 180)
    pdf.cell(0, 15, "材料机器学习诊断报告", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    f = pdf._font("", 11); pdf.set_font(*f)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 7, f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"目标列：{target_column}", align="C", new_x="LMARGIN", new_y="NEXT")
    if data_quality:
        pdf.cell(0, 7, f"数据集：{data_quality.get('dataset', target_column)}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)

    # ========== 诊断等级 ==========
    if diagnosis:
        level = diagnosis["level"]
        label = _level_label(level)
        colors = {"A": (26, 125, 54), "B": (184, 134, 11), "C": (204, 85, 0), "D": (179, 0, 0)}
        color = colors.get(level, (100, 100, 100))
        score = diagnosis["score"]

        pdf.set_fill_color(*color)
        pdf.set_text_color(255, 255, 255)
        f = pdf._font("B", 28); pdf.set_font(*f)
        pdf.cell(30, 30, f"  {level}  ", fill=True, align="C")
        f = pdf._font("", 12); pdf.set_font(*f)
        pdf.cell(0, 30, f"  {label}   评分 {score}/100", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        f = pdf._font("I", 9); pdf.set_font(*f)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 5, diagnosis.get("reason", ""))
        pdf.ln(8)

    # ========== 数据概况 ==========
    pdf.section_title("数据概况")
    pdf.key_value("原始样本数", str(original_shape[0]))
    pdf.key_value("有效建模样本数", str(sample_size))
    pdf.key_value("特征数量", str(feature_count))
    pdf.key_value("目标列", target_column)
    if data_quality:
        dq_parts = []
        if data_quality.get("duplicate_rows", 0) > 0:
            dq_parts.append(f"重复行：{data_quality['duplicate_rows']}")
        if data_quality.get("outlier_rows", 0) > 0:
            dq_parts.append(f"异常值行：{data_quality['outlier_rows']}")
        if data_quality.get("near_dup_rows", 0) > 0:
            dq_parts.append(f"近重复：{data_quality['near_dup_rows']}")
        if dq_parts:
            f = pdf._font("", 9); pdf.set_font(*f)
            pdf.set_text_color(180, 100, 0)
            pdf.cell(0, 7, "  数据质量提示：" + " | ".join(dq_parts), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
    pdf.ln(4)

    # ========== 最佳模型 ==========
    pdf.section_title("最佳模型")
    f = pdf._font("B", 14); pdf.set_font(*f)
    pdf.set_text_color(31, 119, 180)
    pdf.cell(0, 10, best_model_name, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.metric_grid([
        ("交叉验证 R²", f"{cv_r2_mean:.4f}", f"±{cv_r2_std:.4f}"),
        ("训练集 R²", f"{train_r2:.4f}", ""),
        ("测试集 R²", f"{test_r2:.4f}", ""),
    ])
    pdf.metric_grid([
        ("MAE", f"{mae:.4f}", ""),
        ("RMSE", f"{rmse:.4f}", ""),
        ("", "", ""),
    ])
    pdf.ln(4)

    # ========== 诊断建议 ==========
    if suggestions:
        pdf.section_title("诊断建议")
        f = pdf._font("", 10); pdf.set_font(*f)
        pdf.set_text_color(50, 50, 50)
        for s in suggestions:
            pdf.cell(5, 6, "-")
            pdf.multi_cell(0, 6, s)
            pdf.ln(1)
        pdf.ln(2)

    # ========== 免责声明 ==========
    pdf.ln(6)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    f = pdf._font("I", 8); pdf.set_font(*f)
    pdf.set_text_color(140, 140, 140)
    pdf.multi_cell(0, 4.5,
        "免责声明：本工具仅用于科研数据初步分析，"
        "不代表因果结论，不保证论文发表。"
        "所有结果均基于自动化模型流程生成，仅供参考。"
    )

    return bytes(pdf.output())

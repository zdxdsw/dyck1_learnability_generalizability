from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas
from reportlab.lib.colors import black
import io
from matplotlib import font_manager
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

dejavu_path = font_manager.findfont("DejaVu Sans")
pdfmetrics.registerFont(TTFont("DejaVuSans", dejavu_path))

left_pdf = "rnn_stress_test_neg.pdf"
right_pdf = "../../plot_for_paper_transformer/transformer_stress_test_neg/transformer_stress_test_neg.pdf"
out_pdf = "combined_stress_test_neg.pdf"

left_page = PdfReader(left_pdf).pages[0]
right_page = PdfReader(right_pdf).pages[0]

# Use crop boxes; matplotlib tight PDFs often rely on these
left_box = left_page.cropbox
right_box = right_page.cropbox

left_w = float(left_box.width)
left_h = float(left_box.height)
right_w = float(right_box.width)
right_h = float(right_box.height)

gap = -8
title_space = 0
top_margin = 2

new_w = left_w + gap + right_w
plot_h = max(left_h, right_h)
new_h = plot_h + title_space

writer = PdfWriter()
new_page = writer.add_blank_page(width=new_w, height=new_h)

left_y = (plot_h - left_h) / 2
right_y = (plot_h - right_h) / 2

# Important: account for nonzero cropbox lower-left coordinates
left_tx = -float(left_box.left)
left_ty = -float(left_box.bottom) + left_y

right_tx = left_w + gap - float(right_box.left)
right_ty = -float(right_box.bottom) + right_y

new_page.merge_transformed_page(
    left_page,
    Transformation().translate(tx=left_tx, ty=left_ty),
)

new_page.merge_transformed_page(
    right_page,
    Transformation().translate(tx=right_tx, ty=right_ty),
)

with open(out_pdf, "wb") as f:
    writer.write(f)
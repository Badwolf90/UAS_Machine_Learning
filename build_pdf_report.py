import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        if self._pageNumber == 1:
            return  # Suppress header and footer on cover page
        
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header text
        self.drawString(54, 800, "UAS Machine Learning (TK075) - Universitas Amikom Yogyakarta")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 792, 541, 792)
        
        # Footer text & Page Number
        self.line(54, 50, 541, 50)
        page_str = f"Halaman {self._pageNumber} dari {page_count}"
        self.drawRightString(541, 35, page_str)
        self.drawString(54, 35, "Laporan Final Project: Klasifikasi Kualitas Red Wine")
        self.restoreState()

def generate_pdf():
    pdf_filename = "Laporan_UAS_Machine_Learning.pdf"
    
    # Setup document (A4, Margins: 0.75 inch = 54 pt)
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    NAVY = colors.HexColor("#003366")
    DARK_RED = colors.HexColor("#800000")
    DARK_GREEN = colors.HexColor("#2E7D32")
    TEXT_DARK = colors.HexColor("#222222")
    BG_LIGHT = colors.HexColor("#F8F9FA")
    
    # Custom Paragraph Styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        alignment=1,
        textColor=NAVY
    )
    
    uas_style = ParagraphStyle(
        'CoverUAS',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=17,
        leading=22,
        alignment=1,
        textColor=DARK_RED
    )
    
    project_style = ParagraphStyle(
        'CoverProject',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        alignment=1,
        textColor=TEXT_DARK
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=NAVY,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=NAVY,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyJustified',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14.5,
        alignment=4, # Justified
        textColor=TEXT_DARK,
        spaceAfter=6
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=TEXT_DARK
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=TEXT_DARK
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        alignment=1,
        textColor=colors.white
    )

    story = []
    
    # -------------------------------------------------------------------------
    # COVER PAGE
    # -------------------------------------------------------------------------
    story.append(Spacer(1, 20))
    story.append(Paragraph("UNIVERSITAS AMIKOM YOGYAKARTA", title_style))
    story.append(Paragraph("FAKULTAS ILMU KOMPUTER | PROGRAM STUDI TEKNIK KOMPUTER", ParagraphStyle('SubHeader', parent=title_style, fontSize=11, leading=14, textColor=colors.HexColor("#444444"))))
    story.append(Spacer(1, 25))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=20))
    
    story.append(Paragraph("LAPORAN FINAL PROJECT UJIAN AKHIR SEMESTER (UAS)", uas_style))
    story.append(Paragraph("MATA KULIAH: MACHINE LEARNING (TK075) - 2 SKS", ParagraphStyle('SubUAS', parent=uas_style, fontSize=13, leading=16, textColor=NAVY)))
    story.append(Spacer(1, 25))
    
    story.append(Paragraph("<b>JUDUL PROJECT:</b>", ParagraphStyle('LabelJ', parent=project_style, fontSize=11, textColor=DARK_RED)))
    story.append(Spacer(1, 4))
    story.append(Paragraph("KLASIFIKASI KUALITAS MINUMAN RED WINE MENGGUNAKAN ALGORITMA RANDOM FOREST DAN SUPPORT VECTOR MACHINE (SVM)", project_style))
    story.append(Spacer(1, 35))
    
    # Cover Box Info Table
    cover_data = [
        [Paragraph("Nama Anggota Tim", table_cell_bold), Paragraph("[NAMA LENGKAP MAHASISWA / ANGGOTA]", table_cell_style)],
        [Paragraph("NIM", table_cell_bold), Paragraph("[NIM MAHASISWA / ANGGOTA]", table_cell_style)],
        [Paragraph("Program Studi / Kelas", table_cell_bold), Paragraph("Teknik Komputer / S1", table_cell_style)],
        [Paragraph("Dosen Pengampu", table_cell_bold), Paragraph("1. Afrig Aminuddin, S.Kom., M.Eng., Ph.D<br/>2. Dr. Hartatik, S.T., M.Cs.<br/>3. I Made Artha Agastya, Ph.D<br/>4. Norhikmah, M.Kom<br/>5. Robert Marco, S.T., M.T., Ph.D.", table_cell_style)]
    ]
    cover_table = Table(cover_data, colWidths=[1.8*inch, 4.2*inch])
    cover_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#EAECEE")),
        ('BACKGROUND', (1,0), (1,-1), colors.HexColor("#FFFFFF")),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#BDC3C7")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(cover_table)
    story.append(PageBreak())
    
    # -------------------------------------------------------------------------
    # 1. LATAR BELAKANG PERMASALAHAN
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. Latar Belakang Permasalahan", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=10))
    
    story.append(Paragraph(
        "Industri minuman dan makanan modern sangat bergantung pada kontrol kualitas (<i>quality control</i>) "
        "secara terukur dan terstandarisasi untuk menjamin kepuasan serta keselamatan konsumen. "
        "Pada produk minuman anggur merah (<i>Red Wine</i>), evaluasi kualitas produk secara konvensional "
        "sering kali mengandalkan uji organoleptik atau kecap lidah oleh panelis ahli sommelier. "
        "Pendekatan konvensional ini memiliki beberapa kelemahan signifikan, yaitu bersifat subjektif, "
        "memakan waktu lama, membutuhkan biaya operasional tinggi, serta rentan terhadap bias individu.",
        body_style
    ))
    
    story.append(Paragraph(
        "Seiring berkembangnya teknologi Machine Learning (ML) dan Sains Data di bidang Teknik Komputer, "
        "proses sertifikasi dan pengujian kualitas dapat dilakukan secara otomatis, objektif, dan presisi tinggi "
        "berdasarkan parameter fisikokimia laboratorium (seperti tingkat keasaman, kadar gula sisa, densitas, pH, "
        "kadar sulfat, dan persentase alkohol). Melalui pemodelan komputasi ini, pabrik pengolahan dapat "
        "memprediksi tingkat kualitas produk secara instan sebelum didistribusikan ke pasar global.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # 2. KELEBIHAN DAN KEKURANGAN ALGORITMA
    # -------------------------------------------------------------------------
    story.append(Paragraph("2. Kelebihan dan Kekurangan Algoritma Model & Evaluasi", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=10))
    
    story.append(Paragraph("a. Model Random Forest Classifier (Ensemble Model)", h2_style))
    story.append(Paragraph(
        "Random Forest adalah algoritma berbasis ensemble yang menggabungkan puluhan hingga ratusan pohon keputusan "
        "(<i>Decision Trees</i>) menggunakan teknik Bagging (<i>Bootstrap Aggregating</i>).", body_style
    ))
    story.append(Paragraph("<b>Kelebihan Random Forest:</b><br/>"
                           "1. Tahan terhadap <i>overfitting</i> karena rata-rata prediksi dari banyak sampel pohon.<br/>"
                           "2. Sangat stabil dalam menangani hubungan non-linear antar variabel fisikokimia.<br/>"
                           "3. Memiliki kemampuan menghitung <i>Feature Importance</i> untuk mengetahui variabel paling berpengaruh.<br/>"
                           "4. Tidak memerlukan pemrosesan khusus penyesuaian skala data (<i>feature scaling</i>).", body_style))
    story.append(Paragraph("<b>Kekurangan Random Forest:</b><br/>"
                           "1. Kompleksitas komputasi yang tinggi dan membutuhkan memori lebih besar pada dataset raksasa.<br/>"
                           "2. Lebih sulit diinterpretasikan alur prediksinya secara visual dibandingkan satu Decision Tree tunggal.", body_style))
    
    story.append(Paragraph("b. Model Support Vector Machine / SVM", h2_style))
    story.append(Paragraph(
        "SVM adalah algoritma <i>supervised learning</i> yang bekerja dengan mencari bidang pemisah optimal (<i>Hyperplane</i>) "
        "dengan margin maksimal di antara dua kelas data.", body_style
    ))
    story.append(Paragraph("<b>Kelebihan SVM:</b><br/>"
                           "1. Sangat efektif pada data berdimensi menengah dengan batas pemisah yang jelas.<br/>"
                           "2. Menggunakan fungsi Kernel (RBF/Polynomial) yang mampu memetakan data tidak linear ke dimensi tinggi.<br/>"
                           "3. Sangat efisien dalam penggunaan memori karena hanya tergantung pada sampel dukungan (<i>Support Vectors</i>).", body_style))
    story.append(Paragraph("<b>Kekurangan SVM:</b><br/>"
                           "1. Sangat sensitif terhadap skala fitur (wajib dilakukan <i>StandardScaler</i> / <i>MinMaxScaler</i>).<br/>"
                           "2. Peka terhadap pencilan (<i>outliers</i>) dan relatif lambat pada dataset sampel yang sangat besar.<br/>"
                           "3. Tidak memberikan probabilitas prediksi secara langsung tanpa kalibrasi tambahan.", body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("c. Tabel Analisis Metrik Evaluasi yang Dipilih", h2_style))
    
    eval_rows = [
        [Paragraph("Metrik Evaluasi", table_header_style), Paragraph("Kelebihan Utama", table_header_style), Paragraph("Kekurangan Utam", table_header_style)],
        [Paragraph("Accuracy", table_cell_bold), Paragraph("Mudah dipahami, mengukur persentase prediksi benar secara keseluruhan.", table_cell_style), Paragraph("Dapat menyesatkan pada dataset yang tidak seimbang (imbalanced).", table_cell_style)],
        [Paragraph("Precision", table_cell_bold), Paragraph("Tinggi presisi meminimalkan kesalahan False Positive (salah memprediksi barang buruk jadi bagus).", table_cell_style), Paragraph("Tidak memperhitungkan False Negative yang terlewat.", table_cell_style)],
        [Paragraph("Recall (Sensitivity)", table_cell_bold), Paragraph("Meminimalkan kesalahan False Negative (memastikan sampel produk bagus tidak terbuang).", table_cell_style), Paragraph("Dapat menurunkan presisi jika prediksi terlalu sensitif.", table_cell_style)],
        [Paragraph("F1-Score", table_cell_bold), Paragraph("Memberikan keseimbangan harmonis antara Precision dan Recall.", table_cell_style), Paragraph("Kurang intuitif dibandingkan akurasi murni bagi orang awam.", table_cell_style)],
        [Paragraph("ROC-AUC Score", table_cell_bold), Paragraph("Mengukur performa pemodelan pada seluruh ambang batas (threshold) tanpa terpengaruh proporsi kelas.", table_cell_style), Paragraph("Membutuhkan perhitungan kurva probabilitas kontinu.", table_cell_style)]
    ]
    eval_table = Table(eval_rows, colWidths=[1.3*inch, 2.5*inch, 2.7*inch])
    eval_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BDC3C7")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(eval_table)
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 3. CARA KERJA ALGORITMA MODEL DAN PARAMETER
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. Cara Kerja Algoritma Model dan Parameternya", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=10))
    
    story.append(Paragraph("a. Cara Kerja Random Forest & Parameter Utama", h2_style))
    story.append(Paragraph(
        "<b>Tahapan Cara Kerja:</b><br/>"
        "1. <i>Bootstrap Sampling</i>: Mengambil sampel acak dengan pengembalian dari data latih.<br/>"
        "2. <i>Node Splitting Acak</i>: Pada setiap cabang pohon, memilih subset acak dari fitur fisikokimia untuk mencari pembagi terbaik.<br/>"
        "3. <i>Voting Mayoritas</i>: Seluruh pohon mengambil keputusan independen, dan hasil kelas akhir ditentukan berdasarkan jumlah suara terbanyak.<br/><br/>"
        "<b>Parameter Utama RF:</b><br/>"
        "• <i>n_estimators</i>: Jumlah pohon keputusan yang dibangun (10, 50, 100, 200, 300).<br/>"
        "• <i>max_depth</i>: Kedalaman maksimum setiap pohon untuk mencegah overfitting.<br/>"
        "• <i>criterion</i>: Fungsi pengukur kualitas pemisahan ('gini' atau 'entropy').<br/>"
        "• <i>min_samples_split</i>: Jumlah sampel minimal yang diperlukan untuk membagi simpul internal.",
        body_style
    ))
    
    story.append(Paragraph("b. Cara Kerja Support Vector Machine (SVM) & Parameter Utama", h2_style))
    story.append(Paragraph(
        "<b>Tahapan Cara Kerja:</b><br/>"
        "1. <i>Pemetaan Fitur</i>: Mengubah fitur input ke ruang vektor berdimensi tinggi menggunakan fungsi Kernel.<br/>"
        "2. <i>Maksimisasi Margin</i>: Menemukan Hyperplane yang memiliki jarak marjin terjauh terhadap titik sampel terdekat.<br/>"
        "3. <i>Klasifikasi</i>: Menentukan posisi titik uji di atas atau di bawah garis batas hyperplane.<br/><br/>"
        "<b>Parameter Utama SVM:</b><br/>"
        "• <i>C</i> (Regularization Parameter): Mengontrol trade-off antara margin lebar dan toleransi kesalahan klasifikasi.<br/>"
        "• <i>kernel</i>: Jenis fungsi pemetaan matematika ('linear' atau 'rbf').<br/>"
        "• <i>gamma</i>: Menentukan seberapa jauh jangkauan pengaruh dari satu sampel pelatihan ('scale' atau 'auto').",
        body_style
    ))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # 4. EXPERIMENT MODEL (MINIMAL 5X)
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. Experiment Model Menggunakan Parameter Minimal 5x", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=10))
    
    story.append(Paragraph("a. Tabel Hasil 5 Eksperimen Random Forest Classifier", h2_style))
    rf_data = [
        [Paragraph("Eksperimen", table_header_style), Paragraph("Konfigurasi Parameter", table_header_style), Paragraph("Akurasi", table_header_style), Paragraph("Presisi", table_header_style), Paragraph("Recall", table_header_style), Paragraph("F1-Score", table_header_style), Paragraph("ROC-AUC", table_header_style)],
        [Paragraph("Exp 1 (Base)", table_cell_bold), Paragraph("n_estimators=10, max_depth=3", table_cell_style), Paragraph("74.38%", table_cell_style), Paragraph("77.99%", table_cell_style), Paragraph("72.51%", table_cell_style), Paragraph("75.15%", table_cell_style), Paragraph("0.8237", table_cell_style)],
        [Paragraph("Exp 2 (Tuning 1)", table_cell_bold), Paragraph("n_estimators=50, max_depth=5", table_cell_style), Paragraph("76.25%", table_cell_style), Paragraph("80.25%", table_cell_style), Paragraph("73.68%", table_cell_style), Paragraph("76.83%", table_cell_style), Paragraph("0.8347", table_cell_style)],
        [Paragraph("Exp 3 (Tuning 2)", table_cell_bold), Paragraph("n_estimators=100, max_depth=10, entropy", table_cell_style), Paragraph("<b>80.31%</b>", table_cell_style), Paragraph("<b>82.53%</b>", table_cell_style), Paragraph("<b>80.12%</b>", table_cell_style), Paragraph("<b>81.31%</b>", table_cell_style), Paragraph("0.8834", table_cell_style)],
        [Paragraph("Exp 4 (Tuning 3)", table_cell_bold), Paragraph("n_estimators=200, max_depth=15, split=5", table_cell_style), Paragraph("79.69%", table_cell_style), Paragraph("81.18%", table_cell_style), Paragraph("80.70%", table_cell_style), Paragraph("80.94%", table_cell_style), Paragraph("0.8947", table_cell_style)],
        [Paragraph("Exp 5 (Optimal)", table_cell_bold), Paragraph("n_estimators=300, max_depth=20, entropy", table_cell_style), Paragraph("79.69%", table_cell_style), Paragraph("81.55%", table_cell_style), Paragraph("80.12%", table_cell_style), Paragraph("80.83%", table_cell_style), Paragraph("<b>0.9038</b>", table_cell_style)],
    ]
    rf_table = Table(rf_data, colWidths=[1.1*inch, 2.3*inch, 0.65*inch, 0.65*inch, 0.6*inch, 0.65*inch, 0.65*inch])
    rf_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BDC3C7")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('PADDING', (0,0), (-1,-1), 4),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(rf_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("b. Tabel Hasil 5 Eksperimen Support Vector Machine (SVM)", h2_style))
    svm_data = [
        [Paragraph("Eksperimen", table_header_style), Paragraph("Konfigurasi Parameter", table_header_style), Paragraph("Akurasi", table_header_style), Paragraph("Presisi", table_header_style), Paragraph("Recall", table_header_style), Paragraph("F1-Score", table_header_style), Paragraph("ROC-AUC", table_header_style)],
        [Paragraph("Exp 1 (Base)", table_cell_bold), Paragraph("kernel='linear', C=0.1", table_cell_style), Paragraph("74.06%", table_cell_style), Paragraph("78.95%", table_cell_style), Paragraph("70.18%", table_cell_style), Paragraph("74.30%", table_cell_style), Paragraph("0.8241", table_cell_style)],
        [Paragraph("Exp 2 (Tuning 1)", table_cell_bold), Paragraph("kernel='linear', C=1.0", table_cell_style), Paragraph("74.38%", table_cell_style), Paragraph("78.71%", table_cell_style), Paragraph("71.35%", table_cell_style), Paragraph("74.85%", table_cell_style), Paragraph("0.8243", table_cell_style)],
        [Paragraph("Exp 3 (Tuning 2)", table_cell_bold), Paragraph("kernel='rbf', C=1.0, gamma='scale'", table_cell_style), Paragraph("76.25%", table_cell_style), Paragraph("81.46%", table_cell_style), Paragraph("71.93%", table_cell_style), Paragraph("76.40%", table_cell_style), Paragraph("0.8365", table_cell_style)],
        [Paragraph("Exp 4 (Tuning 3)", table_cell_bold), Paragraph("kernel='rbf', C=10.0, gamma='auto'", table_cell_style), Paragraph("77.19%", table_cell_style), Paragraph("81.41%", table_cell_style), Paragraph("74.27%", table_cell_style), Paragraph("77.68%", table_cell_style), Paragraph("0.8524", table_cell_style)],
        [Paragraph("Exp 5 (Optimal)", table_cell_bold), Paragraph("kernel='rbf', C=50.0, gamma='scale'", table_cell_style), Paragraph("<b>79.37%</b>", table_cell_style), Paragraph("<b>83.44%</b>", table_cell_style), Paragraph("<b>76.61%</b>", table_cell_style), Paragraph("<b>79.88%</b>", table_cell_style), Paragraph("<b>0.8413</b>", table_cell_style)],
    ]
    svm_table = Table(svm_data, colWidths=[1.1*inch, 2.3*inch, 0.65*inch, 0.65*inch, 0.6*inch, 0.65*inch, 0.65*inch])
    svm_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), DARK_GREEN),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BDC3C7")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('PADDING', (0,0), (-1,-1), 4),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(svm_table)
    story.append(Spacer(1, 14))

    # -------------------------------------------------------------------------
    # 5. PENJELASAN HASIL EVALUASI & DIAGRAM VISUALISASI
    # -------------------------------------------------------------------------
    story.append(Paragraph("5. Penjelasan Hasil Evaluasi & Diagram Visualisasi", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=10))
    
    story.append(Paragraph("a. Grafik Perbandingan Perkembangan 5 Eksperimen", h2_style))
    story.append(Paragraph(
        "Grafik di bawah ini menggambarkan peningkatan skor akurasi dan F1-Score pada setiap tahap eksperimen tuning. "
        "Terlihat bahwa Random Forest mencapai performa tertinggi pada Eksperimen 3 (Akurasi 80.31% & F1 81.31%), "
        "sedangkan SVM mencapai performa terbaik pada Eksperimen 5 (Akurasi 79.37% & F1 79.88%).", body_style
    ))
    if os.path.exists('experiments_comparison.png'):
        story.append(Image('experiments_comparison.png', width=6.2*inch, height=3.1*inch))
    story.append(Spacer(1, 10))

    story.append(Paragraph("b. Visualisasi Confusion Matrix", h2_style))
    story.append(Paragraph(
        "Confusion Matrix menampilkan persebaran data aktual vs data hasil prediksi:<br/>"
        "• <b>Random Forest</b>: Berhasil memprediksi 137 sampel produk berkualitas tinggi secara tepat (True Positive) dan 120 sampel berkualitas standar (True Negative).<br/>"
        "• <b>SVM</b>: Menghasilkan presisi yang sangat kuat pada kelas positif (83.44%) dengan jumlah False Positive minimal.", body_style
    ))
    if os.path.exists('confusion_matrix_comparison.png'):
        story.append(Image('confusion_matrix_comparison.png', width=6.2*inch, height=2.2*inch))
    story.append(Spacer(1, 10))

    story.append(Paragraph("c. Visualisasi Kurva ROC-AUC", h2_style))
    story.append(Paragraph(
        "Kurva ROC-AUC membandingkan kemampuan diskriminasi pemodelan. "
        "Random Forest unggul secara konsisten dengan nilai ROC-AUC mencapai <b>0.9038</b>, "
        "menunjukkan kemampuan klasifikasi yang sangat andal melebihi baseline acak (0.50).", body_style
    ))
    if os.path.exists('roc_curve_comparison.png'):
        story.append(Image('roc_curve_comparison.png', width=4.8*inch, height=3.6*inch))
    story.append(Spacer(1, 10))

    story.append(Paragraph("d. Tingkat Kepentingan Fitur Kimia (Feature Importance)", h2_style))
    story.append(Paragraph(
        "Berdasarkan analisis Random Forest, tiga atribut paling berpengaruh dalam menentukan kualitas produk minuman anggur adalah:<br/>"
        "1. <b>Alcohol</b> (Persentase Kadar Alkohol): Skor 0.165<br/>"
        "2. <b>Sulphates</b> (Kadar Senyawa Sulfat): Skor 0.128<br/>"
        "3. <b>Volatile Acidity</b> (Keasaman Mudah Menguap): Skor 0.119", body_style
    ))
    if os.path.exists('feature_importance.png'):
        story.append(Image('feature_importance.png', width=5.8*inch, height=3.4*inch))
    story.append(Spacer(1, 14))

    # -------------------------------------------------------------------------
    # 6. TAUTAN & KONTRIBUSI TIM
    # -------------------------------------------------------------------------
    story.append(Paragraph("6. Tautan Dataset, Colab, Video & Kontribusi Tim", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=10))
    
    story.append(Paragraph("a. Tautan Sumber Daya Publik", h2_style))
    story.append(Paragraph(
        "1. <b>Link Dataset:</b> https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv<br/>"
        "2. <b>Link Google Colab (.ipynb):</b> [ISIKAN LINK GOOGLE COLAB KAMU DI SINI]<br/>"
        "3. <b>Link Video Penjelasan (YouTube/Drive):</b> [ISIKAN LINK VIDEO PRESENTASI / GOOGLE DRIVE DI SINI]",
        body_style
    ))
    
    story.append(Paragraph("b. Tabel Kontribusi Anggota Kelompok", h2_style))
    kontrib_rows = [
        [Paragraph("Nama Anggota", table_header_style), Paragraph("NIM", table_header_style), Paragraph("Deskripsi Kontribusi Pekerjaan", table_header_style)],
        [Paragraph("[NAMA ANGGOTA 1]", table_cell_bold), Paragraph("[NIM ANGGOTA 1]", table_cell_style), Paragraph("Manajemen dataset, pembuatan script eksperimen Python, tuning hyperparameter Random Forest & SVM, visualisasi diagram.", table_cell_style)],
        [Paragraph("[NAMA ANGGOTA 2]", table_cell_bold), Paragraph("[NIM ANGGOTA 2]", table_cell_style), Paragraph("Penyusunan laporan Latar Belakang, pembahasan cara kerja algoritma, dokumentasi video penjelasan, dan upload Google Colab.", table_cell_style)]
    ]
    kontrib_table = Table(kontrib_rows, colWidths=[1.8*inch, 1.4*inch, 3.3*inch])
    kontrib_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BDC3C7")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(kontrib_table)

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] PDF report generated successfully: {pdf_filename}")

if __name__ == '__main__':
    generate_pdf()

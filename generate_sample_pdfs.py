import os, json
from fpdf import FPDF

# Load synthetic reports
with open("data/sample_reports/synthetic_100_reports.json", "r", encoding="utf-8") as f:
    reports = json.load(f)

# Pick 4 distinct industries:
# 1. General Manufacturing
# 2. Milk Processing
# 3. Catering
# 4. Storage & Warehouse
targets = [
    ("General Manufacturing", "Report_001_General_Manufacturing.pdf"),
    ("Milk Processing", "Report_014_Milk_Processing.pdf"),
    ("Catering", "Report_053_Catering.pdf"),
    ("Storage & Warehouse", "Report_089_Storage_and_Warehouse.pdf")
]

selected_reports = []
for cat_name, pdf_filename in targets:
    for rep in reports:
        if rep['category'] == cat_name:
            selected_reports.append((rep, pdf_filename))
            break

os.makedirs("sample_pdf_reports", exist_ok=True)

class AuditReportPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(30, 60, 114)
        self.cell(0, 10, 'FOOD SAFETY AND STANDARDS AUTHORITY OF INDIA (FSSAI)', align='C', new_x="LMARGIN", new_y="NEXT")
        self.set_font('Helvetica', 'I', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, 'Recognised Third-Party Food Safety Audit Report', align='C', new_x="LMARGIN", new_y="NEXT")
        self.line(10, 28, 200, 28)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()} | Confidential FSSAI Audit Record', align='C')

generated_files = []

for rep, pdf_name in selected_reports:
    pdf = AuditReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Metadata Block
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(45, 7, 'Audit Report ID:', border=0)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 7, rep['report_id'], new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(45, 7, 'Industry Sector:', border=0)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 7, rep['category'], new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(45, 7, 'Facility Name:', border=0)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 7, rep['facility_name'], new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(45, 7, 'Location:', border=0)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 7, rep['location'], new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(45, 7, 'Lead Auditor:', border=0)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 7, rep['auditor_name'], new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(45, 7, 'Audit Date:', border=0)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 7, rep['date'], new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(45, 7, 'Compliance Profile:', border=0)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 7, f"{rep['compliance_profile']} (Benchmark Standard)", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)
    
    # Report Observations Header
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(40, 80, 150)
    pdf.cell(0, 8, 'AUDITOR OBSERVATIONS & ASSESSMENT NARRATIVE', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    # Body Text
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(30, 30, 30)
    # Clean text to latin-1
    body_text = rep['report_text'].encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, body_text)
    
    pdf.ln(6)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, 'Official Endorsement: The findings detailed above represent factual conditions observed on-site by recognised food safety auditing personnel. Observations have been documented according to statutory requirements under the Food Safety and Standards (Food Safety Auditing) Regulations.')
    
    out_file = os.path.join("sample_pdf_reports", pdf_name)
    pdf.output(out_file)
    generated_files.append((rep['category'], rep['facility_name'], out_file))

print(f"Generated {len(generated_files)} PDF audit reports successfully!")
for cat, fac, path in generated_files:
    print(f"- Industry: {cat} | Facility: {fac} -> {path}")

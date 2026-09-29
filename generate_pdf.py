import os
import subprocess
from markdown_it import MarkdownIt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(CURRENT_DIR, "PROJECT_DOCUMENTATION.md")
HTML_PATH = os.path.join(CURRENT_DIR, "report", "project_documentation.html")
PDF_PATH = os.path.join(CURRENT_DIR, "PROJECT_DOCUMENTATION.pdf")

with open(MD_PATH, "r", encoding="utf-8") as f:
    md_content = f.read()

md = MarkdownIt("commonmark").enable("table")
html_body = md.render(md_content)

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Voice-Enabled Deep Learning Chatbot - Documentation</title>
    <style>
        @page {{
            size: A4;
            margin: 18mm 16mm 18mm 16mm;
            @bottom-right {{
                content: counter(page);
                font-size: 9pt;
                color: #718096;
            }}
        }}
        
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #1a202c;
            line-height: 1.6;
            font-size: 10.5pt;
            background: #ffffff;
            margin: 0;
            padding: 10px;
        }}
        
        h1 {{
            font-size: 22pt;
            color: #0f172a;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 8px;
            margin-top: 0;
            margin-bottom: 14px;
        }}
        
        h2 {{
            font-size: 14.5pt;
            color: #1e3a8a;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 5px;
            margin-top: 22px;
            margin-bottom: 10px;
            page-break-after: avoid;
        }}
        
        h3 {{
            font-size: 12pt;
            color: #2563eb;
            margin-top: 16px;
            margin-bottom: 6px;
            page-break-after: avoid;
        }}
        
        blockquote {{
            margin: 12px 0;
            padding: 10px 16px;
            background: #f0f7ff;
            border-left: 4px solid #3b82f6;
            border-radius: 4px;
            color: #1e293b;
        }}
        
        blockquote p {{
            margin: 4px 0;
        }}
        
        a {{
            color: #2563eb;
            text-decoration: none;
            font-weight: 500;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 14px 0;
            font-size: 9.5pt;
            page-break-inside: avoid;
        }}
        
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 7px 10px;
            text-align: left;
        }}
        
        th {{
            background: #f1f5f9;
            color: #0f172a;
            font-weight: 600;
        }}
        
        tr:nth-child(even) {{
            background: #f8fafc;
        }}
        
        pre {{
            background: #0f172a;
            color: #e2e8f0;
            padding: 12px 14px;
            border-radius: 6px;
            font-family: "Consolas", "Courier New", monospace;
            font-size: 8.5pt;
            line-height: 1.45;
            overflow-x: auto;
            margin: 12px 0;
            page-break-inside: avoid;
        }}
        
        code {{
            font-family: "Consolas", "Courier New", monospace;
            background: #f1f5f9;
            color: #0f172a;
            padding: 2px 5px;
            border-radius: 4px;
            font-size: 9pt;
        }}
        
        pre code {{
            background: transparent;
            color: inherit;
            padding: 0;
        }}
        
        ul, ol {{
            padding-left: 22px;
            margin: 8px 0;
        }}
        
        li {{
            margin-bottom: 4px;
        }}
        
        hr {{
            border: 0;
            border-top: 1px solid #e2e8f0;
            margin: 20px 0;
        }}
        
        .student-badge {{
            display: inline-block;
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: #ffffff;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 10pt;
            font-weight: 600;
            margin-bottom: 15px;
        }}
        
        .footer {{
            margin-top: 30px;
            font-size: 8pt;
            color: #64748b;
            text-align: center;
            border-top: 1px solid #e2e8f0;
            padding-top: 10px;
        }}
    </style>
</head>
<body>
<div class="student-badge">
    Student: Vyom Sen &nbsp;|&nbsp; Registration No: 23BAI0032
</div>
{html_body}
<div class="footer">
    Voice-Enabled Chatbot Deep Learning Project • Vyom Sen (Reg No: 23BAI0032) • Deployed on Vercel: https://voice-chatbot-hoa5fjkn6-vyom-vs11s-projects.vercel.app/
</div>
</body>
</html>
"""

os.makedirs(os.path.dirname(HTML_PATH), exist_ok=True)
with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"HTML written to: {HTML_PATH}")

# Chrome / Edge paths
edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

browser_exe = edge_path if os.path.exists(edge_path) else chrome_path

cmd = [
    browser_exe,
    "--headless",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={PDF_PATH}",
    HTML_PATH
]

print(f"Running command: {' '.join(cmd)}")
res = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", res.returncode)

if os.path.exists(PDF_PATH):
    size_kb = os.path.getsize(PDF_PATH) / 1024
    print(f"SUCCESS: Generated PDF at {PDF_PATH} ({size_kb:.1f} KB)")
    
    # Copy to Desktop
    desktop_pdf = os.path.join(os.path.expanduser("~"), "Desktop", "PROJECT_DOCUMENTATION.pdf")
    import shutil
    shutil.copy2(PDF_PATH, desktop_pdf)
    print(f"SUCCESS: Copied to Desktop at {desktop_pdf}")
else:
    print("FAILED to generate PDF.")
    if res.stderr:
        print("Stderr:", res.stderr)

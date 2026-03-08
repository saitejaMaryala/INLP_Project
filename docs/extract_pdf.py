import PyPDF2
import os
import re

pdf_path = r"c:\Users\choud\OneDrive\Desktop\NewCollegeDocs\3-2\INLP\Project\GithubRepo\INLP_Project\docs\Project_plan.pdf"
output_path = r"c:\Users\choud\OneDrive\Desktop\NewCollegeDocs\3-2\INLP\Project\GithubRepo\INLP_Project\docs\Project_plan.md"

# Extract text from PDF
text_content = []
with open(pdf_path, 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    print(f"Total pages: {len(pdf_reader.pages)}")
    
    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        text_content.append(text)
        print(f"Extracted page {page_num + 1}")

# Combine all text
full_text = "\n\n".join(text_content)

# Clean up excessive spacing (common PyPDF2 issue)
# Replace newlines between words with spaces (keeping paragraph breaks)
# First, mark intentional paragraph breaks
full_text = re.sub(r'\n\n+', '<<<PARAGRAPH>>>', full_text)
# Replace single newlines with spaces
full_text = re.sub(r'\n', ' ', full_text)
# Restore paragraph breaks
full_text = re.sub(r'<<<PARAGRAPH>>>', '\n\n', full_text)
# Replace multiple spaces with single space
full_text = re.sub(r' {2,}', ' ', full_text)
# Clean up multiple blank lines
full_text = re.sub(r'\n{3,}', '\n\n', full_text)

# Write to markdown file
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("# Project Plan for BERT Model\n\n")
    f.write(full_text)

print(f"\nSuccessfully converted PDF to {output_path}")
print(f"Total characters extracted: {len(full_text)}")

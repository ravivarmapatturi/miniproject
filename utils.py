




# Function to extract text from PDF using pdfplumber
def extracting_text_data(path):
    extracted_text = []
    with pdfplumber.open(path) as pdf:
        for i in range(len(pdf.pages)):
            page = pdf.pages[i]
            extracted_text.append(page.extract_text())
    return extracted_text

# Function to extract tables from PDF using tabula
def extracting_tabular_data(path):
    extracted_tabular = []
    dfs = tabula.read_pdf(path)
    for df in dfs:
        extracted_tabular.append(df.to_csv(index=False))
    return extracted_tabular

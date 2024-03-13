from pypdf import PdfReader

class TextExtractor():
    def extract_text_from_pdf(self, file):
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text = text + '\n' + page.extract_text()

        text = text.replace('\n', ' ').strip()
        return text

    def __call__(self, file, file_extension='pdf'):
        if file_extension == 'pdf':
            return self.extract_text_from_pdf(file)
        else:
            raise NotImplementedError("Currently only PDF text extraction is supported")

if __name__ == '__main__':
    text_extractor = TextExtractor()
    resume_text = text_extractor("advait_resume.pdf", file_extension='pdf')
    print(resume_text)
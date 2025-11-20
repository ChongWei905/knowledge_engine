import knowledge_engine

from knowledge_engine.transforms.extraction.extractor import EnhancedPDFTextExtractor

paths = "../pdfs"
index = "demoindex0"
ctx = knowledge_engine.init(exec_mode=knowledge_engine.ExecMode.LOCAL)

ds = (
    ctx.read.binary(paths, binary_format="pdf")
    .extract(EnhancedPDFTextExtractor(context=True, type="pymupdf"))
    .explode()
)

ds.write.files("output_docs")

output = ds.take_all()
for doc in output:
    print(doc)
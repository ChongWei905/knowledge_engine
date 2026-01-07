import knowledge_engine

from knowledge_engine.transforms.extraction.extractor import EnhancedPDFTextExtractor

paths = "../pdfs"
index = "demoindex0"
ctx = knowledge_engine.init(exec_mode=knowledge_engine.ExecMode.RAY)

ds = (
    ctx.read.binary(paths, binary_format="pdf", concurrency=2)
    .extract(EnhancedPDFTextExtractor(context=True, type="pymupdf"), concurrency=2)
    .explode(concurrency=2)
)

output = ds.take_all()
for doc in output:
    print(doc)
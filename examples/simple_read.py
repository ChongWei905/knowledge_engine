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

#
# df = DataFrame()
#
# for doc in output:
#     #
#     # df = df._append({"doc": doc.serialize()}, ignore_index=True)
#     print(doc)

# print(df.size)
#
# def byte_func(bt: bytes) -> bytes:
#     return bt[:10]
#
# def df_func(df: DataFrame) -> DataFrame:
#     docs = df['doc']
#     for doc in docs:
#         doc = Document.deserialize(doc)
#         line = doc.serialize()
#         df['doc'] = line[:10]
#     return df
#
# df = df['doc'].apply(byte_func)
# print(df)

import ray
import pyarrow as pa
import numpy as np


def main():
    # 1. 初始化 Ray 运行环境
    if not ray.is_initialized():
        ray.init()

    print("=== Ray Data Arrow Mapping Inspection ===")

    # 2. 准备模拟数据：一列嵌套字符串列表
    # 这种结构模拟了 Document 中的 elements 文本列表
    raw_data = [
        {"elements_text": ["text_1", "text_2", "text_6"]},
        {"elements_text": ["text_3", "text_4", "text_5"]}, # 测试空列表
    ]

    ds = ray.data.from_items(raw_data)

    # 3. 定义转换函数，检查输入 batch 的真实物理格式
    def inspect_worker_batch(batch: pa.Table) -> pa.Table:
        # 获取目标列数据
        column = batch["elements_text"]

        print("\n" + "=" * 50)
        print(f"Worker process received batch type: {type(batch)}")
        print(f"Column 'elements_text' type: {type(column)}")
        print(f"Arrow Logical Type: {column.type}")

        # 深入检查物理结构
        # 对于 list[str]，它应该是由三个 Buffer 组成的嵌套结构
        if isinstance(column, pa.ChunkedArray):
            # 获取第一个数据块进行分析
            first_chunk = column.chunk(0)
            print(f"Physical Layout: {type(first_chunk)}")

            # 验证它是否是 Arrow 的 ListArray
            if pa.types.is_list(column.type):
                print("SUCCESS: Detected Arrow ListArray structure.")
                # 获取内层类型，应该是 StringArray
                inner_type = column.type.value_type
                print(f"Inner (nested) value type: {inner_type}")

                if pa.types.is_string(inner_type):
                    print("SUCCESS: Inner elements are stored as optimized Arrow Strings.")

        print("=" * 50 + "\n")

        return batch

    # 4. 执行计算并强制指定 batch_format 为 pyarrow
    # 这样可以确保 Ray 不会将数据退化为 np.ndarray(object)
    results = ds.map_batches(
        inspect_worker_batch,
        batch_format="pyarrow"
    ).take_all()

    print(f"Final processed rows count: {len(results)}")


if __name__ == "__main__":
    main()
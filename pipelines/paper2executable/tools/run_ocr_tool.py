# run_ocr_tool.py
# PDF-to-Markdown converter using PaddleOCR PPStructureV3
# Copied from /home/yjh/new_flow/run_ocr_tool.py for self-containment.

import argparse
from pathlib import Path
import sys


def paddle_ocr_equation(pdf_path, output_dir, use_gpu: bool = False, gpu_id: int = 0):
    """Convert a PDF to Markdown using PaddleOCR PPStructureV3.

    Args:
        pdf_path: path to input PDF file
        output_dir: directory where markdown/images will be written
        use_gpu: whether to enable GPU acceleration.
        gpu_id: which GPU card to use (set via CUDA_VISIBLE_DEVICES).
    """
    import os
    from paddleocr import PPStructureV3

    output_path = Path(output_dir)
    if use_gpu:
        # Pin to the requested GPU card so we don't collide with other jobs
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        # Force CPU mode
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    pipeline = PPStructureV3()
    output = pipeline.predict(input=pdf_path)

    markdown_list = []
    markdown_images = []

    for res in output:
        md_info = res.markdown
        markdown_list.append(md_info)
        markdown_images.append(md_info.get("markdown_images", {}))

    markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

    mkd_file_path = output_path / f"{Path(pdf_path).stem}.md"
    mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(mkd_file_path, "w", encoding="utf-8") as f:
        f.write(markdown_texts)

    for item in markdown_images:
        if item:
            for path, image in item.items():
                file_path = output_path / path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(file_path)

    return str(mkd_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration for OCR")
    parser.add_argument("--gpu_id", type=int, default=0, help="Which GPU card to use (default 0)")
    args = parser.parse_args()

    try:
        generated_path = paddle_ocr_equation(args.pdf, args.output_dir, use_gpu=args.use_gpu, gpu_id=args.gpu_id)
        # Print result path with prefix marker so caller can parse it
        print(f"RESULT_PATH:{generated_path}")
    except Exception as e:
        print(f"Error during OCR: {e}", file=sys.stderr)
        sys.exit(1)

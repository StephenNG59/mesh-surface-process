import argparse
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
import tifffile


def to_rgb(img: Image.Image) -> Image.Image:
    """Ensure PIL image is RGB (3 channels)."""
    if img.mode == 'RGB':
        return img
    if img.mode in ('L', 'I;16', 'I', 'F'):
        # 灰度 -> 复制到3通道
        return Image.merge('RGB', (img.convert('L'), img.convert('L'), img.convert('L')))
    if img.mode == 'RGBA':
        return img.convert('RGB')
    # 其他模式统一转为RGB
    return img.convert('RGB')


def read_png_rgb(path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    im = Image.open(path)
    im = to_rgb(im)
    w, h = im.size
    arr = np.array(im)  # (H, W, 3), dtype typically uint8/uint16
    return arr, (w, h)


def resize_to(img_arr: np.ndarray, size_xy: Tuple[int, int], resample=Image.BILINEAR) -> np.ndarray:
    """Resize (H, W, C) numpy array to size_xy=(W, H)."""
    img = Image.fromarray(img_arr)
    img = img.resize(size_xy, resample=resample)
    return np.array(img)


def save_six_channel_tiff(out_path: Path, a_rgb: np.ndarray, b_rgb: np.ndarray):
    """
    a_rgb, b_rgb: (H, W, 3) arrays, same shape/dtype.
    Saves as (C, H, W) with 6 channels: [A.R, A.G, A.B, B.R, B.G, B.B]
    """
    if a_rgb.shape != b_rgb.shape:
        raise ValueError("Shapes of A and B must match before saving. Got %s vs %s" % (a_rgb.shape, b_rgb.shape))
    if a_rgb.dtype != b_rgb.dtype:
        # 统一到更高位深以避免截断
        dtype = np.promote_types(a_rgb.dtype, b_rgb.dtype)
        a_rgb = a_rgb.astype(dtype, copy=False)
        b_rgb = b_rgb.astype(dtype, copy=False)

    # stack to (H, W, 6)
    stacked = np.concatenate([a_rgb, b_rgb], axis=2)
    # transpose to (6, H, W)
    chw = np.transpose(stacked, (2, 0, 1))
    # 写入tiff，保持原dtype
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out_path, chw, photometric='minisblack')  # minisblack适合多通道数据


def find_pngs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.png") if p.is_file()])


def main():
    parser = argparse.ArgumentParser(description="Merge paired PNGs into 6-channel TIFFs for YOLO multispectral.")
    parser.add_argument("folder_a", type=str, help="第一组PNG所在目录（A）")
    parser.add_argument("folder_b", type=str, help="第二组PNG所在目录（B）")
    parser.add_argument("out_dir", type=str, help="输出tiff目录")
    parser.add_argument("--suffix", type=str, default="", help="输出文件名后缀（不含扩展名），如'_ms'")
    parser.add_argument("--resize", action="store_true", help="若A与B尺寸不同，则将B重采样到A的尺寸")
    parser.add_argument("--overwrite", action="store_true", help="若输出存在则覆盖")
    args = parser.parse_args()

    folder_a = Path(args.folder_a)
    folder_b = Path(args.folder_b)
    out_dir = Path(args.out_dir)
    suffix = args.suffix

    a_pngs = find_pngs(folder_a)
    if not a_pngs:
        print(f"[Error] 在 {folder_a} 未发现PNG文件。")
        return

    # 建立B的查找表（基于不含扩展名的文件名）
    b_index = {p.stem: p for p in find_pngs(folder_b)}

    n_total = len(a_pngs)
    n_done = 0
    n_missing = 0
    n_skipped = 0
    problems = []

    for a_path in a_pngs:
        stem = a_path.stem
        b_path = b_index.get(stem, None)
        if b_path is None:
            n_missing += 1
            problems.append(f"[缺失] {stem}.png 在B目录未找到")
            continue

        try:
            a_arr, (aw, ah) = read_png_rgb(a_path)
            b_arr, (bw, bh) = read_png_rgb(b_path)

            if (aw, ah) != (bw, bh):
                if args.resize:
                    b_arr = resize_to(b_arr, (aw, ah), resample=Image.BILINEAR)
                else:
                    n_skipped += 1
                    problems.append(f"[尺寸不匹配] {stem}.png A=({aw}x{ah}) B=({bw}x{bh})，已跳过（可用 --resize 解决）")
                    continue

            out_name = f"{stem}{suffix}.tiff" if suffix else f"{stem}.tiff"
            out_path = out_dir / out_name
            if out_path.exists() and not args.overwrite:
                n_skipped += 1
                problems.append(f"[已存在] {out_name}，使用 --overwrite 覆盖")
                continue

            save_six_channel_tiff(out_path, a_arr, b_arr)
            n_done += 1

        except Exception as e:
            n_skipped += 1
            problems.append(f"[异常] {stem}.png -> {e}")

    print(f"完成：匹配源(A)共 {n_total}，输出 {n_done}，缺失 {n_missing}，跳过 {n_skipped}")
    if problems:
        print("\n问题汇总：")
        for msg in problems:
            print(" - " + msg)


if __name__ == "__main__":
    main()
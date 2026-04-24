import os


def merge_split_zips():
    # 权重文件夹路径（改成你实际的 weights 文件夹路径）
    weights_dir = r"/\weights"

    # 找到所有 weights.zip.xxx 分卷文件
    parts = sorted([f for f in os.listdir(weights_dir) if f.startswith("weights.zip.")])
    if not parts:
        print("❌ 没找到分卷压缩包！请确认路径正确")
        return

    print(f"✅ 找到 {len(parts)} 个分卷：{parts}")

    # 合并所有分卷到一个临时 zip 文件
    temp_zip = os.path.join(weights_dir, "temp_weights.zip")
    with open(temp_zip, "wb") as outfile:
        for part in parts:
            part_path = os.path.join(weights_dir, part)
            with open(part_path, "rb") as infile:
                outfile.write(infile.read())
    print("✅ 分卷合并完成！生成 temp_weights.zip")

    # 解压 temp_weights.zip → 得到 best.pt
    import zipfile
    with zipfile.ZipFile(temp_zip, "r") as zip_ref:
        zip_ref.extractall(weights_dir)
    os.remove(temp_zip)  # 删除临时 zip 文件
    print("✅ 解压完成！weights 文件夹里现在有 best.pt 了")


if __name__ == "__main__":
    merge_split_zips()
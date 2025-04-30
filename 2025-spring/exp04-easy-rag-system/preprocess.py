import os
import json
from bs4 import BeautifulSoup
import re

def extract_text_and_title_from_html(html_filepath):
    """
    从指定的 HTML 文件中提取标题和正文文本。

    Args:
        html_filepath (str): HTML 文件的路径。

    Returns:
        tuple: (标题, 正文文本) 或 (None, None) 如果失败。
    """
    try:
        with open(html_filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'lxml') # 或者使用 'html.parser'

        # --- 提取标题 ---
        title_tag = soup.find('title')
        title_string = title_tag.string if title_tag else None
        # 确保 title_string 不为 None 才调用 strip()
        title = title_string.strip() if title_string else os.path.basename(html_filepath)
        title = title.replace('.html', '') # 清理标题

        # --- 定位正文内容 ---
        # 根据之前的讨论，优先查找 <content> 或特定 class
        content_tag = soup.find('content')
        if not content_tag:
            content_tag = soup.find('div', class_='rich_media_content') # 微信文章常见
        if not content_tag:
            content_tag = soup.find('article') # HTML5 语义标签
        if not content_tag:
            content_tag = soup.find('main') # HTML5 语义标签
        if not content_tag:
             content_tag = soup.find('body') # 最后尝试 body

        if content_tag:
            # 获取文本，尝试保留段落换行符
            text = content_tag.get_text(separator='\n', strip=True)
            # 移除多余的空行
            text = re.sub(r'\n\s*\n', '\n', text).strip()
            # 可选：进一步清理特定模式（如广告、页脚等）
            text = text.replace('阅读原文', '').strip()
            return title, text
        else:
            print(f"警告：在文件 {html_filepath} 中未找到明确的正文标签。")
            return title, None # 返回标题，但文本为 None

    except FileNotFoundError:
        print(f"错误：文件 {html_filepath} 未找到。")
        return None, None
    except Exception as e:
        print(f"处理文件 {html_filepath} 时出错: {e}")
        return None, None

def split_text(text, chunk_size=500, chunk_overlap=50):
    """
    将文本分割成指定大小的块，并带有重叠。

    Args:
        text (str): 要分割的文本。
        chunk_size (int): 每个块的目标字符数。
        chunk_overlap (int): 相邻块之间的重叠字符数。

    Returns:
        list[str]: 文本块列表。
    """
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if start >= len(text): # 避免无限循环（如果 overlap >= size）
             break
        # 如果最后一块太短，并且有重叠，可能导致重复添加，确保不重复
        if start < chunk_size and len(chunks)>1 and chunks[-1] == chunks[-2][chunk_size-chunk_overlap:]:
             chunks.pop() # 移除重复的小尾巴
             start = len(text) # 强制结束

    # 处理最后一块可能不足 chunk_size 的情况
    if start < len(text) and start > 0 : # 确保不是第一个块就小于size
         last_chunk = text[start-chunk_size+chunk_overlap:]
         if chunks and last_chunk != chunks[-1]: # 避免重复添加完全相同的最后一块
            # 检查是否和上一个块的尾部重复太多
            if not chunks[-1].endswith(last_chunk):
                 chunks.append(last_chunk)
         elif not chunks: # 如果是唯一一块且小于size
             chunks.append(last_chunk)


    # 更简洁的实现 (可能需要微调确保边界情况)
    # chunks = []
    # for i in range(0, len(text), chunk_size - chunk_overlap):
    #     chunk = text[i:i + chunk_size]
    #     if chunk: # 确保不添加空块
    #         chunks.append(chunk)
    # # 确保最后一部分被包含 (如果上面步长导致遗漏)
    # if chunks and len(text) > (len(chunks) -1) * (chunk_size - chunk_overlap) + chunk_size :
    #      last_start = (len(chunks) -1) * (chunk_size - chunk_overlap)
    #      final_chunk = text[last_start:]
    #      if final_chunk != chunks[-1]: # 避免重复
    #          # 可以考虑合并最后两个块如果最后一个太小，或直接添加
    #           if len(final_chunk) > chunk_overlap : # 避免添加太小的重叠部分
    #               chunks.append(final_chunk[overlap:]) # 只添加新的部分? 或者完整添加? 取决于需求
    #               # 简单起见，先完整添加
    #               chunks.append(text[last_start + chunk_size - chunk_overlap:])


    return [c.strip() for c in chunks if c.strip()] # 返回非空块

# --- 配置 ---
html_directory = './data/' # **** 修改为你的 HTML 文件夹路径 ****
output_json_path = './data/processed_data.json' # **** 输出 JSON 文件路径 ****
CHUNK_SIZE = 512  # 每个文本块的目标大小（字符数）
CHUNK_OVERLAP = 50 # 相邻文本块的重叠大小（字符数）

# --- 主处理逻辑 ---
all_data_for_milvus = []
file_count = 0
chunk_count = 0

print(f"开始处理目录 '{html_directory}' 中的 HTML 文件...")

# 确保输出目录存在
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

html_files = [f for f in os.listdir(html_directory) if f.endswith('.html')]
print(f"找到 {len(html_files)} 个 HTML 文件。")

for filename in html_files:
    filepath = os.path.join(html_directory, filename)
    print(f"  处理文件: {filename} ...")
    file_count += 1

    title, main_text = extract_text_and_title_from_html(filepath)

    if main_text:
        chunks = split_text(main_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"    提取到文本，分割成 {len(chunks)} 个块。")

        for i, chunk in enumerate(chunks):
            chunk_count += 1
            # 构建符合 milvus_utils.py 期望的字典结构
            milvus_entry = {
                "id": f"{filename}_{i}", # 创建一个唯一的 ID (文件名 + 块索引)
                "title": title or filename, # 使用提取的标题或文件名
                "abstract": chunk, # 将文本块放入 'abstract' 字段
                "source_file": filename, # 添加原始文件名以供参考
                "chunk_index": i
            }
            all_data_for_milvus.append(milvus_entry)
    else:
        print(f"    警告：未能从 {filename} 提取有效文本内容。")

print(f"\n处理完成。共处理 {file_count} 个文件，生成 {chunk_count} 个文本块。")

# --- 保存为 JSON ---
try:
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data_for_milvus, f, ensure_ascii=False, indent=4)
    print(f"结果已保存到: {output_json_path}")
except Exception as e:
    print(f"错误：无法写入 JSON 文件 {output_json_path}: {e}")

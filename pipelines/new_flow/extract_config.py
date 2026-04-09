#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shell Script Parser for Pipeline Configuration Extraction
精确提取 run_pipeline.py 命令中的配置信息并生成 YAML 配置文件
"""

import re
import sys
import yaml
import shlex
from pathlib import Path
from typing import List, Dict, Optional


def merge_continuation_lines(content: str) -> List[str]:
    """
    合并shell脚本中的续行符（反斜杠+换行），返回合并后的行列表
    """
    lines = content.splitlines()
    merged_lines = []
    current_line = []
    
    for line in lines:
        line = line.rstrip()
        # 跳过空行和纯注释行（但保留包含命令的注释行）
        if not line and not current_line:
            continue
            
        # 检查是否为续行（行尾有反斜杠，且反斜杠前无非空白字符）
        if line.endswith('\\'):
            # 移除尾部反斜杠及可能的空格
            line = line[:-1].rstrip()
            current_line.append(line)
        else:
            current_line.append(line)
            # 合并当前累积的行
            merged_lines.append(' '.join(current_line))
            current_line = []
    
    # 处理文件末尾的续行（理论上不应存在）
    if current_line:
        merged_lines.append(' '.join(current_line))
    
    return merged_lines


def extract_commands(lines: List[str]) -> List[str]:
    """
    从合并后的行中提取所有包含 run_pipeline.py 的命令（包括注释行中的命令）
    """
    commands = []
    for line in lines:
        # 跳过纯空行
        if not line.strip():
            continue
            
        # 保留包含 run_pipeline.py 的行（无论是否被注释）
        if 'run_pipeline.py' in line:
            commands.append(line)
    
    return commands


def parse_command_line(line: str) -> Dict[str, str]:
    """
    解析单行命令，提取参数字典
    处理行首注释符号，使用 shlex 安全解析带引号的参数
    """
    # 移除行首可能的注释符号和空格（仅当整行被注释时）
    # 保留引号内的 # 符号
    cleaned_line = re.sub(r'^\s*#*\s*', '', line)
    
    try:
        # 使用 shlex 安全分割命令（处理引号、空格等）
        tokens = shlex.split(cleaned_line)
    except ValueError as e:
        # 尝试修复常见的引号不匹配问题
        if "No closing quotation" in str(e):
            # 简单修复：移除行尾可能缺失的引号
            cleaned_line = cleaned_line.rstrip("'\"")
            try:
                tokens = shlex.split(cleaned_line)
            except:
                print(f"  ⚠ 警告: 无法解析命令行（引号问题）: {line[:60]}...", file=sys.stderr)
                return {}
        else:
            print(f"  ⚠ 警告: 无法解析命令行: {line[:60]}...", file=sys.stderr)
            return {}
    
    # 找到 run_pipeline.py 的位置
    try:
        pipeline_idx = next(i for i, token in enumerate(tokens) if 'run_pipeline.py' in token)
    except StopIteration:
        return {}
    
    # 提取 run_pipeline.py 之后的所有参数
    args = {}
    i = pipeline_idx + 1
    while i < len(tokens):
        token = tokens[i]
        if token.startswith('--'):
            # 处理 --key=value 格式
            if '=' in token:
                key, value = token[2:].split('=', 1)
                args[key] = value
                i += 1
            # 处理 --key value 格式
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith('--'):
                key = token[2:]
                value = tokens[i + 1]
                args[key] = value
                i += 2
            else:
                # 标志位参数（无值），跳过
                i += 1
        else:
            i += 1
    
    return args


def extract_python_path(command_str: str) -> Optional[str]:
    """
    从 --command 参数值中提取 Python 解释器路径
    例如: "/home/.../bin/python sim_code.py" -> "/home/.../bin/python"
    """
    if not command_str:
        return None
    
    try:
        # 安全分割 command 字符串（处理引号）
        parts = shlex.split(command_str)
        if parts and ('python' in Path(parts[0]).name.lower()):
            return parts[0]
    except:
        # 回退：使用正则提取第一个包含 python 的路径
        match = re.search(r'(["\']?)([^"\']*\b(?:python3?|pypy3?)\b[^"\']*)\1', command_str)
        if match:
            # 提取路径部分（到第一个空格前）
            path = match.group(2).split()[0]
            return path
    
    return None


def infer_name(tutorial_name: Optional[str], gt_code_path: Optional[str]) -> str:
    """
    智能推断 name 字段：
    1. 优先使用 tutorial_name，移除常见后缀 (_with_eval, _tutorial, _code 等)
    2. 其次从代码文件名提取（移除 _code.py 后缀）
    """
    # 从 tutorial_name 提取
    if tutorial_name:
        # 定义要移除的常见后缀（按优先级排序）
        suffixes = [
            '_with_eval', '_tutorial', '_code', '_claude_prompt', 
            '_prompt', '_clean_up', '_history', '_task1', '_task2', '_task3',
            '_simple_ring', '_quad_quasar', '_double_quasar', '_shapelets', '_host_decomp',
            '_admm', '_dl', '_sh_wfs', '_image_zernike', '_pnp_cassi', '_fpm_inr'
        ]
        name = tutorial_name
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        
        # 移除可能残留的下划线
        name = name.rstrip('_')
        if name:
            return name
    
    # 从代码文件名提取
    if gt_code_path:
        filename = Path(gt_code_path).stem  # 移除扩展名
        # 移除常见后缀
        for suffix in ['_code', '_main', '_script']:
            if filename.endswith(suffix):
                filename = filename[:-len(suffix)]
                break
        return filename
    
    return "unknown"


def process_script(filepath: str) -> List[Dict]:
    """
    主处理函数：解析脚本文件并生成配置条目列表
    """
    # 读取文件内容
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 - {filepath}", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"❌ 错误: 无法解码文件 {filepath}，尝试使用 latin-1 编码", file=sys.stderr)
        with open(filepath, 'r', encoding='latin-1') as f:
            content = f.read()
    
    # 合并续行符
    merged_lines = merge_continuation_lines(content)
    
    # 提取命令行
    command_lines = extract_commands(merged_lines)
    
    if not command_lines:
        print(f"⚠ 警告: 在 {filepath} 中未找到任何包含 run_pipeline.py 的命令", file=sys.stderr)
        return []
    
    print(f"🔍 找到 {len(command_lines)} 个 pipeline 命令，开始解析...\n")
    
    config_entries = []
    valid_count = 0
    
    for idx, line in enumerate(command_lines, 1):
        # 显示原始命令（截断过长的）
        display_line = line.strip()
        if len(display_line) > 80:
            display_line = display_line[:77] + "..."
        print(f"命令 {idx}: {display_line}")
        
        # 解析参数
        args = parse_command_line(line)
        
        # 提取必要字段
        gt_code_path = args.get('working_folder_file') or args.get('working_folder_file')
        working_folder = args.get('working_folder')
        command_arg = args.get('command')
        tutorial_name = args.get('tutorial_name') or args.get('tutorial_name')
        
        # 提取 Python 路径
        python_path = extract_python_path(command_arg) if command_arg else None
        
        # 推断 name
        name = infer_name(tutorial_name, gt_code_path)
        
        # 验证必要字段
        missing = []
        if not gt_code_path:
            missing.append("working_folder_file")
        if not working_folder:
            missing.append("working_folder")
        if not python_path:
            missing.append("command (python path)")
        
        if missing:
            print(f"  ⚠ 跳过: 缺少必要字段 - {', '.join(missing)}\n")
            continue
        
        # 构建配置条目
        entry = {
            'gt_code_path': gt_code_path,
            'name': name,
            'working_folder': working_folder,
            'python_path': python_path
        }
        
        config_entries.append(entry)
        valid_count += 1
        print(f"  ✓ 提取成功: name={name}, gt_code_path={gt_code_path}\n")
    
    print(f"✅ 成功提取 {valid_count}/{len(command_lines)} 个有效配置条目")
    return config_entries


def write_yaml_config(entries: List[Dict], output_path: str):
    """
    将配置条目写入 YAML 文件（使用块格式，更易读）
    """
    # 构建 YAML 内容（使用块格式）
    yaml_lines = [
        "# Auto-generated configuration from shell script",
        "# Extracted from run_pipeline.py commands",
        "# Format: YAML list of configuration entries",
        ""
    ]
    
    for entry in entries:
        yaml_lines.append("- gt_code_path: " + entry['gt_code_path'])
        yaml_lines.append("  name: " + entry['name'])
        yaml_lines.append("  working_folder: " + entry['working_folder'])
        yaml_lines.append("  python_path: " + entry['python_path'])
        yaml_lines.append("")  # 空行分隔
    
    yaml_content = '\n'.join(yaml_lines)
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n📄 配置文件已生成: {output_path}")
    print(f"   共 {len(entries)} 个配置条目")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='从 shell 脚本中提取 pipeline 配置生成 YAML 文件',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_script', help='输入的 shell 脚本文件路径')
    parser.add_argument('-o', '--output', default='config.yaml',
                        help='输出的 YAML 配置文件路径 (默认: config.yaml)')
    parser.add_argument('-f', '--force', action='store_true',
                        help='覆盖已存在的输出文件')
    
    args = parser.parse_args()
    
    # 检查输出文件是否存在
    if Path(args.output).exists() and not args.force:
        print(f"❌ 错误: 输出文件 {args.output} 已存在。使用 -f 参数强制覆盖", file=sys.stderr)
        sys.exit(1)
    
    # 处理脚本
    entries = process_script(args.input_script)
    
    if not entries:
        print("❌ 错误: 未提取到任何有效配置条目", file=sys.stderr)
        sys.exit(1)
    
    # 写入 YAML
    write_yaml_config(entries, args.output)
    
    # 显示预览
    print("\n📊 配置预览 (前3个条目):")
    for i, entry in enumerate(entries[:3], 1):
        print(f"  {i}. name: {entry['name']:<20} | path: {entry['gt_code_path']}")
    if len(entries) > 3:
        print(f"  ... 共 {len(entries)} 个条目")


if __name__ == '__main__':
    # 检查 pyyaml 依赖
    try:
        import yaml
    except ImportError:
        print("❌ 错误: 未安装 PyYAML 依赖", file=sys.stderr)
        print("   请运行: pip install pyyaml", file=sys.stderr)
        sys.exit(1)
    
    main()
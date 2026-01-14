#!/usr/bin/env python3
"""
Odoo Dataset Generator

This script parses Odoo addon modules to extract class and function information.
It outputs an Excel file with separate sheets for each module.
It can optionally use a local Ollama LLM to generate descriptive summaries for each function.

Usage:
    python3 generate_odoo_dataset.py /path/to/odoo /path/to/output.xlsx [--ollama-model model_name]
"""
import ast
import os
import sys
import argparse
import pandas as pd
import requests
import json
from tqdm import tqdm

def get_decorators(node):
    decorators = []
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name):
            decorators.append(f"@{decorator.id}")
        elif isinstance(decorator, ast.Attribute):
            parts = []
            curr = decorator
            while isinstance(curr, ast.Attribute):
                parts.append(curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                parts.append(curr.id)
            decorators.append(f"@{'.'.join(reversed(parts))}")
        elif isinstance(decorator, ast.Call):
            # Handle @api.depends('arg')
            func = decorator.func
            func_name = ""
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                parts = []
                curr = func
                while isinstance(curr, ast.Attribute):
                    parts.append(curr.attr)
                    curr = curr.value
                if isinstance(curr, ast.Name):
                    parts.append(curr.id)
                func_name = ".".join(reversed(parts))
            
            args = []
            for arg in decorator.args:
                if isinstance(arg, ast.Constant):
                    args.append(repr(arg.value))
            args_str = ", ".join(args)
            decorators.append(f"@{func_name}({args_str})")
            
    return decorators

def get_function_signature(node, decorators):
    # Reconstruct def foo(self, ...):
    args = []
    for arg in node.args.args:
        args.append(arg.arg)
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    
    sig = f"def {node.name}({', '.join(args)}):"
    if decorators:
        return "\n".join(decorators) + "\n    " + sig
    return sig

def generate_description_ollama(source_code, model):
    url = "http://localhost:11434/api/generate"
    prompt = f"""
    Analyze the following Odoo Python code (function/method) and provide a concise, high-level description of what it does.
    If the code has a docstring, use it as the primary source of truth but enhance it.
    Focus on the business logic or purpose.
    Return ONLY the description text, no preamble or markdown formatting.
    
    Code:
    ```python
    {source_code}
    ```
    """
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return f"[AI Generation Failed: {e}]"

def process_file(file_path, module_name, relative_path, data_list, ollama_model=None):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return

    # File Directory and Name
    file_dir = os.path.dirname(relative_path)
    file_name = os.path.basename(relative_path)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            model_name = ""
            
            # Try to find _name or _inherit
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            if target.id == "_name" and isinstance(item.value, ast.Constant):
                                model_name = item.value.value
                            elif target.id == "_inherit" and not model_name:
                                if isinstance(item.value, ast.Constant):
                                    model_name = item.value.value
                                elif isinstance(item.value, ast.List) and item.value.elts:
                                    if isinstance(item.value.elts[0], ast.Constant):
                                        model_name = item.value.elts[0].value

            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    decorators = get_decorators(item)
                    func_signature = get_function_signature(item, decorators)
                    docstring = ast.get_docstring(item) or ""
                    docstring = docstring.strip()
                    
                    # Extract source code for AI if model provided
                    source_segment = ast.get_source_segment(content, item) if ollama_model else None
                    
                    data_list.append({
                        "Module Name": module_name,
                        "File Directory": file_dir,
                        "File Name": file_name,
                        "model name": model_name,
                        "Class name": class_name,
                        "Function name": func_signature,
                        "Description": docstring,
                        "Source Code": source_segment # Temporary storage for batch processing
                    })

def main():
    parser = argparse.ArgumentParser(description="Generate Odoo Dataset with AI descriptions")
    parser.add_argument("odoo_path", help="Path to Odoo source code")
    parser.add_argument("output_file", help="Output Excel file path")
    parser.add_argument("--ollama-model", help="Ollama model to use (e.g., mistral, llama3)", default=None)
    parser.add_argument("--limit", type=int, help="Limit number of functions to process with AI (for testing)", default=0)
    
    args = parser.parse_args()

    odoo_path = args.odoo_path
    output_file = args.output_file
    ollama_model = args.ollama_model
    
    addons_path = os.path.join(odoo_path, "addons")
    if not os.path.exists(addons_path):
        addons_path = odoo_path

    print(f"Scanning {addons_path}...")
    
    data = []
    
    # Walk through directories
    modules = [d for d in os.listdir(addons_path) if os.path.isdir(os.path.join(addons_path, d))]
    modules.sort()
    
    for module in modules:
        module_path = os.path.join(addons_path, module)
        if not os.path.exists(os.path.join(module_path, "__init__.py")) and not os.path.exists(os.path.join(module_path, "__manifest__.py")):
            continue
            
        for root, dirs, files in os.walk(module_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, module_path)
                    process_file(full_path, module, rel_path, data, ollama_model)
    
    print(f"Collected {len(data)} records.")

    if ollama_model:
        print(f"Generating AI descriptions using {ollama_model}...")
        
        count = 0
        limit = args.limit
        
        # Use tqdm for progress bar
        for record in tqdm(data, desc="AI Processing"):
            if limit > 0 and count >= limit:
                break
                
            source_code = record.get("Source Code")
            # Skip if source code missing or description already exists (optional: override existing)
            # Strategy: if docstring exists, use it + AI. If no docstring, AI is crucial.
            # User request: "description has to be more descriptive". So we always generate.
            
            if source_code:
                new_desc = generate_description_ollama(source_code, ollama_model)
                if new_desc and not new_desc.startswith("[AI Generation Failed"):
                   record["Description"] = new_desc
                elif record["Description"]:
                   # Fallback to existing docstring if AI fails
                   pass
            
            count += 1

    # Remove Source Code column before saving
    for record in data:
        if "Source Code" in record:
            del record["Source Code"]

    print("Creating DataFrame...")
    df = pd.DataFrame(data)
    
    # Remove illegal characters from string columns
    def clean_text(text):
        if isinstance(text, str):
            return "".join(c for c in text if (0x20 <= ord(c) <= 0xD7FF) or c in ('\t', '\n', '\r') or (0xE000 <= ord(c) <= 0xFFFD) or (0x10000 <= ord(c) <= 0x10FFFF))
        return text

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(clean_text)

    # Write to Excel with separate sheets
    print("Writing to Excel file (this may take a while)...")
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            if df.empty:
                df.to_excel(writer, sheet_name="Sheet1", index=False)
            else:
                groups = df.groupby("Module Name")
                existing_sheet_names = set()
                
                for module_name, group_df in groups:
                    clean_name = str(module_name)
                    clean_name = "".join(c for c in clean_name if c not in ":\\/?*[]")
                    sheet_name = clean_name[:31]
                    
                    if sheet_name in existing_sheet_names:
                        base_name = sheet_name
                        counter = 1
                        while sheet_name in existing_sheet_names:
                            suffix = str(counter)
                            sheet_name = base_name[:31-len(suffix)] + suffix
                            counter += 1
                    
                    existing_sheet_names.add(sheet_name)
                    group_df.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        print(f"Error writing Excel file: {e}")
        sys.exit(1)

    print(f"Dataset finished: {output_file}")

if __name__ == "__main__":
    main()
